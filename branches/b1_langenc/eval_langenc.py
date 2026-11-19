"""支线 1 · 闭环评估（口径完全复用冻结的 `eval_closed_loop.evaluate`）。

审计修复：
  1. 强制单线程 OpenMP/PyTorch + ProcessPoolExecutor 并行评估，避免多核自旋争抢
  2. 将 b3_bc30 / b2_film_nwd 等跨支线对照独立成组（bc30 / b2_film_nwd），
     绝不混入 B1 的 mean（消除旧版 n=6 / S7N4 漏掉 b1_mean 的 bucket 混叠 bug）
  3. 增加 3/3 seed 配对差值检验（对比 bc30、b1 mean、b2_film_nwd、以及编码器配对对照）

用法:
    OMP_NUM_THREADS=1 python branches/b1_langenc/eval_langenc.py \
        --runs runs/b1_*_seed* --baseline-runs runs/b3_bc30_seed* runs/b2_film_nwd_seed* \
        --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers 12 \
        --out reports/b1_langenc_eval.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from eval_closed_loop import evaluate, parse_seeds  # noqa: E402
from lang_policy import load_any_run, make_policy_fn  # noqa: E402


def group_name(run: str, cfg: dict) -> str:
    """清晰区分各实验组，绝不把跨支线运行混进 B1 的 mean 组。"""
    branch = cfg.get("branch")
    raw = cfg.get("arch", {}).get("variant")
    name = Path(run).name
    if branch == "b3_dagger" or name.startswith("b3_bc30"):
        return "bc30"
    if branch is None:
        return "baseline"
    if branch == "b2_film":
        is_nwd = "_nwd_" in name or (
            cfg.get("optim", {}).get("weight_decay", 0.0) == 0.0
            and cfg.get("optim", {}).get("film_l2", 0.0) == 0.0
            and raw in ("film", "concat")
        )
        return f"b2_{raw}_nwd" if is_nwd else f"b2_{raw}"
    if branch == "b1_langenc":
        return raw or "mean"
    return raw or branch or "unknown"


def eval_one_task(task: tuple[str, str, list[int], str]) -> dict:
    run, env_id, seeds, role = task
    torch.set_num_threads(1)
    model, cfg, _ = load_any_run(run)
    grp = group_name(run, cfg)
    t0 = time.time()
    res = evaluate(make_policy_fn(model), env_id, seeds)
    res.pop("_trajectories", None)
    res.update(
        {
            "run": run,
            "role": role,
            "group": grp,
            "variant": grp,
            "raw_variant": cfg.get("arch", {}).get("variant"),
            "branch": cfg.get("branch", "baseline"),
            "train_seed": cfg.get("seed"),
            "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
            "best_epoch": cfg.get("result", {}).get("best_epoch"),
            "n_params": cfg.get("arch", {}).get("n_params"),
            "n_lang_params": cfg.get("arch", {}).get("n_lang_params"),
            "weight_decay": cfg.get("optim", {}).get("weight_decay"),
            "film_l2": cfg.get("optim", {}).get("film_l2"),
            "env_id": env_id,
            "eval_seconds": round(time.time() - t0, 2),
        }
    )
    return res


def paired_diff(target_runs: list[dict], ref_runs: list[dict]) -> dict | None:
    by_seed_t = {r["train_seed"]: r for r in target_runs if r.get("train_seed") is not None}
    by_seed_r = {r["train_seed"]: r for r in ref_runs if r.get("train_seed") is not None}
    common_seeds = sorted(set(by_seed_t) & set(by_seed_r))
    if not common_seeds:
        return None
    deltas = []
    val_deltas = []
    rows = []
    for s in common_seeds:
        tr = by_seed_t[s]
        rr = by_seed_r[s]
        d = round(tr["closed_loop_success"] - rr["closed_loop_success"], 6)
        deltas.append(d)
        dv = None
        if tr.get("val_action_acc") is not None and rr.get("val_action_acc") is not None:
            dv = round(tr["val_action_acc"] - rr["val_action_acc"], 6)
            val_deltas.append(dv)
        rows.append(
            {
                "train_seed": s,
                "target_success": tr["closed_loop_success"],
                "ref_success": rr["closed_loop_success"],
                "delta": d,
                "delta_val_acc": dv,
            }
        )
    return {
        "n_pairs": len(common_seeds),
        "delta_mean": round(float(np.mean(deltas)), 4),
        "delta_sd": round(float(np.std(deltas)), 4),
        "n_positive": sum(1 for d in deltas if d > 0),
        "all_positive": all(d > 0 for d in deltas),
        "delta_val_acc_mean": round(float(np.mean(val_deltas)), 4) if val_deltas else None,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline-runs", nargs="*", default=[])
    ap.add_argument("--env-id", default="BabyAI-GoToLocalS6N3-v0")
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="reports/b1_langenc_eval.json")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    # 按 run 去重（保持传入顺序），防止重复统计
    seen = set()
    tasks = []
    for r in args.runs:
        if r not in seen:
            seen.add(r)
            tasks.append((r, args.env_id, seeds, "eval"))
    for r in args.baseline_runs:
        if r not in seen:
            seen.add(r)
            tasks.append((r, args.env_id, seeds, "ref"))

    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            all_results = list(ex.map(eval_one_task, tasks))
    else:
        all_results = [eval_one_task(t) for t in tasks]

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for r in all_results:
        buckets[r["group"]].append(r)

    def agg(rs: list[dict]) -> dict:
        succ = [r["closed_loop_success"] for r in rs]
        acc = [r["val_action_acc"] for r in rs if r["val_action_acc"] is not None]
        lps = [r["n_lang_params"] for r in rs if r["n_lang_params"] is not None]
        return {
            "n": len(rs),
            "success_mean": round(float(np.mean(succ)), 4),
            "success_sd": round(float(np.std(succ)), 4),
            "val_acc_mean": round(float(np.mean(acc)), 4) if acc else None,
            "val_acc_sd": round(float(np.std(acc)), 4) if acc else None,
            "mean_return": round(float(np.mean([r["mean_return"] for r in rs])), 4),
            "median_len_success": round(float(np.mean([r["median_len_success"] for r in rs])), 2),
            "mean_len_all": round(float(np.mean([r["mean_len_all"] for r in rs])), 2),
            "timeouts_mean": round(float(np.mean([r["n_timeout"] for r in rs])), 2),
            "n_params": rs[0]["n_params"],
            "n_lang_params": lps[0] if lps else None,
        }

    by_variant = {v: agg(rs) for v, rs in buckets.items()}
    ref_b1_mean = buckets.get("mean", [])
    ref_bc30 = buckets.get("bc30", [])
    ref_film_nwd = buckets.get("b2_film_nwd", [])

    for v, a in by_variant.items():
        rs = buckets[v]
        if ref_b1_mean and v != "mean":
            pd = paired_diff(rs, ref_b1_mean)
            a["delta_success_vs_mean"] = pd["delta_mean"] if pd else round(a["success_mean"] - by_variant["mean"]["success_mean"], 4)
            a["paired_vs_mean"] = pd
        if ref_bc30 and v != "bc30":
            pd = paired_diff(rs, ref_bc30)
            a["delta_success_vs_bc30"] = pd["delta_mean"] if pd else round(a["success_mean"] - by_variant["bc30"]["success_mean"], 4)
            a["paired_vs_bc30"] = pd
        if ref_film_nwd and v != "b2_film_nwd":
            pd = paired_diff(rs, ref_film_nwd)
            a["delta_success_vs_film_nwd"] = pd["delta_mean"] if pd else round(a["success_mean"] - by_variant["b2_film_nwd"]["success_mean"], 4)
            a["paired_vs_film_nwd"] = pd

    key_pairs = [
        ("dan_vs_mean", "dan", "mean"),
        ("pos_mlp_vs_dan", "pos_mlp", "dan"),
        ("gru_small_vs_dan", "gru_small", "dan"),
        ("gru_small_vs_pos_mlp", "gru_small", "pos_mlp"),
        ("gru_vs_mean_wide", "gru", "mean_wide"),
        ("transformer_vs_gru_small", "transformer", "gru_small"),
        ("film_gru_small_vs_b2_film_nwd", "film_gru_small", "b2_film_nwd"),
        ("film_gru_vs_b2_film_nwd", "film_gru", "b2_film_nwd"),
    ]
    paired_comparisons = {}
    for label, lhs, rhs in key_pairs:
        if lhs in buckets and rhs in buckets:
            pd = paired_diff(buckets[lhs], buckets[rhs])
            if pd:
                paired_comparisons[label] = pd

    summary = {
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "runs": all_results,
        "by_variant": by_variant,
        "paired_comparisons": paired_comparisons,
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"{'run':32s} {'group':15s} {'acc':>6s} {'succ':>6s} {'ret':>6s} {'tout':>5s} {'params':>9s}")
    for r in all_results:
        print(
            f"{Path(r['run']).name:32s} {r['group']:15s} "
            f"{(r['val_action_acc'] or 0):6.3f} {r['closed_loop_success']:6.3f} "
            f"{r['mean_return']:6.3f} {r['n_timeout']:5d} {(r['n_params'] or 0):9d}"
        )
    print()
    print(f"{'group':15s} {'n':>2s} {'success':>16s} {'val_acc':>16s} {'Δvs_bc30':>9s} {'Δvs_film':>9s} {'lang_p':>7s}")
    for v, a in sorted(by_variant.items()):
        dbc = a.get("delta_success_vs_bc30")
        dfl = a.get("delta_success_vs_film_nwd")
        lp = a["n_lang_params"] if a["n_lang_params"] is not None else -1
        print(
            f"{v:15s} {a['n']:2d} {a['success_mean']:8.3f}±{a['success_sd']:.3f} "
            f"{(a['val_acc_mean'] or 0):8.3f}±{(a['val_acc_sd'] or 0):.3f} "
            f"{('—' if dbc is None else f'{dbc:+.3f}'):>9s} "
            f"{('—' if dfl is None else f'{dfl:+.3f}'):>9s} {lp:7d}"
        )
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
