"""支线 4 · 闭环评估与 5 级环境难度谱评估（严格复用冻结 `eval_closed_loop.evaluate`）。

功能：
  1. 单环境闭环评估（S6N3 test 或 S7N4 test_s7），严格按 group_name 隔离对照，
     自动计算 3/3 seed 配对差值（vs bc30 / b2_film_nwd / 课程排序 vs 均匀混合）。
  2. --spectrum 模式：在 S5N2 → S6N3 → S6N4 → S7N4(未见7×7) → S8N6 五级梯度上评估跨尺寸与高干扰物泛化谱。

用法:
    OMP_NUM_THREADS=1 python branches/b4_curriculum/eval_curriculum.py \
        --runs runs/b4_*_seed* --baseline-runs runs/b3_bc30_seed* runs/b2_film_nwd_seed* \
        --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers 15 \
        --out reports/b4_curriculum_eval.json
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
sys.path.insert(0, str(ROOT / "branches" / "b4_curriculum"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from curriculum_policy import load_any_run, make_policy_fn  # noqa: E402
from eval_closed_loop import evaluate, parse_seeds  # noqa: E402

SPECTRUM_ENVS = [
    ("S5N2", "BabyAI-GoToLocalS5N2-v0", "test"),
    ("S6N3", "BabyAI-GoToLocalS6N3-v0", "test"),
    ("S6N4", "BabyAI-GoToLocalS6N4-v0", "test"),
    ("S7N4", "BabyAI-GoToLocalS7N4-v0", "test_s7"),
    ("S8N6", "BabyAI-GoToLocalS8N6-v0", "test"),
]

KEY_PAIRS = [
    ("pace_easy_vs_pace_rand", "pace_easy", "pace_rand"),
    ("pace_hard_vs_pace_easy", "pace_hard", "pace_easy"),
    ("env_easy_cumul_vs_env_easy_mix", "env_easy_cumul", "env_easy_mix"),
    ("env_easy_stage_vs_env_easy_mix", "env_easy_stage", "env_easy_mix"),
    ("env_s6n4_prog_vs_env_s6n4_mix", "env_s6n4_prog", "env_s6n4_mix"),
    ("env_multiscale_prog_vs_env_multiscale_mix", "env_multiscale_prog", "env_multiscale_mix"),
    ("film_multiscale_prog_vs_film_multiscale_mix", "film_multiscale_prog", "film_multiscale_mix"),
    ("film_easy_stage_vs_b2_film_nwd", "film_easy_stage", "b2_film_nwd"),
    ("film_multiscale_prog_vs_b2_film_nwd", "film_multiscale_prog", "b2_film_nwd"),
    ("film_multiscale_mix_vs_b2_film_nwd", "film_multiscale_mix", "b2_film_nwd"),
]


def group_name(run: str, cfg: dict) -> str:
    branch = cfg.get("branch")
    name = Path(run).name
    if branch == "b3_dagger" or name.startswith("b3_bc30"):
        return "bc30"
    if branch == "b4_curriculum":
        return cfg.get("variant") or cfg.get("arch", {}).get("variant") or "unknown"
    if branch == "b2_film":
        raw = cfg.get("arch", {}).get("variant", "film")
        is_nwd = "_nwd_" in name or (
            cfg.get("optim", {}).get("weight_decay", 0.0) == 0.0
            and cfg.get("optim", {}).get("film_l2", 0.0) == 0.0
            and raw in ("film", "concat")
        )
        return f"b2_{raw}_nwd" if is_nwd else f"b2_{raw}"
    return "baseline"


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
            "branch": cfg.get("branch", "baseline"),
            "train_seed": cfg.get("seed"),
            "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
            "best_epoch": cfg.get("result", {}).get("best_epoch"),
            "n_params": cfg.get("arch", {}).get("n_params"),
            "use_film": bool(cfg.get("arch", {}).get("use_film", False)),
            "env_id": env_id,
            "eval_seconds": round(time.time() - t0, 2),
        }
    )
    return res


def eval_spectrum_run_task(task: tuple[str, str]) -> dict:
    run, role = task
    torch.set_num_threads(1)
    model, cfg, _ = load_any_run(run)
    grp = group_name(run, cfg)
    policy = make_policy_fn(model)
    per_env = {}
    t0 = time.time()
    for tag, eid, seed_spec in SPECTRUM_ENVS:
        seeds = parse_seeds(seed_spec)
        res = evaluate(policy, eid, seeds)
        res.pop("_trajectories", None)
        per_env[tag] = {
            "env_id": eid,
            "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
            "closed_loop_success": res["closed_loop_success"],
            "mean_return": res["mean_return"],
            "median_len_success": res["median_len_success"],
            "n_timeout": res["n_timeout"],
        }
    return {
        "run": run,
        "role": role,
        "group": grp,
        "train_seed": cfg.get("seed"),
        "use_film": bool(cfg.get("arch", {}).get("use_film", False)),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "eval_seconds": round(time.time() - t0, 2),
        "spectrum": per_env,
    }


def paired_diff(target_runs: list[dict], ref_runs: list[dict], key_fn=None) -> dict | None:
    if key_fn is None:
        key_fn = lambda r: r["closed_loop_success"]  # noqa: E731
    by_seed_t = {r["train_seed"]: r for r in target_runs if r.get("train_seed") is not None}
    by_seed_r = {r["train_seed"]: r for r in ref_runs if r.get("train_seed") is not None}
    common = sorted(set(by_seed_t) & set(by_seed_r))
    if not common:
        return None
    deltas, val_deltas, rows = [], [], []
    for s in common:
        tr, rr = by_seed_t[s], by_seed_r[s]
        d = round(float(key_fn(tr)) - float(key_fn(rr)), 6)
        deltas.append(d)
        dv = None
        if tr.get("val_action_acc") is not None and rr.get("val_action_acc") is not None:
            dv = round(tr["val_action_acc"] - rr["val_action_acc"], 6)
            val_deltas.append(dv)
        rows.append(
            {
                "train_seed": s,
                "target": float(key_fn(tr)),
                "ref": float(key_fn(rr)),
                "delta": d,
                "delta_val_acc": dv,
            }
        )
    return {
        "n_pairs": len(common),
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
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default="reports/b4_curriculum_eval.json")
    ap.add_argument("--spectrum", action="store_true",
                    help="在 S5N2/S6N3/S6N4/S7N4/S8N6 五级梯度上跑全谱评估")
    ap.add_argument("--spectrum-out", default="reports/b4_curriculum_spectrum.json")
    args = ap.parse_args()

    seen = set()
    ordered_runs: list[tuple[str, str]] = []
    for r in args.runs:
        if r not in seen:
            seen.add(r)
            ordered_runs.append((r, "eval"))
    for r in args.baseline_runs:
        if r not in seen:
            seen.add(r)
            ordered_runs.append((r, "ref"))

    if args.spectrum:
        if args.workers > 1 and len(ordered_runs) > 1:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(ordered_runs))) as ex:
                sresults = list(ex.map(eval_spectrum_run_task, ordered_runs))
        else:
            sresults = [eval_spectrum_run_task(t) for t in ordered_runs]

        sbuckets: dict[str, list[dict]] = collections.defaultdict(list)
        for r in sresults:
            sbuckets[r["group"]].append(r)

        sby = {}
        for g, rs in sbuckets.items():
            row = {"n": len(rs), "use_film": rs[0]["use_film"]}
            all_vals = []
            for tag, eid, _ in SPECTRUM_ENVS:
                vals = [x["spectrum"][tag]["closed_loop_success"] for x in rs]
                all_vals.extend(vals)
                row[tag] = {
                    "env_id": eid,
                    "success_mean": round(float(np.mean(vals)), 4),
                    "success_sd": round(float(np.std(vals)), 4),
                }
            # 宏平均跨五环境成功率
            row["macro_mean"] = round(float(np.mean([row[tag]["success_mean"] for tag, _, _ in SPECTRUM_ENVS])), 4)
            sby[g] = row

        spaired = {}
        for tag, _, _ in SPECTRUM_ENVS:
            spaired[tag] = {}
            for label, lhs, rhs in KEY_PAIRS:
                if lhs in sbuckets and rhs in sbuckets:
                    pd = paired_diff(
                        sbuckets[lhs],
                        sbuckets[rhs],
                        key_fn=lambda r, t=tag: r["spectrum"][t]["closed_loop_success"],
                    )
                    if pd:
                        spaired[tag][label] = pd

        spec_summary = {
            "envs": [{"tag": t, "env_id": e, "seeds": s} for t, e, s in SPECTRUM_ENVS],
            "runs": sresults,
            "by_variant": sby,
            "paired_by_env": spaired,
        }
        sp = Path(args.spectrum_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(spec_summary, indent=2, ensure_ascii=False))
        print(f"\n[5-Env Spectrum] {'group':22s} {'S5N2':>7s} {'S6N3':>7s} {'S6N4':>7s} {'S7N4':>7s} {'S8N6':>7s} {'Macro':>7s}")
        for g, a in sorted(sby.items()):
            print(
                f"                 {g:22s} "
                + " ".join(f"{a[t]['success_mean']:7.3f}" for t, _, _ in SPECTRUM_ENVS)
                + f" {a['macro_mean']:7.3f}"
            )
        print(f"wrote {sp}")
        return

    seeds = parse_seeds(args.seeds)
    tasks = [(r, args.env_id, seeds, role) for r, role in ordered_runs]
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
        return {
            "n": len(rs),
            "use_film": rs[0]["use_film"],
            "success_mean": round(float(np.mean(succ)), 4),
            "success_sd": round(float(np.std(succ)), 4),
            "val_acc_mean": round(float(np.mean(acc)), 4) if acc else None,
            "val_acc_sd": round(float(np.std(acc)), 4) if acc else None,
            "mean_return": round(float(np.mean([r["mean_return"] for r in rs])), 4),
            "median_len_success": round(float(np.mean([r["median_len_success"] for r in rs])), 2),
            "mean_len_all": round(float(np.mean([r["mean_len_all"] for r in rs])), 2),
            "timeouts_mean": round(float(np.mean([r["n_timeout"] for r in rs])), 2),
            "n_params": rs[0]["n_params"],
        }

    by_variant = {v: agg(rs) for v, rs in buckets.items()}
    ref_bc30 = buckets.get("bc30", [])
    ref_film = buckets.get("b2_film_nwd", [])

    for v, a in by_variant.items():
        rs = buckets[v]
        if ref_bc30 and v != "bc30":
            pd = paired_diff(rs, ref_bc30)
            a["delta_success_vs_bc30"] = pd["delta_mean"] if pd else round(a["success_mean"] - by_variant["bc30"]["success_mean"], 4)
            a["paired_vs_bc30"] = pd
        if ref_film and v != "b2_film_nwd":
            pd = paired_diff(rs, ref_film)
            a["delta_success_vs_film_nwd"] = pd["delta_mean"] if pd else round(a["success_mean"] - by_variant["b2_film_nwd"]["success_mean"], 4)
            a["paired_vs_film_nwd"] = pd

    paired_comparisons = {}
    for label, lhs, rhs in KEY_PAIRS:
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

    print(f"\n[{args.env_id}] {'group':22s} {'n':>2s} {'success':>16s} {'val_acc':>16s} {'Δvs_bc30':>9s} {'Δvs_film':>9s}")
    for v, a in sorted(by_variant.items()):
        dbc = a.get("delta_success_vs_bc30")
        dfl = a.get("delta_success_vs_film_nwd")
        print(
            f"{' ':17s}{v:22s} {a['n']:2d} {a['success_mean']:8.3f}±{a['success_sd']:.3f} "
            f"{(a['val_acc_mean'] or 0):8.3f}±{(a['val_acc_sd'] or 0):.3f} "
            f"{('—' if dbc is None else f'{dbc:+.3f}'):>9s} "
            f"{('—' if dfl is None else f'{dfl:+.3f}'):>9s}"
        )
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
