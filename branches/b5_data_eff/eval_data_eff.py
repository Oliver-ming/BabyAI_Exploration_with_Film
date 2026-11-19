"""支线 5 · 数据效率全量闭环评估与采样压缩倍数统计（复用冻结 `eval_closed_loop.evaluate`）。

包含：
  - 6 条完整扩展曲线（bc_naive / bc_step / bc_reg / bc_d4 / film_step / film_d4 × 5%..100%）
  - bc_d4_tta 与 film_d4_tta（f005/f010/f020 零额外训练开销 D4 群推理增强）
  - 20% 预算（900 ep）13 路机制消融（bc_z2 / bc_c4 / bc_wd / bc_bal / gru_small_step / b5_defrac 等）
  - 自动汇总 generalization_gap、3-seed 配对检验与样本压缩率（达成 100% BC30 所需最少样本量）

用法:
    OMP_NUM_THREADS=1 python branches/b5_data_eff/eval_data_eff.py \
        --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers 16 \
        --out reports/b5_data_eff_eval.json
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
sys.path.insert(0, str(ROOT / "branches" / "b5_data_eff"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from data_eff_policy import load_any_run, make_policy_fn  # noqa: E402
from data_eff_utils import FRACS, METHODS, frac_tag  # noqa: E402
from eval_closed_loop import evaluate, parse_seeds  # noqa: E402


def discover_eval_specs(runs_dir: Path) -> list[dict]:
    """收集全部待评估任务（去重物理评估，支持逻辑映射别名）。"""
    specs: list[dict] = []

    def add(run: str, logical_entries: list[tuple[str, float, str]], d4_tta: bool = False):
        if not Path(run).exists():
            return
        specs.append(
            {
                "run": run,
                "d4_tta": d4_tta,
                "aliases": logical_entries,  # [(method, frac, ftag), ...]
            }
        )

    # 1. 官方 100% 参考对照：b3_bc30 (Concat) & b2_film_nwd (FiLM)
    for s in (0, 1, 2):
        add(
            f"runs/b3_bc30_seed{s}",
            [("bc_naive", 1.0, "f100"), ("bc_step", 1.0, "f100")],
            d4_tta=False,
        )
        add(
            f"runs/b2_film_nwd_seed{s}",
            [("film_step", 1.0, "f100")],
            d4_tta=False,
        )

    # 2. 手册历史随机基线（20% / 50%）作为外部对齐核对
    for f, rname_f in [(0.2, "0.2"), (0.5, "0.5")]:
        ft = frac_tag(f)
        for s in (0, 1, 2):
            add(f"runs/b5_defrac{rname_f}_seed{s}", [("b5_defrac", f, ft)], d4_tta=False)

    # 3. 本支线 96 个新训练模型
    for m, mcfg in METHODS.items():
        frac_list = FRACS if mcfg["family"] == "scaling" else (0.20,)
        for f in frac_list:
            ft = frac_tag(f)
            for s in (0, 1, 2):
                rpath = f"runs/b5_{m}_{ft}_seed{s}"
                if not Path(rpath).exists():
                    continue
                add(rpath, [(m, f, ft)], d4_tta=False)
                # 对低/中数据上的 D4 模型追加零训练成本 8x TTA 推理评估
                if m in ("bc_d4", "film_d4") and f in (0.05, 0.10, 0.20):
                    add(rpath, [(f"{m}_tta", f, ft)], d4_tta=True)

    return specs


def eval_task(task: tuple[dict, str, list[int]]) -> list[dict]:
    spec, env_id, seeds = task
    torch.set_num_threads(1)
    run = spec["run"]
    d4_tta = spec["d4_tta"]
    model, cfg, _ = load_any_run(run)
    t0 = time.time()
    res = evaluate(make_policy_fn(model, d4_tta=d4_tta), env_id, seeds)
    res.pop("_trajectories", None)
    elapsed = round(time.time() - t0, 2)

    train_acc_at_best = cfg.get("result", {}).get("train_acc_at_best")
    val_acc = cfg.get("result", {}).get("best_val_acc")
    gen_gap = cfg.get("result", {}).get("generalization_gap")
    if train_acc_at_best is None and Path(run, "history.json").exists():
        # 为复用 bc30 / b2_film_nwd / defrac 补充 best_epoch 处的 train_acc
        hist = json.loads(Path(run, "history.json").read_text())
        be = cfg.get("result", {}).get("best_epoch")
        if be and 1 <= be <= len(hist):
            train_acc_at_best = hist[be - 1].get("train_acc")
            if train_acc_at_best is not None and val_acc is not None:
                gen_gap = round(float(train_acc_at_best) - float(val_acc), 6)

    out_rows = []
    for method, frac, ftag in spec["aliases"]:
        row = dict(res)
        row.update(
            {
                "run": run,
                "method": method,
                "frac": frac,
                "frac_tag": ftag,
                "d4_tta": d4_tta,
                "n_train_episodes": cfg.get("data", {}).get("n_train_episodes"),
                "n_raw_transitions": cfg.get("data", {}).get("n_raw_transitions")
                or cfg.get("data", {}).get("n_train_transitions"),
                "n_aug_transitions": cfg.get("data", {}).get("n_aug_transitions")
                or cfg.get("data", {}).get("n_train_transitions"),
                "train_seed": cfg.get("seed"),
                "val_action_acc": val_acc,
                "train_acc_at_best": train_acc_at_best,
                "generalization_gap": gen_gap,
                "best_epoch": cfg.get("result", {}).get("best_epoch"),
                "n_params": cfg.get("arch", {}).get("n_params"),
                "use_film": bool(cfg.get("arch", {}).get("use_film", False)) or cfg.get("branch") == "b2_film",
                "env_id": env_id,
                "eval_seconds": elapsed,
            }
        )
        out_rows.append(row)
    return out_rows


def agg_rows(rs: list[dict]) -> dict:
    succ = [r["closed_loop_success"] for r in rs]
    acc = [r["val_action_acc"] for r in rs if r["val_action_acc"] is not None]
    tr = [r["train_acc_at_best"] for r in rs if r["train_acc_at_best"] is not None]
    gap = [r["generalization_gap"] for r in rs if r["generalization_gap"] is not None]
    return {
        "n": len(rs),
        "frac": rs[0]["frac"],
        "frac_tag": rs[0]["frac_tag"],
        "n_episodes": rs[0]["n_train_episodes"],
        "success_mean": round(float(np.mean(succ)), 4),
        "success_sd": round(float(np.std(succ)), 4),
        "val_acc_mean": round(float(np.mean(acc)), 4) if acc else None,
        "val_acc_sd": round(float(np.std(acc)), 4) if acc else None,
        "train_acc_mean": round(float(np.mean(tr)), 4) if tr else None,
        "gen_gap_mean": round(float(np.mean(gap)), 4) if gap else None,
        "mean_return": round(float(np.mean([r["mean_return"] for r in rs])), 4),
        "median_len_success": round(float(np.mean([r["median_len_success"] for r in rs])), 2),
        "timeouts_mean": round(float(np.mean([r["n_timeout"] for r in rs])), 2),
        "n_params": rs[0]["n_params"],
    }


def paired_diff(lhs: list[dict], rhs: list[dict]) -> dict | None:
    by_l = {r["train_seed"]: r for r in lhs if r.get("train_seed") is not None}
    by_r = {r["train_seed"]: r for r in rhs if r.get("train_seed") is not None}
    common = sorted(set(by_l) & set(by_r))
    if not common:
        return None
    deltas, val_deltas, rows = [], [], []
    for s in common:
        d = round(by_l[s]["closed_loop_success"] - by_r[s]["closed_loop_success"], 6)
        deltas.append(d)
        dv = None
        if by_l[s].get("val_action_acc") is not None and by_r[s].get("val_action_acc") is not None:
            dv = round(by_l[s]["val_action_acc"] - by_r[s]["val_action_acc"], 6)
            val_deltas.append(dv)
        rows.append({"train_seed": s, "lhs": by_l[s]["closed_loop_success"], "rhs": by_r[s]["closed_loop_success"], "delta": d})
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
    ap.add_argument("--env-id", default="BabyAI-GoToLocalS6N3-v0")
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="reports/b5_data_eff_eval.json")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    specs = discover_eval_specs(ROOT / "runs")
    tasks = [(sp, args.env_id, seeds) for sp in specs]
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            nested = list(ex.map(eval_task, tasks))
    else:
        nested = [eval_task(t) for t in tasks]

    rows = [r for sub in nested for r in sub]
    buckets: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        buckets[(r["method"], r["frac_tag"])].append(r)

    scaling_curves: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for (m, ft), rs in sorted(buckets.items()):
        scaling_curves[m][ft] = agg_rows(rs)

    f020_ablations = {m: dict(d) for m, d in scaling_curves.items() if "f020" in d}
    f020_ablations = {m: d["f020"] for m, d in f020_ablations.items()}

    # 3-seed 配对比较：各比例组内比较 + 对比 100% BC30 / 100% FiLM
    paired: dict[str, dict] = {}
    ref_bc100 = buckets.get(("bc_step", "f100"), [])
    ref_fl100 = buckets.get(("film_step", "f100"), [])

    for ft in ("f005", "f010", "f020", "f050", "f100"):
        for name, lhs_m, rhs_m in [
            ("bc_step_vs_bc_naive", "bc_step", "bc_naive"),
            ("bc_reg_vs_bc_step", "bc_reg", "bc_step"),
            ("bc_d4_vs_bc_step", "bc_d4", "bc_step"),
            ("bc_d4_tta_vs_bc_d4", "bc_d4_tta", "bc_d4"),
            ("film_step_vs_bc_step", "film_step", "bc_step"),
            ("film_d4_vs_film_step", "film_d4", "film_step"),
            ("film_d4_tta_vs_film_d4", "film_d4_tta", "film_d4"),
        ]:
            if (lhs_m, ft) in buckets and (rhs_m, ft) in buckets and lhs_m != rhs_m:
                pd = paired_diff(buckets[(lhs_m, ft)], buckets[(rhs_m, ft)])
                if pd:
                    paired[f"{ft}_{name}"] = pd

        for m in ("bc_step", "bc_d4", "bc_d4_tta", "film_step", "film_d4", "film_d4_tta"):
            if (m, ft) in buckets and ref_bc100 and not (m == "bc_step" and ft == "f100"):
                pd = paired_diff(buckets[(m, ft)], ref_bc100)
                if pd:
                    paired[f"{m}_{ft}_vs_bc30_100"] = pd

    for m in ("bc_reg", "bc_z2", "bc_c4", "bc_d4", "bc_d4_tta", "bc_wd", "bc_bal", "gru_small_step", "film_step", "film_d4", "film_d4_tta"):
        if (m, "f020") in buckets and ("bc_step", "f020") in buckets:
            pd = paired_diff(buckets[(m, "f020")], buckets[("bc_step", "f020")])
            if pd:
                paired[f"f020_{m}_vs_bc_step"] = pd

    bc100_succ = scaling_curves.get("bc_step", {}).get("f100", {}).get("success_mean", 0.0)
    milestones = {"bc30_100pct_target": bc100_succ, "match_or_beat_bc30_100": [], "reach_0900": [], "reach_0940": []}
    for m, fdict in sorted(scaling_curves.items()):
        for ft in ("f005", "f010", "f020", "f050", "f100"):
            if ft in fdict:
                sm = fdict[ft]["success_mean"]
                nep = fdict[ft]["n_episodes"]
                if sm >= bc100_succ:
                    milestones["match_or_beat_bc30_100"].append({"method": m, "frac": ft, "n_episodes": nep, "compression": round(4500 / nep, 1), "success_mean": sm})
                if sm >= 0.900:
                    milestones["reach_0900"].append({"method": m, "frac": ft, "n_episodes": nep, "compression": round(4500 / nep, 1), "success_mean": sm})
                if sm >= 0.940:
                    milestones["reach_0940"].append({"method": m, "frac": ft, "n_episodes": nep, "compression": round(4500 / nep, 1), "success_mean": sm})

    summary = {
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "runs": rows,
        "scaling_curves": dict(scaling_curves),
        "f020_ablations": f020_ablations,
        "paired_comparisons": paired,
        "compression_milestones": milestones,
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[{args.env_id}] {'method':15s}  {'5%(225)':>13s}  {'10%(450)':>13s}  {'20%(900)':>13s}  {'50%(2250)':>13s}  {'100%(4500)':>13s}")
    for m in ("bc_naive", "bc_step", "bc_reg", "bc_d4", "bc_d4_tta", "film_step", "film_d4", "film_d4_tta"):
        if m not in scaling_curves:
            continue
        cells = []
        for ft in ("f005", "f010", "f020", "f050", "f100"):
            if ft in scaling_curves[m]:
                a = scaling_curves[m][ft]
                cells.append(f"{a['success_mean']:.3f}±{a['success_sd']:.3f}")
            else:
                cells.append(f"{'—':^13s}")
        print(f"  {m:15s}  " + "  ".join(f"{c:>13s}" for c in cells))
    print(f"\n[20% Ablation] {'method':15s} {'succ':>15s} {'val_acc':>8s} {'tr_acc':>8s} {'gap':>8s}")
    for m, a in sorted(f020_ablations.items(), key=lambda kv: -kv[1]["success_mean"]):
        print(
            f"  {m:15s} {a['success_mean']:7.3f}±{a['success_sd']:.3f} "
            f"{(a['val_acc_mean'] or 0):8.4f} {(a['train_acc_mean'] or 0):8.4f} {(a['gen_gap_mean'] or 0):+8.4f}"
        )
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
