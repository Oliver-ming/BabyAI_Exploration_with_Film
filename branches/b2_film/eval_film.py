"""支线 2 · 闭环评估（只做模型包装，评估口径完全复用冻结的 `eval_closed_loop.evaluate`）。

用法:
    python branches/b2_film/eval_film.py \
        --runs runs/b2_film_seed0 runs/b2_film_seed1 runs/b2_film_seed2 \
        --baseline-runs runs/b3_bc30_seed0 runs/b3_bc30_seed1 runs/b3_bc30_seed2 \
        --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --out reports/b2_film_eval.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from film_policy import load_any_run, make_policy_fn  # noqa: E402
from eval_closed_loop import evaluate, parse_seeds  # noqa: E402


def eval_one(run: str, env_id: str, seeds: list[int]) -> dict:
    model, cfg, _ = load_any_run(run)
    t0 = time.time()
    res = evaluate(make_policy_fn(model), env_id, seeds)
    res.pop("_trajectories", None)
    res.update(
        {
            "run": run,
            "variant": cfg.get("arch", {}).get("variant", "concat"),
            "branch": cfg.get("branch", "baseline"),
            "train_seed": cfg.get("seed"),
            "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
            "best_epoch": cfg.get("result", {}).get("best_epoch"),
            "n_params": cfg.get("arch", {}).get("n_params"),
            "weight_decay": cfg.get("optim", {}).get("weight_decay"),
            "_weight_decay": cfg.get("optim", {}).get("weight_decay"),
            "film_l2": cfg.get("optim", {}).get("film_l2"),
            "env_id": env_id,
            "eval_seconds": round(time.time() - t0, 2),
        }
    )
    return res


def group_key(r: dict) -> str:
    """分组键：variant + weight-decay 档位（不同 wd 必须分开，否则会互相稀释）。"""
    v = r.get("variant", "concat")
    wd = r.get("_weight_decay")
    if wd is None:
        return v
    return f"{v}{'_nwd' if wd == 0 else '_wd'}"


def summarize(rs: list[dict], key: str) -> dict:
    vals = [r[key] for r in rs if r.get(key) is not None]
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": round(st.mean(vals), 6),
        "sd": round(st.pstdev(vals), 6),
        "values": [round(v, 6) for v in vals],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline-runs", nargs="*", default=[])
    ap.add_argument("--env-id", default="BabyAI-GoToLocalS6N3-v0")
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--out", default="reports/b2_film_eval.json")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    results = [eval_one(r, args.env_id, seeds) for r in args.runs]
    base_results = [eval_one(r, args.env_id, seeds) for r in args.baseline_runs]

    # 按 variant(+wd 档位) 聚合
    groups: dict[str, list[dict]] = {}
    for r in base_results:
        groups.setdefault("bc30", []).append(r)
    for r in results:
        groups.setdefault(group_key(r), []).append(r)

    agg = {
        name: {
            "closed_loop_success": summarize(rs, "closed_loop_success"),
            "val_action_acc": summarize(rs, "val_action_acc"),
            "mean_return": summarize(rs, "mean_return"),
            "median_len_success": summarize(rs, "median_len_success"),
            "n_timeout": summarize(rs, "n_timeout"),
        }
        for name, rs in groups.items()
    }

    # 配对比较（同 train_seed）
    base_by_seed = {r["train_seed"]: r for r in base_results}
    paired = {}
    for r in results:
        b = base_by_seed.get(r["train_seed"])
        if not b:
            continue
        paired.setdefault(group_key(r), []).append(
            {
                "train_seed": r["train_seed"],
                "bc30_success": b["closed_loop_success"],
                "variant_success": r["closed_loop_success"],
                "delta": round(r["closed_loop_success"] - b["closed_loop_success"], 6),
                "bc30_val_acc": b["val_action_acc"],
                "variant_val_acc": r["val_action_acc"],
                "delta_val_acc": (
                    round(r["val_action_acc"] - b["val_action_acc"], 6)
                    if b["val_action_acc"] is not None and r["val_action_acc"] is not None
                    else None
                ),
            }
        )

    paired_summary = {}
    for v, rows in paired.items():
        d = [x["delta"] for x in rows]
        da = [x["delta_val_acc"] for x in rows if x["delta_val_acc"] is not None]
        paired_summary[v] = {
            "n_pairs": len(d),
            "delta_mean": round(st.mean(d), 6),
            "delta_sd": round(st.pstdev(d), 6) if len(d) > 1 else 0.0,
            "n_positive": sum(1 for x in d if x > 0),
            "delta_val_acc_mean": round(st.mean(da), 6) if da else None,
            "rows": rows,
        }

    payload = {
        "branch": "b2_film",
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "n_episodes_per_run": len(seeds),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": results,
        "baseline_runs": base_results,
        "aggregate": agg,
        "paired": paired_summary,
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"{'group':14s} {'n':>2s} {'val_acc':>15s} {'success':>15s} {'return':>8s} {'to':>6s}")
    for name, a in agg.items():
        s, acc = a["closed_loop_success"], a["val_action_acc"]
        print(
            f"{name:14s} {s['n']:2d} "
            f"{acc['mean']:.4f}±{acc['sd']:.4f} "
            f"{s['mean']:.4f}±{s['sd']:.4f} "
            f"{a['mean_return']['mean']:.3f} "
            f"{a['n_timeout']['mean']:.1f}",
            end="",
        )
        ps = paired_summary.get(name)
        if ps:
            print(f"   Δ={ps['delta_mean']:+.4f} ({ps['n_positive']}/{ps['n_pairs']} 正)")
        else:
            print()
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
