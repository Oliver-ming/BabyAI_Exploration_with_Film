"""支线 3 · 失败构成对比：DAgger 到底修掉了哪一类失败？

动机：闭环 Success 只从 0.747 涨到 0.887（单 seed），这么小的变化无法说明
"DAgger 是否按它的机制在工作"。机制层面的检验是：
**它该修的（原地转圈/顶墙，即状态覆盖问题）是否真的少了，
它不该修的（认错物体属性，即表征问题）是否基本不动。**

用法:
    python branches/b3_dagger/compare_failures.py \
        --runs runs/baseline_seed0 runs/b3_dagger_seed0 --seeds test \
        --out reports/b3_dagger_failure_compare.json
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from common import load_run  # noqa: E402
from render_episodes import classify_failure, run_episode  # noqa: E402

# 按机制分组：DAgger 应该只影响第一组
MECHANISM = {
    "orientation_lock": "state_coverage",
    "turn_oscillation": "state_coverage",
    "approach_then_lost": "state_coverage",
    "forward_stall": "state_coverage",
    "wander_timeout": "state_coverage",
    "attribute_confusion": "representation",
    "distractor_priority": "representation",
}


def analyze(run: str, seeds, env_id: str) -> dict:
    model, cfg, _ = load_run(run)
    env = make_env(env_id)
    trajs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for s in seeds:
            trajs.append(run_episode(env, model, s, render=False))
    env.close()

    types = collections.Counter()
    n_succ = 0
    for t in trajs:
        if t["success"]:
            n_succ += 1
        else:
            ft, _ = classify_failure(t)
            types[ft] += 1

    groups = collections.Counter()
    for k, v in types.items():
        groups[MECHANISM.get(k, "other")] += v

    per_mission = collections.defaultdict(lambda: [0, 0])
    for t in trajs:
        m = per_mission[t["mission"]]
        m[0] += 1
        m[1] += int(t["success"])

    return {
        "run": run,
        "train_seed": cfg["seed"],
        "val_action_acc": cfg["result"]["best_val_acc"],
        "n_episodes": len(trajs),
        "n_success": n_succ,
        "closed_loop_success": round(n_succ / len(trajs), 6),
        "failure_types": dict(types),
        "failure_groups": dict(groups),
        "n_failure": len(trajs) - n_succ,
        "per_mission": {k: {"n": v[0], "success": round(v[1] / v[0], 4)} for k, v in sorted(per_mission.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--env-id", default="BabyAI-GoToLocalS6N3-v0")
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--out", default="reports/b3_dagger_failure_compare.json")
    args = ap.parse_args()

    from eval_closed_loop import parse_seeds

    seeds = parse_seeds(args.seeds)
    results = [analyze(r, seeds, args.env_id) for r in args.runs]

    # 汇总：按 run 名分组（baseline / dagger / 其它）
    def agg(prefix: str):
        rs = [r for r in results if Path(r["run"]).name.startswith(prefix)]
        if not rs:
            return None
        return {
            "n_runs": len(rs),
            "closed_loop_success_mean": round(sum(r["closed_loop_success"] for r in rs) / len(rs), 6),
            "failure_types_mean": {
                k: round(sum(r["failure_types"].get(k, 0) for r in rs) / len(rs), 2)
                for k in sorted({k for r in rs for k in r["failure_types"]})
            },
            "failure_groups_mean": {
                k: round(sum(r["failure_groups"].get(k, 0) for r in rs) / len(rs), 2)
                for k in sorted({k for r in rs for k in r["failure_groups"]})
            },
        }

    summary = {
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "runs": results,
        "aggregate": {
            "baseline": agg("baseline"),
            "dagger": agg("b3_dagger"),
            "bc_extra": agg("b3_bcextra"),
            "dagger_r2": agg("b3_r2"),
            "novel_only": agg("b3_novel"),
        },
        "mechanism_map": MECHANISM,
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"{'run':38s} {'succ':>6s} {'fail':>5s}  失败构成")
    for r in results:
        ft = " ".join(f"{k[:6]}={v}" for k, v in sorted(r["failure_types"].items(), key=lambda kv: -kv[1]))
        print(f"{Path(r['run']).name:38s} {r['closed_loop_success']:6.3f} {r['n_failure']:5d}  {ft}")
    for name in ("baseline", "dagger", "dagger_r2", "bc_extra", "novel_only"):
        a = summary["aggregate"][name]
        if a:
            print(f"[{name}] mean success={a['closed_loop_success_mean']:.3f}  groups={a['failure_groups_mean']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
