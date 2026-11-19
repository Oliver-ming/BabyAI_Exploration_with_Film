"""支线 4 · 失败构成对比（严格复用冻结的 `render_episodes.classify_failure` 口径）。

支持 S6N3 (test) 与 S7N4 (test_s7)，拆解「表征混淆类」与「状态覆盖类」失败下降来源。

用法:
    OMP_NUM_THREADS=1 python branches/b4_curriculum/compare_failures.py \
        --runs runs/b3_bc30_seed* runs/b2_film_nwd_seed* runs/b4_*_seed* \
        --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers 15 \
        --out reports/b4_curriculum_failures.json
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
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b4_curriculum"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from curriculum_policy import load_any_run, make_policy_fn  # noqa: E402
from eval_closed_loop import parse_seeds  # noqa: E402
from eval_curriculum import group_name  # noqa: E402
from render_episodes import classify_failure  # noqa: E402

MECHANISM = {
    "orientation_lock": "state_coverage",
    "turn_oscillation": "state_coverage",
    "approach_then_lost": "state_coverage",
    "forward_stall": "state_coverage",
    "wander_timeout": "state_coverage",
    "attribute_confusion": "representation",
    "distractor_priority": "representation",
}

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def run_episode_with_policy(env, policy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    mission = obs["mission"]

    desc = getattr(u.instrs, "desc", None)
    target_poss = [tuple(int(v) for v in p) for p in (desc.obj_poss if desc else [])]
    target_color = desc.color if desc is not None else None
    target_type = desc.type if desc is not None else None

    objects = []
    for i in range(u.grid.width):
        for j in range(u.grid.height):
            c = u.grid.get(i, j)
            if c is not None and c.type not in ("wall", "door"):
                objects.append({"pos": [i, j], "type": c.type, "color": getattr(c, "color", None)})

    def manhattan(a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    steps = []
    terminated = truncated = False
    for t in range(u.max_steps):
        a = policy(obs)
        agent_pos = tuple(int(v) for v in u.agent_pos)
        front = tuple(int(v) for v in u.front_pos)
        cell = u.grid.get(*front)
        steps.append(
            {
                "t": t,
                "action": ACTION_NAMES[a],
                "agent_pos": list(agent_pos),
                "agent_dir": int(u.agent_dir),
                "front_cell": None if cell is None else {"type": cell.type, "color": getattr(cell, "color", None)},
                "facing_target": front in target_poss,
                "manhattan_to_target": min(manhattan(agent_pos, p) for p in target_poss) if target_poss else None,
            }
        )
        obs, _r, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            break

    final_pos = tuple(int(v) for v in u.agent_pos)
    final_front = tuple(int(v) for v in u.front_pos)
    fc = u.grid.get(*final_front)
    distractors = [
        o for o in objects
        if tuple(o["pos"]) not in target_poss
        and (o["type"] == target_type or o["color"] == target_color)
    ]

    return {
        "seed": seed,
        "mission": mission,
        "success": bool(terminated),
        "truncated": bool(truncated),
        "episode_len": len(steps),
        "steps": steps,
        "target_poss": [list(p) for p in target_poss],
        "target": {"color": target_color, "type": target_type},
        "distractors": distractors,
        "distractor_dists": [manhattan(final_pos, d["pos"]) for d in distractors],
        "final_agent_pos": list(final_pos),
        "final_agent_dir": int(u.agent_dir),
        "final_front_cell": None if fc is None else {"type": fc.type, "color": getattr(fc, "color", None)},
        "final_facing_target": bool(final_front in target_poss),
        "final_distance_to_target": min(manhattan(final_pos, p) for p in target_poss) if target_poss else None,
    }


def analyze_task(task: tuple[str, list[int], str]) -> dict:
    run, seeds, env_id = task
    torch.set_num_threads(1)
    model, cfg, _ = load_any_run(run)
    policy = make_policy_fn(model)
    grp = group_name(run, cfg)

    env = make_env(env_id)
    trajs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for s in seeds:
            trajs.append(run_episode_with_policy(env, policy, s))
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

    return {
        "run": run,
        "group": grp,
        "variant": grp,
        "train_seed": cfg.get("seed"),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "n_episodes": len(trajs),
        "n_success": n_succ,
        "closed_loop_success": round(n_succ / len(trajs), 6),
        "failure_types": dict(types),
        "failure_groups": dict(groups),
        "n_failure": len(trajs) - n_succ,
    }


def agg(rs: list[dict]) -> dict:
    keys = sorted({k for r in rs for k in r["failure_types"]})
    gkeys = sorted({k for r in rs for k in r["failure_groups"]})
    return {
        "n_runs": len(rs),
        "closed_loop_success_mean": round(float(np.mean([r["closed_loop_success"] for r in rs])), 4),
        "n_failure_mean": round(float(np.mean([r["n_failure"] for r in rs])), 2),
        "failure_types_mean": {
            k: round(float(np.mean([r["failure_types"].get(k, 0) for r in rs])), 2) for k in keys
        },
        "failure_groups_mean": {
            k: round(float(np.mean([r["failure_groups"].get(k, 0) for r in rs])), 2) for k in gkeys
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--env-id", default="BabyAI-GoToLocalS6N3-v0")
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default="reports/b4_curriculum_failures.json")
    args = ap.parse_args()

    runs = list(dict.fromkeys(args.runs))
    seeds = parse_seeds(args.seeds)
    tasks = [(r, seeds, args.env_id) for r in runs]
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            results = list(ex.map(analyze_task, tasks))
    else:
        results = [analyze_task(t) for t in tasks]

    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)

    summary = {
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "mechanism_map": MECHANISM,
        "runs": results,
        "aggregate": {g: agg(rs) for g, rs in by_group.items()},
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[{args.env_id}] {'group':22s} {'succ':>6s} {'fail':>5s} {'repr':>6s} {'state':>6s} {'attr_conf':>10s} {'approach':>9s} {'wander':>7s}")
    for name, a in sorted(summary["aggregate"].items()):
        fg = a["failure_groups_mean"]
        ft = a["failure_types_mean"]
        print(
            f"{name:22s} {a['closed_loop_success_mean']:6.3f} {a['n_failure_mean']:5.1f} "
            f"{fg.get('representation', 0.0):6.1f} {fg.get('state_coverage', 0.0):6.1f} "
            f"{ft.get('attribute_confusion', 0.0):10.1f} {ft.get('approach_then_lost', 0.0):9.1f} {ft.get('wander_timeout', 0.0):7.1f}"
        )
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
