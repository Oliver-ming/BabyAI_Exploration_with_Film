"""支线 5 · 失败构成随数据规模演化拆解（复用冻结 `render_episodes.classify_failure` 口径）。

检验数据规模扩大与 D4 群对称增广究竟如何分别压低「表征混淆类」与「状态覆盖类」失败。

用法:
    OMP_NUM_THREADS=1 python branches/b5_data_eff/compare_failures.py \
        --workers 16 --out reports/b5_data_eff_failures.json
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
sys.path.insert(0, str(ROOT / "branches" / "b5_data_eff"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from data_eff_policy import load_any_run, make_policy_fn  # noqa: E402
from data_eff_utils import frac_tag  # noqa: E402
from eval_closed_loop import parse_seeds  # noqa: E402
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


def build_tasks() -> list[dict]:
    tasks = []
    for s in (0, 1, 2):
        tasks.append({"run": f"runs/b3_bc30_seed{s}", "method": "bc_step", "frac": 1.0, "ftag": "f100", "d4_tta": False})
        tasks.append({"run": f"runs/b2_film_nwd_seed{s}", "method": "film_step", "frac": 1.0, "ftag": "f100", "d4_tta": False})
    for m in ("bc_naive", "bc_step", "bc_d4", "film_step", "film_d4"):
        for f in (0.05, 0.10, 0.20, 0.50, 1.00):
            if m in ("bc_naive", "bc_step", "film_step") and f == 1.00:
                continue
            ft = frac_tag(f)
            for s in (0, 1, 2):
                p = f"runs/b5_{m}_{ft}_seed{s}"
                if Path(p).exists():
                    tasks.append({"run": p, "method": m, "frac": f, "ftag": ft, "d4_tta": False})
                    if m in ("bc_d4", "film_d4") and f == 0.20:
                        tasks.append({"run": p, "method": f"{m}_tta", "frac": f, "ftag": ft, "d4_tta": True})
    for m in ("bc_reg", "bc_z2", "bc_c4", "bc_wd", "bc_bal", "gru_small_step"):
        for s in (0, 1, 2):
            p = f"runs/b5_{m}_f020_seed{s}"
            if Path(p).exists():
                tasks.append({"run": p, "method": m, "frac": 0.20, "ftag": "f020", "d4_tta": False})
    return tasks


def analyze_task(args: tuple[dict, list[int], str]) -> dict:
    spec, seeds, env_id = args
    torch.set_num_threads(1)
    model, cfg, _ = load_any_run(spec["run"])
    policy = make_policy_fn(model, d4_tta=spec["d4_tta"])

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
        "run": spec["run"],
        "method": spec["method"],
        "frac": spec["frac"],
        "frac_tag": spec["ftag"],
        "d4_tta": spec["d4_tta"],
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
        "method": rs[0]["method"],
        "frac": rs[0]["frac"],
        "frac_tag": rs[0]["frac_tag"],
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
    ap.add_argument("--env-id", default="BabyAI-GoToLocalS6N3-v0")
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="reports/b5_data_eff_failures.json")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    specs = build_tasks()
    tasks = [(sp, seeds, args.env_id) for sp in specs]
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            results = list(ex.map(analyze_task, tasks))
    else:
        results = [analyze_task(t) for t in tasks]

    buckets: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in results:
        buckets[(r["method"], r["frac_tag"])].append(r)

    aggregate = {f"{m}_{ft}": agg(rs) for (m, ft), rs in sorted(buckets.items())}
    summary = {
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "mechanism_map": MECHANISM,
        "runs": results,
        "aggregate": aggregate,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[{args.env_id}] {'group':22s} {'succ':>6s} {'fail':>5s} {'repr':>6s} {'state':>6s} {'attr_conf':>10s} {'ori_lock':>9s}")
    for name, a in aggregate.items():
        fg, ft = a["failure_groups_mean"], a["failure_types_mean"]
        print(
            f"  {name:20s} {a['closed_loop_success_mean']:6.3f} {a['n_failure_mean']:5.1f} "
            f"{fg.get('representation', 0.0):6.1f} {fg.get('state_coverage', 0.0):6.1f} "
            f"{ft.get('attribute_confusion', 0.0):10.1f} {ft.get('orientation_lock', 0.0):9.1f}"
        )
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
