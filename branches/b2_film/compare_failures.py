"""支线 2 · 失败构成对比（复刻 b3 的口径，但支持加载 FiLM 变体）。

为什么要重写而不是直接用 `b3_dagger/compare_failures.py`：
b3 的版本硬编码了 `common.load_run`（基线 BCNet），FiLM 的 run 目录没有 `img_mode` 字段，
会直接 KeyError。这里只替换"加载模型"这一行，失败归因逻辑**原样复用**冻结的
`render_episodes.classify_failure`，保证与基线的分类口径完全一致。

用法:
    python branches/b2_film/compare_failures.py \
        --baseline-runs runs/b3_bc30_seed0 runs/b3_bc30_seed1 runs/b3_bc30_seed2 \
        --film-runs runs/b2_film_seed0 runs/b2_film_seed1 runs/b2_film_seed2 \
        --seeds test --out reports/b2_failure_compare.json
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
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from common import load_run  # noqa: E402
from film_policy import load_any_run, make_policy_fn  # noqa: E402
from render_episodes import classify_failure  # noqa: E402

# 与 b3 相同的机制分组
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
    """闭环跑一条，返回与 `render_episodes.run_episode` 兼容的轨迹 dict。"""
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
                "front_cell": None
                if cell is None
                else {"type": cell.type, "color": getattr(cell, "color", None)},
                "facing_target": front in target_poss,
                "manhattan_to_target": min(manhattan(agent_pos, p) for p in target_poss)
                if target_poss
                else None,
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
        "final_front_cell": None
        if fc is None
        else {"type": fc.type, "color": getattr(fc, "color", None)},
        "final_facing_target": bool(final_front in target_poss),
        "final_distance_to_target": min(manhattan(final_pos, p) for p in target_poss)
        if target_poss
        else None,
    }


def analyze(run: str, seeds, env_id: str) -> dict:
    try:
        model, cfg, _ = load_any_run(run)
        policy = make_policy_fn(model)
    except Exception:
        model, cfg, _ = load_run(run)
        from common import predict_actions

        policy = lambda obs: int(predict_actions(model, [obs])[0])  # noqa: E731

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
        "variant": cfg.get("arch", {}).get("variant", "concat"),
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
    if not rs:
        return {}
    keys = sorted({k for r in rs for k in r["failure_types"]})
    gkeys = sorted({k for r in rs for k in r["failure_groups"]})
    return {
        "n_runs": len(rs),
        "closed_loop_success_mean": round(sum(r["closed_loop_success"] for r in rs) / len(rs), 6),
        "n_failure_mean": round(sum(r["n_failure"] for r in rs) / len(rs), 2),
        "failure_types_mean": {
            k: round(sum(r["failure_types"].get(k, 0) for r in rs) / len(rs), 2) for k in keys
        },
        "failure_groups_mean": {
            k: round(sum(r["failure_groups"].get(k, 0) for r in rs) / len(rs), 2) for k in gkeys
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-runs", nargs="+", default=[])
    ap.add_argument("--film-runs", nargs="+", required=True)
    ap.add_argument("--env-id", default="BabyAI-GoToLocalS6N3-v0")
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--out", default="reports/b2_failure_compare.json")
    args = ap.parse_args()

    from eval_closed_loop import parse_seeds

    seeds = parse_seeds(args.seeds)
    base = [analyze(r, seeds, args.env_id) for r in args.baseline_runs]
    film = [analyze(r, seeds, args.env_id) for r in args.film_runs]

    by_variant: dict[str, list[dict]] = collections.defaultdict(list)
    for r in film:
        by_variant[r["variant"]].append(r)

    summary = {
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "mechanism_map": MECHANISM,
        "runs": base + film,
        "aggregate": {
            "bc30": agg(base),
            **{f"{v}": agg(rs) for v, rs in by_variant.items()},
        },
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    for name, a in summary["aggregate"].items():
        if not a:
            continue
        ft = " ".join(f"{k[:8]}={v:.1f}" for k, v in sorted(a["failure_types_mean"].items(), key=lambda kv: -kv[1]))
        print(f"[{name:12s}] succ={a['closed_loop_success_mean']:.3f} fail={a['n_failure_mean']:5.1f}  {ft}")
        print(f"               groups={a['failure_groups_mean']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
