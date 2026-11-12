"""P1-2 · 闭环评估（§7）——本赛道最重要的脚本。

用法:
    python src/eval_closed_loop.py \
        --runs runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
        --env-id BabyAI-GoToLocalS6N3-v0 --seeds 4500:4800 \
        --out reports/baseline_eval.json

协议要点：
  - 闭环：每步把模型预测喂回环境，**不做 teacher forcing**
  - `torch.no_grad()` + `model.eval()`
  - 成功判定用 `terminated`（BabyAI GoToInstr 只有 success 才会 terminated），`truncated` 是超时
  - 记录 per-mission 成功率、动作直方图、成功 episode 长度中位数
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ENV_ID,
    ENV_ID_S7,
    MAX_STEPS_S6N3,
    MAX_STEPS_S7N4,
    TEST_SEEDS,
    TEST_SEEDS_S7,
    load_run,
    predict_actions,
)

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
SEED_PRESETS = {
    "test": TEST_SEEDS,
    "test_s7": TEST_SEEDS_S7,
    "val": list(range(4000, 4500)),
    "train": list(range(0, 200)),  # 仅用于 DAgger 采集/自检，不做评估
}


def parse_seeds(spec: str) -> list[int]:
    """支持 `4500:4800`、`10000:10300`、或预设名。"""
    if spec in SEED_PRESETS:
        return list(SEED_PRESETS[spec])
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    return [int(x) for x in spec.split(",")]


def evaluate(policy, env_id: str, seeds: list[int], max_episodes: int | None = None):
    """闭环评估。policy: obs -> action（可以是模型包装，也可以是平凡策略）。"""
    seeds = seeds[:max_episodes] if max_episodes else seeds
    env = gym.make(env_id)

    successes, lengths, returns, trajectories = [], [], [], []
    action_ctr = collections.Counter()
    per_mission = collections.defaultdict(lambda: {"n": 0, "success": 0})

    with contextlib.redirect_stdout(io.StringIO()):  # 屏蔽 Sampling rejected 噪声
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            mission = obs["mission"]
            ep_actions, ep_rewards = [], []
            terminated = truncated = False
            for _t in range(env.unwrapped.max_steps):
                a = int(policy(obs))  # 闭环：策略自己的输出喂回环境
                ep_actions.append(a)
                obs, r, terminated, truncated, _ = env.step(a)
                ep_rewards.append(float(r))
                if terminated or truncated:
                    break

            action_ctr.update(ep_actions)
            m = per_mission[mission]
            m["n"] += 1
            m["success"] += int(terminated)
            successes.append(bool(terminated))
            returns.append(sum(ep_rewards))
            if terminated:
                lengths.append(len(ep_actions))
            trajectories.append(
                {
                    "seed": seed,
                    "mission": mission,
                    "success": bool(terminated),
                    "truncated": bool(truncated),
                    "episode_len": len(ep_actions),
                    "actions": ep_actions,
                }
            )

    env.close()

    n = len(seeds)
    total_actions = sum(action_ctr.values())
    return {
        "n_episodes": n,
        "closed_loop_success": round(float(np.mean(successes)), 6),
        "mean_return": round(float(np.mean(returns)), 6),
        "median_len_success": float(np.median(lengths)) if lengths else None,
        "mean_len_success": round(float(np.mean(lengths)), 3) if lengths else None,
        "mean_len_all": round(float(np.mean([t["episode_len"] for t in trajectories])), 3),
        "n_timeout": int(sum(t["truncated"] for t in trajectories)),
        "per_mission": {
            k: {"n": v["n"], "success": round(v["success"] / v["n"], 6)}
            for k, v in sorted(per_mission.items())
        },
        "action_histogram": {
            ACTION_NAMES[k]: round(v / total_actions, 6)
            for k, v in sorted(action_ctr.items())
        },
        "invalid_action_rate": 0.0,  # argmax over 7 个合法动作，结构上不可能非法
        "_trajectories": trajectories,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="run 目录列表（--policies 里可用 model）")
    ap.add_argument("--policies", nargs="*", default=None,
                    help="额外评估的平凡策略：always_forward / always_left / always_right / random")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--seeds", default="test", help="`4500:4800` / `test` / `test_s7` / 逗号列表")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--out", default="reports/baseline_eval.json")
    ap.add_argument("--save-trajectories", action="store_true",
                    help="把逐 episode 轨迹写进 json（体积较大，失败案例采集时需要）")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    results = []
    t0 = time.time()

    # 平凡策略：用于验证"动作准确率 57% 但 Success 0"这一论点
    def trivial(name, seed=0):
        rng = np.random.default_rng(seed)
        table = {
            "always_forward": lambda obs: 2,
            "always_left": lambda obs: 0,
            "always_right": lambda obs: 1,
            "random": lambda obs: int(rng.integers(0, 3)),
        }
        return table[name]

    for run_dir in args.runs:
        model, cfg, _vocab = load_run(run_dir)
        t1 = time.time()
        res = evaluate(lambda obs: predict_actions(model, [obs])[0], args.env_id, seeds,
                       args.max_episodes or None)
        traj = res.pop("_trajectories")
        if args.save_trajectories:
            res["trajectories"] = traj
        res.update(
            {
                "run": run_dir,
                "kind": "bc",
                "env_id": args.env_id,
                "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
                "train_seed": cfg["seed"],
                "val_action_acc": cfg["result"]["best_val_acc"],
                "best_epoch": cfg["result"]["best_epoch"],
                "eval_seconds": round(time.time() - t1, 2),
            }
        )
        results.append(res)
        print(
            f"{run_dir}: success={res['closed_loop_success']:.3f} "
            f"val_acc={res['val_action_acc']:.3f} "
            f"return={res['mean_return']:.3f} "
            f"median_len={res['median_len_success']} "
            f"timeouts={res['n_timeout']}/{res['n_episodes']} "
            f"({res['eval_seconds']}s)"
        )
        print(f"   动作分布: {res['action_histogram']}")

    for name in args.policies or []:
        t1 = time.time()
        res = evaluate(trivial(name), args.env_id, seeds, args.max_episodes or None)
        res.pop("_trajectories")
        res.update(
            {
                "run": f"trivial:{name}",
                "kind": "trivial",
                "env_id": args.env_id,
                "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
                "train_seed": None,
                "val_action_acc": None,
                "best_epoch": None,
                "eval_seconds": round(time.time() - t1, 2),
            }
        )
        results.append(res)
        print(
            f"trivial:{name}: success={res['closed_loop_success']:.3f} "
            f"return={res['mean_return']:.3f} timeouts={res['n_timeout']}/{res['n_episodes']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "n_episodes_per_run": len(seeds) if not args.max_episodes else args.max_episodes,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_seconds": round(time.time() - t0, 2),
        "runs": results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
