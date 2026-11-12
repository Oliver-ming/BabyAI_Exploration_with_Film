"""P0-2 · 生成成功专家 episode（BabyAI-GoToLocalS6N3）。

用法:
    python src/gen_expert.py                     # 生成 5000 条到 data/expert_5000.pkl
    python src/gen_expert.py --episodes 200 --out data/expert_200.pkl

规则（§4.2）：
  - 只用 TRAIN_SEEDS 里的 seed（见 common.py 的划分说明）
  - 只保留 terminated=True（即真正成功）的 episode
  - 每步记录 (image, direction, mission) 与专家动作
  - reset 的 "Sampling rejected" 噪声用 redirect_stdout 屏蔽
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pickle
import sys
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  (注册 BabyAI-* 环境)
from minigrid.utils.baby_ai_bot import BabyAIBot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ENV_ID,
    EXPERT_ALL_SEEDS,
    N_EXPERT_TOTAL,
    N_EXPERT_TRAIN,
    N_EXPERT_VAL,
    TRAIN_SEEDS,
    VAL_SEEDS,
    split_of,
)

PRESETS = {
    "train": (TRAIN_SEEDS, N_EXPERT_TRAIN),
    "val": (VAL_SEEDS, N_EXPERT_VAL),
    "all": (EXPERT_ALL_SEEDS, N_EXPERT_TOTAL),  # 训练+验证 = 5000，对应 §4.3 的验收
}


def gen_one(env, seed: int):
    """跑一条专家 episode；成功返回 dict，失败返回 None。"""
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    bot = BabyAIBot(u)

    images, dirs, actions, positions = [], [], [], []
    last_action = None
    terminated = truncated = False

    for _ in range(u.max_steps):
        action = bot.replan(last_action)
        if action is None:  # pragma: no cover - bot 理论上不返回 None
            break
        images.append(obs["image"].copy())
        dirs.append(int(obs["direction"]))
        actions.append(int(action))
        positions.append(tuple(int(v) for v in u.agent_pos))

        obs, _r, terminated, truncated, _ = env.step(action)
        last_action = action
        if terminated or truncated:
            break

    if not terminated:  # 只保留成功
        return None

    return dict(
        seed=seed,
        split=split_of(seed),
        mission=obs["mission"],
        obj_poss_n=len(u.instrs.desc.obj_poss) if hasattr(u.instrs, "desc") else None,
        images=images,
        directions=dirs,
        actions=actions,
        agent_positions=positions,
        episode_len=len(actions),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0, help="0 = 该 seed 段的全部")
    ap.add_argument("--out", default="")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--seed-set", default="train", choices=sorted(PRESETS),
                    help="train / val / all(=train+val, 写成一个文件, 带 split 字段)")
    ap.add_argument("--seed-start", type=int, default=0, help="从该 seed 段第几个开始")
    ap.add_argument("--limit-seeds", type=int, default=0, help=">0 时只尝试前 N 个 seed（调试用）")
    args = ap.parse_args()

    seed_pool, default_n = PRESETS[args.seed_set]
    if args.episodes <= 0:
        args.episodes = default_n
    if not args.out:
        args.out = {
            "train": "data/expert_train.pkl",
            "val": "data/expert_val.pkl",
            "all": "data/expert_5000.pkl",
        }[args.seed_set]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seeds = seed_pool[args.seed_start :]
    if args.limit_seeds > 0:
        seeds = seeds[: args.limit_seeds]

    env = gym.make(args.env_id)
    episodes, failed = [], []
    t0 = time.time()

    # 环境 reset 会向 stdout 打印 "Sampling rejected: ..."（§4.1 ⑧），整体屏蔽
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in seeds:
            if len(episodes) >= args.episodes:
                break
            ep = gen_one(env, seed)
            if ep is None:
                failed.append(seed)
            else:
                episodes.append(ep)

    env.close()

    with open(out, "wb") as f:
        pickle.dump(episodes, f)

    n_tr = sum(e["episode_len"] for e in episodes)
    avg = n_tr / max(1, len(episodes))
    print(
        f"{len(episodes)} episodes, {n_tr} transitions, avg {avg:.2f} steps "
        f"(expert failures: {len(failed)}) -> {out}"
    )
    print(f"elapsed {time.time() - t0:.1f}s")
    if len(episodes) < args.episodes:
        print(
            f"WARNING: 只拿到 {len(episodes)} 条（目标 {args.episodes}），"
            f"seed 池耗尽。请扩大 TRAIN_SEEDS。"
        )


if __name__ == "__main__":
    main()
