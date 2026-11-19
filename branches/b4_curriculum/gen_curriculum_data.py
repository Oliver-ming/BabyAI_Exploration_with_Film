"""支线 4 · 生成多难度辅助专家轨迹池（严格隔离 TEST_SEEDS，复用 BabyAIBot 口径）。

默认生成 4 份辅助环境数据（每份 2,250 条成功专家 episode，严格取自 TRAIN_SEEDS）：
  - data/b4_curriculum_objs6_2250.pkl  (BabyAI-GoToObjS6-v1: 6x6 无干扰物，先易)
  - data/b4_curriculum_s5n2_2250.pkl   (BabyAI-GoToLocalS5N2-v0: 5x5 2干扰物，官方候选)
  - data/b4_curriculum_s6n4_2250.pkl   (BabyAI-GoToLocalS6N4-v0: 6x6 4干扰物，高密度同房间)
  - data/b4_curriculum_s8n6_2250.pkl   (BabyAI-GoToLocalS8N6-v0: 8x8 6干扰物，更难跨尺寸)

注：训练池绝不接触 7x7 S7N4 与 TEST_SEEDS / TEST_SEEDS_S7，保持 S7N4 为纯未见迁移基准。

用法:
    OMP_NUM_THREADS=1 python branches/b4_curriculum/gen_curriculum_data.py --workers 4
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import pickle
import sys
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.utils.baby_ai_bot import BabyAIBot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    TRAIN_SEEDS,
    assert_no_leak,
    mission_to_ids,
    split_of,
)

DEFAULT_ENVS = [
    "objs6:BabyAI-GoToObjS6-v1",
    "s5n2:BabyAI-GoToLocalS5N2-v0",
    "s6n4:BabyAI-GoToLocalS6N4-v0",
    "s8n6:BabyAI-GoToLocalS8N6-v0",
]


def gen_one(env, env_id: str, seed: int) -> dict | None:
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    mission_to_ids(obs["mission"])  # 校验无 OOV token
    bot = BabyAIBot(u)

    images, dirs, actions, positions = [], [], [], []
    last_action = None
    terminated = truncated = False

    for _ in range(u.max_steps):
        action = bot.replan(last_action)
        if action is None:
            break
        images.append(obs["image"].copy())
        dirs.append(int(obs["direction"]))
        actions.append(int(action))
        positions.append(tuple(int(v) for v in u.agent_pos))

        obs, _r, terminated, truncated, _ = env.step(action)
        last_action = action
        if terminated or truncated:
            break

    if not terminated:
        return None

    return dict(
        seed=seed,
        split=split_of(seed),
        env_id=env_id,
        mission=obs["mission"],
        obj_poss_n=len(u.instrs.desc.obj_poss) if hasattr(u.instrs, "desc") else None,
        images=images,
        directions=dirs,
        actions=actions,
        agent_positions=positions,
        episode_len=len(actions),
    )


def gen_env_task(task: tuple[str, str, int, str]) -> dict:
    tag, env_id, n_episodes, out_dir = task
    out_path = Path(out_dir) / f"b4_curriculum_{tag}_{n_episodes}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    episodes, failed = [], []
    t0 = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TRAIN_SEEDS:
            if len(episodes) >= n_episodes:
                break
            ep = gen_one(env, env_id, seed)
            if ep is None:
                failed.append(seed)
            else:
                episodes.append(ep)
    env.close()

    assert len(episodes) == n_episodes, f"{tag} ({env_id}) 仅收集到 {len(episodes)}/{n_episodes}"
    assert_no_leak(e["seed"] for e in episodes)
    assert all(e["split"] == "train" for e in episodes), f"{tag} 含非 train split"

    with open(out_path, "wb") as f:
        pickle.dump(episodes, f)

    n_tr = sum(e["episode_len"] for e in episodes)
    return {
        "tag": tag,
        "env_id": env_id,
        "out": str(out_path),
        "n_episodes": len(episodes),
        "n_transitions": n_tr,
        "avg_len": round(n_tr / len(episodes), 2),
        "n_failed": len(failed),
        "seconds": round(time.time() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=DEFAULT_ENVS,
                    help="格式为 tag:EnvId 的列表")
    ap.add_argument("--n-episodes", type=int, default=2250)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tasks = []
    for spec in args.envs:
        tag, env_id = spec.split(":", 1)
        tasks.append((tag, env_id, args.n_episodes, args.out_dir))

    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            results = list(ex.map(gen_env_task, tasks))
    else:
        results = [gen_env_task(t) for t in tasks]

    for r in results:
        print(
            f"[gen {r['tag']:6s}] {r['env_id']:26s} eps={r['n_episodes']} "
            f"steps={r['n_transitions']} avg_len={r['avg_len']:.2f} "
            f"fail={r['n_failed']} ({r['seconds']}s) -> {r['out']}"
        )


if __name__ == "__main__":
    main()
