"""支线 3 · DAgger 数据采集（状态由策略访问，标签由 BabyAIBot 提供）。

用法:
    # 标准 DAgger：策略访问状态，专家打标签
    python branches/b3_dagger/collect_dagger.py --episodes 1500 --out data/b3_dagger_1500.pkl

    # 等量对照：同样条数的**纯专家**轨迹（环境按专家动作推进）
    python branches/b3_dagger/collect_dagger.py --episodes 1500 --mode expert \
        --out data/b3_expertextra_1500.pkl

    # 只采策略失败的轨迹（failure-focused 变体）
    python branches/b3_dagger/collect_dagger.py --episodes 1500 --failure-only \
        --out data/b3_dagger_fail.pkl

三条不动的规则（见 branches/b3_dagger/LOG.md §2.2）：
  1. 环境永远执行**策略**动作；只有标签来自专家 —— 否则退化成 BC
  2. `bot.replan(last_expert_a)` 传专家自己的动作链，不传策略动作
     （实测两种做法标签分歧率 0.0%，见 reports/b3_dagger_label_diag.json，
      因为 BabyAIBot 每步都 `_process_obs()` 重置规划栈）
  3. 采集池只用 TRAIN_SEEDS：VAL_SEEDS 是选 checkpoint 的唯一依据，不可污染
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import assert_no_test_seed, load_expert, make_env  # noqa: E402
from common import ENV_ID, TRAIN_SEEDS, VAL_SEEDS, load_run, predict_actions  # noqa: E402

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def collect_one(env, model, bot_cls, seed: int, mode: str) -> dict:
    """跑一条 episode，返回带专家标签的状态序列。

    mode="dagger": 环境按策略动作推进（状态来自 d_π），标签来自专家
    mode="expert": 环境按专家动作推进（状态来自 d_π*），标签也是专家
                   —— 这是等量对照用的"纯专家"采集
    """
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    bot = bot_cls(u) if bot_cls is not None else None

    images, dirs, missions, expert_actions, policy_actions = [], [], [], [], []
    terminated = truncated = False
    last_expert = None

    for _t in range(u.max_steps):
        if mode == "expert":
            expert_a = bot.replan(last_expert)
            if expert_a is None:
                break
            taken_a = int(expert_a)
            last_expert = expert_a
        else:  # dagger
            policy_a = int(predict_actions(model, [obs])[0])
            expert_a = bot.replan(last_expert)
            last_expert = expert_a
            taken_a = policy_a
            policy_actions.append(policy_a)

        if expert_a is None:  # bot 认为自己已完成；这一状态没有可用标签
            break

        images.append(obs["image"].copy())
        dirs.append(int(obs["direction"]))
        missions.append(obs["mission"])
        expert_actions.append(int(expert_a))

        obs, _r, terminated, truncated, _ = env.step(taken_a)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "mode": mode,
        "mission": missions[0] if missions else None,
        "images": images,
        "directions": dirs,
        "actions": expert_actions,  # 标签永远是专家动作
        "policy_actions": policy_actions,  # 仅供参考：策略实际做了什么
        "expert_success": bool(terminated) if mode == "expert" else None,
        "policy_success": bool(terminated) if mode == "dagger" else None,
        "episode_len": len(images),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/baseline_seed0", help="用哪个策略去访问状态")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--seeds", default="train", choices=["train", "val"],
                    help="train=只在 TRAIN_SEEDS 上采集（默认）。"
                         "**不用 val**：VAL_SEEDS 是选 checkpoint 的唯一依据，"
                         "把它的状态拿去训练会让验证准确率失去意义")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--mode", default="dagger", choices=["dagger", "expert"])
    ap.add_argument("--failure-only", action="store_true",
                    help="只保留策略失败的轨迹（DAgger 的 failure-focused 变体）")
    ap.add_argument("--out", default="")
    ap.add_argument("--policy-seed", type=int, default=0, help="策略的初始化 seed（仅写进 meta）")
    args = ap.parse_args()

    seed_pool = TRAIN_SEEDS if args.seeds == "train" else VAL_SEEDS
    assert_no_test_seed(seed_pool)  # 硬断言：绝不碰 TEST_SEEDS
    assert not (set(seed_pool) & set(VAL_SEEDS)), (
        "❌ 采集池里混进了 VAL_SEEDS：会污染验证集口径，"
        "导致「验证准确率选 checkpoint」这条硬规定失效"
    )

    out = Path(args.out or f"data/b3_{args.mode}_{args.episodes}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)

    model, cfg, _ = load_run(args.run)
    from minigrid.utils.baby_ai_bot import BabyAIBot

    env = make_env(args.env_id)
    episodes, used_seeds = [], []
    t0 = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        n_tried = 0
        for seed in seed_pool:
            if len(episodes) >= args.episodes:
                break
            if n_tried >= args.episodes * 12:  # 防止 failure-only 时无限扫描
                break
            n_tried += 1
            ep = collect_one(env, model, BabyAIBot, seed, args.mode)
            if not ep["images"]:
                continue
            if args.failure_only and ep.get("policy_success", True):
                continue
            episodes.append(ep)
            used_seeds.append(seed)

    env.close()

    n_tr = sum(e["episode_len"] for e in episodes)
    meta = {
        "branch": "b3_dagger",
        "mode": args.mode,
        "failure_only": args.failure_only,
        "policy_run": args.run,
        "policy_seed": args.policy_seed,
        "policy_val_acc": cfg["result"]["best_val_acc"],
        "env_id": args.env_id,
        "seed_pool": args.seeds,
        "n_episodes": len(episodes),
        "n_transitions": n_tr,
        "seeds": used_seeds,
        "dagger_policy_success_rate": (
            float(np.mean([e["policy_success"] for e in episodes])) if args.mode == "dagger" else None
        ),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    with open(out, "wb") as f:
        pickle.dump({"meta": meta, "episodes": episodes}, f)

    print(
        f"{args.mode}: {len(episodes)} episodes, {n_tr} transitions "
        f"(avg {n_tr / max(1, len(episodes)):.2f}) -> {out}"
    )
    if args.mode == "dagger":
        print(f"  被采集策略在 VAL 上的成功率：{meta['dagger_policy_success_rate']:.3f}")
    print(f"  elapsed {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
