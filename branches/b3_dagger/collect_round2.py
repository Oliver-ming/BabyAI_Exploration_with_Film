"""支线 3 · 第二轮采集（DAgger 迭代）：用**已经训练好的 DAgger 策略**再采一轮。

动机（LOG.md §7 改进 B）：文献说 DAgger 多轮收益会饱和，但本环境的 S7N4 迁移仍明显偏低，
说明可能尚未饱和。本脚本产出第二轮的采集数据，用于检验：
**第二轮增益是否 < 第一轮的一半？**（若是，则判定饱和）

用法:
    python branches/b3_dagger/collect_round2.py --policy runs/b3_dagger_seed0 \
        --episodes 1500 --out data/b3_dagger_r2_pol0.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "branches" / "b3_dagger"))

from collect_dagger import collect_one, main as _unused  # noqa: F401,E402
import contextlib  # noqa: E402
import io  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from branch_utils import assert_no_test_seed, make_env  # noqa: E402
from common import ENV_ID, TRAIN_SEEDS, VAL_SEEDS, load_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="第一轮训练出的 DAgger 策略目录")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    assert_no_test_seed(TRAIN_SEEDS)
    out = Path(args.out or f"data/b3_dagger_r{args.round}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)

    model, cfg, _ = load_run(args.policy)
    from minigrid.utils.baby_ai_bot import BabyAIBot

    env = make_env(args.env_id)
    episodes, seeds = [], []
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TRAIN_SEEDS:
            if len(episodes) >= args.episodes:
                break
            ep = collect_one(env, model, BabyAIBot, seed, "dagger")
            if not ep["images"]:
                continue
            episodes.append(ep)
            seeds.append(seed)
    env.close()

    n_tr = sum(e["episode_len"] for e in episodes)
    meta = {
        "branch": "b3_dagger",
        "mode": "dagger",
        "round": args.round,
        "policy_run": args.policy,
        "policy_val_acc": cfg["result"]["best_val_acc"],
        "env_id": args.env_id,
        "seed_pool": "train",
        "n_episodes": len(episodes),
        "n_transitions": n_tr,
        "seeds": seeds,
        "policy_success_rate": float(np.mean([e["policy_success"] for e in episodes])),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out, "wb") as f:
        pickle.dump({"meta": meta, "episodes": episodes}, f)

    print(
        f"round {args.round}: {len(episodes)} episodes, {n_tr} transitions "
        f"-> {out}\n  策略成功率 {meta['policy_success_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
