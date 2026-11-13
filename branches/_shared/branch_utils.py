"""支线共用的只读工具（不修改任何冻结模块，只 import 它们）。

放在 `branches/_shared/` 而不是 `src/`，是为了让 `src/` 保持冻结。

用法:
    import sys; sys.path.insert(0, "branches/_shared")
    from branch_utils import (
        load_expert, run_model_episode, label_with_bot, compare_with_baseline, BRANCH_SD
    )
"""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# 冻结模块：只读复用
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    ENV_ID,
    N_ACTIONS,
    TEST_SEEDS,
    TEST_SEEDS_S7,
    VAL_SEEDS,
    load_run,
    predict_actions,
)

# 基线（来自 reports/baseline_eval.json，冻结）
BASELINE = {
    "val_action_acc": 0.843,
    "val_action_acc_sd": 0.033,
    "closed_loop_success": 0.747,
    "closed_loop_success_sd": 0.093,
    "mean_return": 0.656,
    "median_len_success": 4.0,
}

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def load_expert(path: str = "data/expert_5000.pkl", split: str = "train") -> list[dict]:
    """读专家 episode（按 split 过滤）。只读，不写回。"""
    eps = pickle.load(open(path, "rb"))
    if split == "all":
        return eps
    return [e for e in eps if e.get("split") == split]


def quiet(fn, *a, **kw):
    """屏蔽 BabyAI reset 的 `Sampling rejected` 噪声。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def make_env(env_id: str = ENV_ID, render_mode=None):
    import gymnasium as gym
    import minigrid  # noqa: F401

    return gym.make(env_id, render_mode=render_mode)


def run_model_episode(env, model, seed: int, bot_relabel: bool = False):
    """跑一条闭环 episode。

    bot_relabel=True 时，每个**访问到的状态**都额外让 BabyAIBot 标注专家动作
    （这是 DAgger 的核心：环境按策略动作推进，标签来自专家）。

    返回 dict: success / steps / observations / policy_actions / expert_actions
    """
    from minigrid.utils.baby_ai_bot import BabyAIBot

    obs, _ = quiet(env.reset, seed=seed)
    u = env.unwrapped
    bot = BabyAIBot(u) if bot_relabel else None

    observations, policy_actions, expert_actions = [], [], []
    terminated = truncated = False
    last_expert = None

    for _t in range(u.max_steps):
        observations.append(
            {
                "image": obs["image"].copy(),
                "direction": int(obs["direction"]),
                "mission": obs["mission"],
            }
        )
        policy_a = int(predict_actions(model, [obs])[0])

        if bot is not None:
            # 保守做法（指导文档 §10.1 ②）：传给 bot 的是专家自己的动作链，
            # 让 bot 的内部规划状态与它看到的网格一致
            expert_a = bot.replan(last_expert)
            expert_actions.append(int(expert_a) if expert_a is not None else None)
            last_expert = expert_a

        policy_actions.append(policy_a)
        obs, _r, terminated, truncated, _ = env.step(policy_a)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "mission": observations[0]["mission"] if observations else None,
        "success": bool(terminated),
        "truncated": bool(truncated),
        "episode_len": len(policy_actions),
        "observations": observations,
        "policy_actions": policy_actions,
        "expert_actions": expert_actions,
    }


def evaluate_policy_fn(policy_fn, env_id: str = ENV_ID, seeds=None, max_steps: int | None = None):
    """用冻结的评估口径跑闭环，返回 success rate 等。"""
    from eval_closed_loop import evaluate

    seeds = list(TEST_SEEDS if seeds is None else seeds)
    res = evaluate(policy_fn, env_id, seeds)
    res.pop("_trajectories", None)
    return res


def compare_with_baseline(res: dict, baseline: dict | None = None) -> dict:
    """与基线比较，并按 seed 间 SD 判定显著性。"""
    b = baseline or BASELINE
    d_acc = res["val_action_acc"] - b["val_action_acc"] if res.get("val_action_acc") is not None else None
    d_succ = res["closed_loop_success"] - b["closed_loop_success"]
    out = {
        "delta_val_acc": d_acc,
        "delta_success": round(d_succ, 6),
        "significant_success": abs(d_succ) > b["closed_loop_success_sd"],
        "noise_floor_success_sd": b["closed_loop_success_sd"],
    }
    if d_acc is not None:
        out["significant_val_acc"] = abs(d_acc) > b["val_action_acc_sd"]
        out["noise_floor_val_acc_sd"] = b["val_action_acc_sd"]
    return out


def save_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"wrote {p}")


def assert_no_test_seed(seeds) -> None:
    """硬断言：一批 seed 里不得含任何测试 seed（支线采集数据时必调）。"""
    from common import TEST_SEEDS, TEST_SEEDS_S7

    bad = set(seeds) & (set(TEST_SEEDS) | set(TEST_SEEDS_S7))
    assert not bad, f"❌ 采集用到了测试 seed：{sorted(bad)[:10]}"


def load_dagger(path: str | Path) -> tuple[dict, list[dict]]:
    """读 DAgger 采集结果，返回 (meta, episodes)。"""
    obj = pickle.load(open(path, "rb"))
    return obj["meta"], obj["episodes"]


def frozen_diff_guard() -> str:
    """检查冻结区是否被改动（支线提交前跑一次）。"""
    import subprocess

    frozen = [
        "src/common.py", "src/train_bc.py", "src/eval_closed_loop.py",
        "src/gen_expert.py", "src/audit_data.py", "src/mission_breakdown.py",
        "src/analyze_failures.py", "src/render_episodes.py", "src/make_figures.py",
        "src/make_report.py", "src/check_env.py",
    ]
    r = subprocess.run(
        ["git", "diff", "--stat", "--"] + frozen,
        cwd=ROOT, capture_output=True, text=True,
    )
    diff = r.stdout.strip()
    if diff:
        raise SystemExit(
            "❌ 冻结区被改动，这会让其它四条支线的结果不可比：\n" + diff
        )
    return "✅ 冻结区干净"
