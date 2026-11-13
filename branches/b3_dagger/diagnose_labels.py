"""支线 3 · 诊断：DAgger 标签质量（不训练，只测标签）。

动机：第一轮用"保守做法"（`bot.replan(expert_a)`）采集，结果 val_acc 从 0.890 掉到 0.813。
这不是噪声量级，怀疑是**标签与世界状态脱同步**。本脚本量化三件事：

1. **保守 vs 激进**：`replan(expert_a)` 与 `replan(policy_a)` 的标签分歧率
   （BabyAIBot 的 docstring 明说：参数是"agent 实际执行的动作"，用于 advising mode。
    指导文档推荐的"保守做法"与 bot 自身设计存在矛盾，值得实测）
2. **标签自洽性**：把一个"全新 bot"放到同一个世界状态上、并复制累积的 vis_mask，
   看它给出的动作与记录的标签是否一致。不一致 = 规划栈已脱同步
3. **标签振荡率**：连续两步标签是否 left/right 互翻（DAgger 应该修掉的东西，
   如果标签本身在翻，说明操作符有问题）

用法:
    python branches/b3_dagger/diagnose_labels.py --episodes 100
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import assert_no_test_seed, make_env  # noqa: E402
from common import TRAIN_SEEDS, load_run, predict_actions  # noqa: E402

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def run_seed(env, model, bot_cls, seed: int) -> dict:
    """在同一个 episode 上同时算两种标注方式，逐步记录。"""
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped

    bot_cons = bot_cls(u)  # 保守：喂专家自己的上一动作
    bot_aggr = bot_cls(u)  # 激进：喂实际执行的动作（bot docstring 的做法）

    rows = []
    last_cons = None
    terminated = truncated = False

    for _t in range(u.max_steps):
        policy_a = int(predict_actions(model, [obs])[0])

        a_cons = bot_cons.replan(last_cons)
        a_aggr = bot_aggr.replan(policy_a)  # 实际执行的就是 policy_a

        # 参考标签：全新 bot + 复制累积的可见掩码（近似"这个状态下专家会怎么做"）
        ref_bot = bot_cls(u)
        ref_bot.vis_mask = bot_aggr.vis_mask.copy()
        a_ref = ref_bot.replan(None)

        rows.append(
            {
                "t": _t,
                "policy": policy_a,
                "label_cons": int(a_cons) if a_cons is not None else None,
                "label_aggr": int(a_aggr) if a_aggr is not None else None,
                "label_ref": int(a_ref) if a_ref is not None else None,
                "agent_pos": list(map(int, u.agent_pos)),
            }
        )
        last_cons = a_cons
        obs, _r, terminated, truncated, _ = env.step(policy_a)
        if terminated or truncated:
            break

    return {"seed": seed, "success": bool(terminated), "steps": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/baseline_seed0")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="reports/b3_dagger_label_diag.json")
    args = ap.parse_args()

    assert_no_test_seed(TRAIN_SEEDS[:1])  # 只在 TRAIN_SEEDS 上诊断
    model, _cfg, _ = load_run(args.run)
    from minigrid.utils.baby_ai_bot import BabyAIBot

    env = make_env()
    results = []
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TRAIN_SEEDS[: args.episodes]:
            results.append(run_seed(env, model, BabyAIBot, seed))
    env.close()

    all_rows = [r for ep in results for r in ep["steps"]]
    have = [r for r in all_rows if r["label_cons"] is not None and r["label_aggr"] is not None]

    def rate(fn):
        return round(float(np.mean([fn(r) for r in have])), 4) if have else None

    cons_vs_aggr = rate(lambda r: r["label_cons"] != r["label_aggr"])
    cons_vs_ref = rate(
        lambda r: r["label_ref"] is not None and r["label_cons"] != r["label_ref"]
    )
    aggr_vs_ref = rate(
        lambda r: r["label_ref"] is not None and r["label_aggr"] != r["label_ref"]
    )

    # 标签振荡：连续两步都是转向且方向相反
    flips = {"cons": 0, "aggr": 0, "total": 0}
    for ep in results:
        s = ep["steps"]
        for a, b in zip(s, s[1:]):
            if a["label_cons"] is None or b["label_cons"] is None:
                continue
            flips["total"] += 1
            if a["label_cons"] in (0, 1) and b["label_cons"] == 1 - a["label_cons"]:
                flips["cons"] += 1
            if a["label_aggr"] in (0, 1) and b["label_aggr"] == 1 - a["label_aggr"]:
                flips["aggr"] += 1

    dist = {
        name: dict(
            collections.Counter(
                ACTION_NAMES[r[key]] for r in have if r[key] is not None
            )
        )
        for name, key in (("cons", "label_cons"), ("aggr", "label_aggr"), ("ref", "label_ref"))
    }

    out = {
        "run": args.run,
        "n_episodes": len(results),
        "n_steps": len(have),
        "policy_success_rate": round(float(np.mean([e["success"] for e in results])), 4),
        "disagreement": {
            "cons_vs_aggr": cons_vs_aggr,
            "cons_vs_reference": cons_vs_ref,
            "aggr_vs_reference": aggr_vs_ref,
        },
        "label_flip_rate": {
            "cons": round(flips["cons"] / max(1, flips["total"]), 4),
            "aggr": round(flips["aggr"] / max(1, flips["total"]), 4),
        },
        "label_distribution": dist,
        "samples": results[0]["steps"][:12],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"episodes={len(results)} steps={len(have)} policy_success={out['policy_success_rate']:.3f}")
    print(f"标签分歧  cons vs aggr = {cons_vs_aggr:.1%}")
    print(f"           cons vs 参考 = {cons_vs_ref:.1%}")
    print(f"           aggr vs 参考 = {aggr_vs_ref:.1%}")
    print(f"标签振荡率 cons = {out['label_flip_rate']['cons']:.1%}  aggr = {out['label_flip_rate']['aggr']:.1%}")
    print(f"标签分布 cons={dist['cons']}")
    print(f"        aggr={dist['aggr']}")
    print(f"        ref ={dist['ref']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
