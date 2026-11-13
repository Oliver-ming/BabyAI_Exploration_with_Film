"""支线 3 · 机制验证：DAgger 为什么连「表征类」失败也降低了？

发现（reports/b3_dagger_failure_compare.json）：DAgger 把表征类失败
（attribute_confusion / distractor_priority）从 45 降到 26，
这**不符合**"DAgger 只修状态覆盖"的朴素预期。本脚本检验一个替代机制：

  专家的最优轨迹**几乎从不面向错误属性的物体**（它直接去目标），
  所以「面前是 purple key，但指令要 purple box」这个状态在专家数据里几乎无监督信号。
  而策略会走偏，DAgger 采集正好把它带进这些状态，并给出"别停在这，走开"的标签。

如果本假设成立，应当看到：
  H1. DAgger 数据里「面前是**部分属性匹配**的物体」的状态占比，显著高于专家数据
  H2. 这些状态下专家的标签**不是** done/stopped，而是转向离开（left/right）

用法:
    python branches/b3_dagger/verify_attribute_mechanism.py
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def front_cell_of(env, obs):
    """由 obs 反推 agent 面前那格的 (type, color) —— 用 image 第 3 行中心像素。

    Minigrid 的 7×7 视野里，agent 自己位于最下行中心，正前方是 (view_size-2, center)。
    """
    from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

    img = obs["image"]
    h = img.shape[0]
    c = img.shape[1] // 2
    cell = img[h - 2, c]
    return IDX_TO_OBJECT[int(cell[0])], IDX_TO_COLOR[int(cell[1])]


def scan(episodes, env_id: str, name: str) -> dict:
    """统计「面前是部分属性匹配物体」的状态比例，及其标签分布。"""
    env = make_env(env_id)
    stats = collections.Counter()
    label_dist = collections.Counter()
    n_states = 0

    with contextlib.redirect_stdout(__import__("io").StringIO()):
        for e in episodes:
            parts = e["mission"].split()
            color, obj_type = parts[-2], parts[-1]
            obs, _ = env.reset(seed=e["seed"])
            for t, (img, d, a) in enumerate(
                zip(e["images"], e["directions"], e["actions"])
            ):
                # 直接用记录下来的 image/direction 构造一个伪 obs
                fake = {"image": img, "direction": d, "mission": e["mission"]}
                otype, ocolor = front_cell_of(env, fake)
                n_states += 1
                if otype == "empty" or otype == "wall":
                    continue
                if otype not in ("ball", "box", "key"):
                    continue
                same_type = otype == obj_type
                same_color = ocolor == color
                if same_type and same_color:
                    stats["exact_target"] += 1
                elif same_type or same_color:
                    stats["partial_match"] += 1
                    label_dist[f"{'same_type' if same_type else 'same_color'}:{ACTION_NAMES[a]}"] += 1
                else:
                    stats["irrelevant_obj"] += 1
                if t >= len(e["images"]) - 1:
                    break
    env.close()

    n_obj = stats["exact_target"] + stats["partial_match"] + stats["irrelevant_obj"]
    return {
        "name": name,
        "n_episodes": len(episodes),
        "n_states": n_states,
        "n_states_facing_object": n_obj,
        "pct_facing_partial_match": round(stats["partial_match"] / max(1, n_obj), 4),
        "pct_facing_exact_target": round(stats["exact_target"] / max(1, n_obj), 4),
        "pct_facing_irrelevant": round(stats["irrelevant_obj"] / max(1, n_obj), 4),
        "label_dist_when_partial": dict(label_dist.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/b3_dagger_attribute_mechanism.json")
    args = ap.parse_args()

    expert = [e for e in pickle.load(open("data/expert_5000.pkl", "rb")) if e["split"] == "train"]
    dg = pickle.load(open("data/b3_dagger1500_pol0.pkl", "rb"))["episodes"]
    # 专家轨迹很短，DAgger 很长；为了可比，各取相同 episode 数的**前 8 步**
    expert_c = [dict(e, images=e["images"][:8], directions=e["directions"][:8], actions=e["actions"][:8]) for e in expert]
    dg_c = [dict(e, images=e["images"][:8], directions=e["directions"][:8], actions=e["actions"][:8]) for e in dg]

    r_exp = scan(expert_c, "BabyAI-GoToLocalS6N3-v0", "expert_train(first8)")
    r_dg = scan(dg_c, "BabyAI-GoToLocalS6N3-v0", "dagger(first8)")

    out = {
        "hypothesis": "DAgger 提供了专家数据里缺失的「面向错误属性物体」的负样本",
        "results": [r_exp, r_dg],
        "verdict": {},
    }
    lift = r_dg["pct_facing_partial_match"] - r_exp["pct_facing_partial_match"]
    out["verdict"] = {
        "H1_partial_match_ratio_lift": round(lift, 4),
        "H1_supported": lift > 0.02,
        "expert_counts": {k: r_exp[k] for k in ("pct_facing_partial_match", "pct_facing_exact_target", "pct_facing_irrelevant")},
        "dagger_counts": {k: r_dg[k] for k in ("pct_facing_partial_match", "pct_facing_exact_target", "pct_facing_irrelevant")},
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"{'set':28s} {'facing obj':>11s} {'partial':>9s} {'exact':>8s} {'other':>8s}")
    for r in (r_exp, r_dg):
        print(
            f"{r['name']:28s} {r['n_states_facing_object']:11d} "
            f"{r['pct_facing_partial_match']:9.1%} {r['pct_facing_exact_target']:8.1%} "
            f"{r['pct_facing_irrelevant']:8.1%}"
        )
    print(f"\n部分匹配状态占比提升 {lift:+.1%} → H1 {'成立' if out['verdict']['H1_supported'] else '不成立'}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
