"""支线 2 · 语言敏感度探针（反事实测试）。

**为什么必须做**：支线 3 的教训是「Success 涨了不等于机制起作用了」。
如果 FiLM 的 Success 提升来自「多了 2k 参数带来的正则效应」，那它与跨模态融合无关。
本探针直接测模型到底有多依赖 mission。

方法：
  对每个测试状态 (image, direction, 原 mission)：
    1. 换颜色 → 同类型、换一个场景里确实存在的颜色
    2. 换类型 → 同颜色、换一个场景里确实存在的类型
  只保留「场景里确实存在这个替换目标」的状态（否则换指令本来就该无动作变化，
  会人为压低敏感度）。
  敏感度 = 贪心动作发生改变的比例。

用法:
    python branches/b2_film/probe_language.py \
        --runs runs/b3_bc30_seed0 runs/b2_film_seed0 \
        --n-states 600 --out reports/b2_language_sensitivity.json
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
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from common import TEST_SEEDS  # noqa: E402
from film_policy import load_any_run, make_policy_fn  # noqa: E402

COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
TYPES = ["ball", "box", "key"]


def scene_objects(u) -> list[tuple[int, int, str, str]]:
    """场景里所有可交互物体 (x, y, type, color)。"""
    out = []
    for i in range(u.grid.width):
        for j in range(u.grid.height):
            c = u.grid.get(i, j)
            if c is not None and c.type in TYPES:
                out.append((i, j, c.type, c.color))
    return out


def collect_states(n_states: int, scan: int = 300) -> list[dict]:
    """在测试 seed 上采集「存在可切换候选」的状态，并生成反事实 mission。"""
    env = make_env()
    states = []
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TEST_SEEDS[:scan]:
            if len(states) >= n_states:
                break
            obs, _ = env.reset(seed=seed)
            u = env.unwrapped
            mission = obs["mission"]
            objs = scene_objects(u)

            parts = mission.split()
            if len(parts) < 5:
                continue
            article, color, otype = parts[2], parts[-2], parts[-1]

            alt_color = next(
                (c for c in COLORS
                 if c != color and any(o[3] == c and o[2] == otype for o in objs)),
                None,
            )
            alt_type = next(
                (t for t in TYPES
                 if t != otype and any(o[2] == t and o[3] == color for o in objs)),
                None,
            )
            if alt_color is None and alt_type is None:
                continue  # 这个场景没有可切换的候选，换指令无意义

            states.append(
                {
                    "seed": seed,
                    "image": obs["image"].copy(),
                    "direction": int(obs["direction"]),
                    "mission": mission,
                    "color": color,
                    "type": otype,
                    "article": article,
                    "alt_color_mission": f"go to {article} {alt_color} {otype}" if alt_color else None,
                    "alt_type_mission": f"go to {article} {color} {alt_type}" if alt_type else None,
                }
            )
    env.close()
    return states


def act(policy, image, direction, mission) -> int:
    return policy({"image": image, "direction": direction, "mission": mission})


def probe_run(run: str, states: list[dict]) -> dict:
    model, cfg, _ = load_any_run(run)
    policy = make_policy_fn(model)

    color_changed = color_total = 0
    type_changed = type_total = 0
    for s in states:
        base = act(policy, s["image"], s["direction"], s["mission"])
        if s["alt_color_mission"]:
            color_total += 1
            if act(policy, s["image"], s["direction"], s["alt_color_mission"]) != base:
                color_changed += 1
        if s["alt_type_mission"]:
            type_total += 1
            if act(policy, s["image"], s["direction"], s["alt_type_mission"]) != base:
                type_changed += 1

    return {
        "run": run,
        "variant": cfg.get("arch", {}).get("variant", "concat"),
        "train_seed": cfg.get("seed"),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "n_states": len(states),
        "color_swap_sensitivity": round(color_changed / max(1, color_total), 4),
        "type_swap_sensitivity": round(type_changed / max(1, type_total), 4),
        "combined_sensitivity": round(
            (color_changed + type_changed) / max(1, color_total + type_total), 4
        ),
        "n_color_pairs": color_total,
        "n_type_pairs": type_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-states", type=int, default=600)
    ap.add_argument("--out", default="reports/b2_language_sensitivity.json")
    args = ap.parse_args()

    states = collect_states(args.n_states)
    arts = dict(collections.Counter(s["article"] for s in states))
    print(f"采集到 {len(states)} 个可切换候选的状态（冠词分布 {arts}）")

    results = [probe_run(r, states) for r in args.runs]

    agg: dict[str, list] = collections.defaultdict(list)
    for r in results:
        agg[r["variant"]].append(r)

    summary = {
        "n_states": len(states),
        "article_distribution": arts,
        "probe_definition": (
            "固定 (image, direction)，把 mission 换成「同色异类」或「同类异色」，"
            "贪心动作改变的比例"
        ),
        "runs": results,
        "by_variant": {
            v: {
                "n": len(rs),
                "combined_sensitivity_mean": round(float(np.mean([x["combined_sensitivity"] for x in rs])), 4),
                "color_sensitivity_mean": round(float(np.mean([x["color_swap_sensitivity"] for x in rs])), 4),
                "type_sensitivity_mean": round(float(np.mean([x["type_swap_sensitivity"] for x in rs])), 4),
            }
            for v, rs in agg.items()
        },
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'run':34s} {'variant':11s} {'val_acc':>8s}  颜色换 类型换 合计")
    for r in results:
        print(
            f"{Path(r['run']).name:34s} {r['variant']:11s} {r['val_action_acc']:8.3f}  "
            f"{r['color_swap_sensitivity']:6.3f} {r['type_swap_sensitivity']:6.3f} "
            f"{r['combined_sensitivity']:6.3f}"
        )
    print("\n按 variant 平均：")
    for v, a in summary["by_variant"].items():
        print(f"  {v:11s} n={a['n']} 颜色={a['color_sensitivity_mean']:.3f} "
              f"类型={a['type_sensitivity_mean']:.3f} 合计={a['combined_sensitivity_mean']:.3f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
