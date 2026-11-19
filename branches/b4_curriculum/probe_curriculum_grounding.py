"""支线 4 · 语义接地反事实探针（183 个可切换候选目标 S6N3 状态，复刻 b2_film/probe_language.py 口径）。

检验课程学习与跨环境域随机化是否真正提升了颜色/类型指令依赖度，还是仅依靠视觉快捷路径获益。

用法:
    OMP_NUM_THREADS=1 python branches/b4_curriculum/probe_curriculum_grounding.py \
        --runs runs/b3_bc30_seed* runs/b2_film_nwd_seed* runs/b4_*_seed* \
        --workers 15 --out reports/b4_curriculum_sensitivity.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b4_curriculum"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from common import TEST_SEEDS  # noqa: E402
from curriculum_policy import load_any_run, make_policy_fn  # noqa: E402
from eval_curriculum import group_name  # noqa: E402

COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
TYPES = ["ball", "box", "key"]


def scene_objects(u) -> list[tuple[int, int, str, str]]:
    out = []
    for i in range(u.grid.width):
        for j in range(u.grid.height):
            c = u.grid.get(i, j)
            if c is not None and c.type in TYPES:
                out.append((i, j, c.type, c.color))
    return out


def collect_states(n_states: int = 600, scan: int = 300) -> list[dict]:
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
                (c for c in COLORS if c != color and any(o[3] == c and o[2] == otype for o in objs)),
                None,
            )
            alt_type = next(
                (t for t in TYPES if t != otype and any(o[2] == t and o[3] == color for o in objs)),
                None,
            )
            if alt_color is None and alt_type is None:
                continue
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


def probe_task(task: tuple[str, list[dict]]) -> dict:
    run, states = task
    torch.set_num_threads(1)
    model, cfg, _ = load_any_run(run)
    policy = make_policy_fn(model)
    grp = group_name(run, cfg)

    color_changed = color_total = 0
    type_changed = type_total = 0
    for s in states:
        base = policy({"image": s["image"], "direction": s["direction"], "mission": s["mission"]})
        if s["alt_color_mission"]:
            color_total += 1
            if policy({"image": s["image"], "direction": s["direction"], "mission": s["alt_color_mission"]}) != base:
                color_changed += 1
        if s["alt_type_mission"]:
            type_total += 1
            if policy({"image": s["image"], "direction": s["direction"], "mission": s["alt_type_mission"]}) != base:
                type_changed += 1

    return {
        "run": run,
        "group": grp,
        "variant": grp,
        "train_seed": cfg.get("seed"),
        "use_film": bool(cfg.get("arch", {}).get("use_film", False)),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "color_swap_sensitivity": round(color_changed / max(1, color_total), 4),
        "type_swap_sensitivity": round(type_changed / max(1, type_total), 4),
        "combined_sensitivity": round((color_changed + type_changed) / max(1, color_total + type_total), 4),
        "n_color_pairs": color_total,
        "n_type_pairs": type_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-states", type=int, default=600)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default="reports/b4_curriculum_sensitivity.json")
    args = ap.parse_args()

    runs = list(dict.fromkeys(args.runs))
    states = collect_states(args.n_states, scan=len(TEST_SEEDS))
    print(f"采集到 {len(states)} 个可切换候选物体的测试状态")

    tasks = [(r, states) for r in runs]
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            results = list(ex.map(probe_task, tasks))
    else:
        results = [probe_task(t) for t in tasks]

    agg: dict[str, list] = collections.defaultdict(list)
    for r in results:
        agg[r["group"]].append(r)

    summary = {
        "n_states": len(states),
        "probe_definition": "固定 (image, direction)，将 mission 替换为场景中实际存在的同色异类或同类异色目标，统计贪心动作改变率",
        "runs": results,
        "by_variant": {
            v: {
                "n": len(rs),
                "use_film": rs[0]["use_film"],
                "combined_sensitivity_mean": round(float(np.mean([x["combined_sensitivity"] for x in rs])), 4),
                "combined_sensitivity_sd": round(float(np.std([x["combined_sensitivity"] for x in rs])), 4),
                "color_sensitivity_mean": round(float(np.mean([x["color_swap_sensitivity"] for x in rs])), 4),
                "color_sensitivity_sd": round(float(np.std([x["color_swap_sensitivity"] for x in rs])), 4),
                "type_sensitivity_mean": round(float(np.mean([x["type_swap_sensitivity"] for x in rs])), 4),
                "type_sensitivity_sd": round(float(np.std([x["type_swap_sensitivity"] for x in rs])), 4),
            }
            for v, rs in agg.items()
        },
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'group':22s} {'n':>2s} {'color':>8s} {'type':>8s} {'combined':>9s}")
    for v, a in sorted(summary["by_variant"].items()):
        print(
            f"{v:22s} {a['n']:2d} {a['color_sensitivity_mean']:8.4f} "
            f"{a['type_sensitivity_mean']:8.4f} {a['combined_sensitivity_mean']:9.4f}"
        )
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
