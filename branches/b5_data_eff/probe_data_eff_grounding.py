"""支线 5 · 语义接地随数据规模演化的反事实探针（183 个可切换目标状态）。

检验「颜色/类型语义绑定究竟需要多少样本才会涌现？」，对比 Concat vs FiLM 以及 D4 群增广的作用。

用法:
    OMP_NUM_THREADS=1 python branches/b5_data_eff/probe_data_eff_grounding.py \
        --workers 16 --out reports/b5_data_eff_sensitivity.json
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
sys.path.insert(0, str(ROOT / "branches" / "b5_data_eff"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from common import TEST_SEEDS  # noqa: E402
from data_eff_policy import load_any_run, make_policy_fn  # noqa: E402
from data_eff_utils import frac_tag  # noqa: E402

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


def build_probe_tasks() -> list[dict]:
    tasks = []
    for s in (0, 1, 2):
        tasks.append({"run": f"runs/b3_bc30_seed{s}", "method": "bc_step", "frac": 1.0, "ftag": "f100", "d4_tta": False})
        tasks.append({"run": f"runs/b2_film_nwd_seed{s}", "method": "film_step", "frac": 1.0, "ftag": "f100", "d4_tta": False})
    for m in ("bc_step", "bc_d4", "film_step", "film_d4"):
        for f in (0.05, 0.10, 0.20, 0.50, 1.00):
            if m in ("bc_step", "film_step") and f == 1.00:
                continue
            ft = frac_tag(f)
            for s in (0, 1, 2):
                p = f"runs/b5_{m}_{ft}_seed{s}"
                if Path(p).exists():
                    tasks.append({"run": p, "method": m, "frac": f, "ftag": ft, "d4_tta": False})
                    if m in ("bc_d4", "film_d4") and f == 0.20:
                        tasks.append({"run": p, "method": f"{m}_tta", "frac": f, "ftag": ft, "d4_tta": True})
    for m in ("bc_naive", "bc_reg", "bc_z2", "bc_c4", "bc_wd", "bc_bal", "gru_small_step"):
        for s in (0, 1, 2):
            p = f"runs/b5_{m}_f020_seed{s}"
            if Path(p).exists():
                tasks.append({"run": p, "method": m, "frac": 0.20, "ftag": "f020", "d4_tta": False})
    return tasks


def probe_task(args: tuple[dict, list[dict]]) -> dict:
    spec, states = args
    torch.set_num_threads(1)
    model, cfg, _ = load_any_run(spec["run"])
    policy = make_policy_fn(model, d4_tta=spec["d4_tta"])

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
        "run": spec["run"],
        "method": spec["method"],
        "frac": spec["frac"],
        "frac_tag": spec["ftag"],
        "d4_tta": spec["d4_tta"],
        "train_seed": cfg.get("seed"),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "n_states": len(states),
        "color_swap_sensitivity": round(color_changed / max(1, color_total), 4),
        "type_swap_sensitivity": round(type_changed / max(1, type_total), 4),
        "combined_sensitivity": round((color_changed + type_changed) / max(1, color_total + type_total), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="reports/b5_data_eff_sensitivity.json")
    args = ap.parse_args()

    states = collect_states(args.n_states, scan=len(TEST_SEEDS))
    print(f"采集到 {len(states)} 个可切换候选物体的测试状态")

    specs = build_probe_tasks()
    tasks = [(sp, states) for sp in specs]
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            results = list(ex.map(probe_task, tasks))
    else:
        results = [probe_task(t) for t in tasks]

    agg: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in results:
        agg[(r["method"], r["frac_tag"])].append(r)

    by_group: dict[str, dict] = {}
    for (m, ft), rs in sorted(agg.items()):
        by_group[f"{m}_{ft}"] = {
            "method": m,
            "frac": rs[0]["frac"],
            "frac_tag": ft,
            "n": len(rs),
            "color_sensitivity_mean": round(float(np.mean([x["color_swap_sensitivity"] for x in rs])), 4),
            "color_sensitivity_sd": round(float(np.std([x["color_swap_sensitivity"] for x in rs])), 4),
            "type_sensitivity_mean": round(float(np.mean([x["type_swap_sensitivity"] for x in rs])), 4),
            "type_sensitivity_sd": round(float(np.std([x["type_swap_sensitivity"] for x in rs])), 4),
            "combined_sensitivity_mean": round(float(np.mean([x["combined_sensitivity"] for x in rs])), 4),
        }

    summary = {
        "n_states": len(states),
        "probe_definition": "固定 (image, direction)，更换同色异类或同类异色目标后贪心动作改变率",
        "runs": results,
        "by_group": by_group,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'group':22s} {'color':>8s} {'type':>8s} {'combined':>9s}")
    for g, a in by_group.items():
        print(f"  {g:20s} {a['color_sensitivity_mean']:8.4f} {a['type_sensitivity_mean']:8.4f} {a['combined_sensitivity_mean']:9.4f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
