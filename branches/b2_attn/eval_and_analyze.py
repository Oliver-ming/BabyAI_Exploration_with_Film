"""支线 2 扩展消融 · 全量评估、机制探针、失败拆解、个案核验与 300-DPI 作图脚本。

运行:
    OMP_NUM_THREADS=1 env_embodied/bin/python branches/b2_attn/eval_and_analyze.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import collections
import contextlib
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b2_attn"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from attn_net import ATTN_VARIANTS, build as build_attn  # noqa: E402
from branch_utils import make_env  # noqa: E402
from common import (  # noqa: E402
    ENV_ID,
    ENV_ID_S7,
    TEST_SEEDS,
    TEST_SEEDS_S7,
    encode_batch,
)
from compare_failures import MECHANISM, run_episode_with_policy  # noqa: E402
from eval_closed_loop import evaluate  # noqa: E402
from film_policy import load_any_run, make_policy_fn  # noqa: E402
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT  # noqa: E402
from probe_language import collect_states, probe_run as probe_film_or_bc  # noqa: E402
from render_episodes import classify_failure  # noqa: E402

ENV_ID_S8 = "BabyAI-GoToLocalS8N6-v0"
SEEDS = [0, 1, 2]
F020_VARIANTS = ("spatial_attn_gate", "cross_attn_mha", "film_spatial_attn")
CASE_SEEDS = [4501, 4516, 4560, 4568, 4502]

LABELS = {
    "bc30": "BC30 (Concat)",
    "film_nwd": "FiLM-nwd (Primary)",
    "spatial_attn_gate": "Spatial-Gate",
    "spatial_softmax_attn": "Spatial-Softmax",
    "cross_attn_mha": "Cross-MHA",
    "film_spatial_attn": "FiLM+Spatial",
    "film_cross_mha": "FiLM+MHA",
}


def load_run_universal(run_dir: str | Path):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg.get("branch") == "b2_attn":
        vocab = json.loads((run_dir / "vocab.json").read_text())
        a = cfg["arch"]
        model = build_attn(
            a["variant"],
            vocab_size=len(vocab["vocab"]),
            obj_emb_dim=a["obj_emb_dim"],
            lang_dim=a["lang_dim"],
            hidden=a.get("hidden", 128),
            dropout=a.get("dropout", 0.0),
        )
        state = torch.load(run_dir / "best_model.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, cfg, vocab
    return load_any_run(run_dir)


def mean_sd(vals: list[float]) -> dict:
    vals = [float(v) for v in vals]
    return {
        "n": len(vals),
        "mean": round(st.mean(vals), 6),
        "sd": round(st.pstdev(vals), 6),  # ddof=0 与仓库 JSON 严格一致
        "values": [round(v, 6) for v in vals],
    }


def eval_run(run_dir: str, env_id: str, seeds: list[int]) -> dict:
    model, cfg, _ = load_run_universal(run_dir)
    t0 = time.time()
    res = evaluate(make_policy_fn(model), env_id, seeds)
    res.pop("_trajectories", None)
    return {
        "run": run_dir,
        "variant": cfg.get("arch", {}).get("variant", "concat"),
        "branch": cfg.get("branch", "baseline"),
        "train_seed": cfg.get("seed"),
        "frac": cfg.get("frac", 1.0),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "best_epoch": cfg.get("result", {}).get("best_epoch"),
        "n_params": cfg.get("arch", {}).get("n_params"),
        "train_seconds": cfg.get("result", {}).get("train_seconds"),
        "env_id": env_id,
        "eval_seconds": round(time.time() - t0, 2),
        **res,
    }


def summarize_group(runs: list[dict]) -> dict:
    runs = sorted(runs, key=lambda r: r["train_seed"])
    return {
        "n_params": runs[0]["n_params"],
        "closed_loop_success": mean_sd([r["closed_loop_success"] for r in runs]),
        "val_action_acc": mean_sd([r["val_action_acc"] for r in runs]),
        "mean_return": mean_sd([r["mean_return"] for r in runs]),
        "median_len_success": mean_sd([r["median_len_success"] for r in runs]),
        "n_timeout": mean_sd([r["n_timeout"] for r in runs]),
    }


def paired_delta(runs: list[dict], ref_runs: list[dict]) -> dict:
    rmap = {r["train_seed"]: r for r in runs}
    refmap = {r["train_seed"]: r for r in ref_runs}
    diffs = []
    rows = []
    for s in SEEDS:
        d = round(rmap[s]["closed_loop_success"] - refmap[s]["closed_loop_success"], 6)
        diffs.append(d)
        rows.append(
            {
                "seed": s,
                "variant": rmap[s]["closed_loop_success"],
                "ref": refmap[s]["closed_loop_success"],
                "delta": d,
            }
        )
    return {
        "n_pairs": len(diffs),
        "delta_mean": round(st.mean(diffs), 6),
        "delta_sd": round(st.pstdev(diffs), 6),
        "n_positive": sum(1 for d in diffs if d > 0),
        "rows": rows,
    }


def run_main_eval() -> dict:
    print("== 1. Running S6N3 / S7N4 / S8N6 evaluations ==")
    b2_s6 = json.loads((ROOT / "reports/b2_film_eval.json").read_text())
    b2_s7 = json.loads((ROOT / "reports/b2_film_s7n4.json").read_text())

    bc30_s6 = [r for r in b2_s6["baseline_runs"] if "b3_bc30" in r["run"]]
    film_s6 = [r for r in b2_s6["runs"] if "b2_film_nwd" in r["run"]]
    bc30_s7 = [r for r in b2_s7["baseline_runs"] if "b3_bc30" in r["run"]]
    film_s7 = [r for r in b2_s7["runs"] if "b2_film_nwd" in r["run"]]

    # S8N6: evaluate BC30, FiLM-nwd, and all 5 attn variants on 10000:10300 (plus 4500:4800 ref for BC30/FiLM?)
    # Using TEST_SEEDS_S7 (10000:10300) as clean unseen 300-seed OOD set for S8N6 across all methods
    bc30_s8 = [eval_run(f"runs/b3_bc30_seed{s}", ENV_ID_S8, TEST_SEEDS_S7) for s in SEEDS]
    film_s8 = [eval_run(f"runs/b2_film_nwd_seed{s}", ENV_ID_S8, TEST_SEEDS_S7) for s in SEEDS]

    results = {
        "branch": "b2_attn",
        "sd_convention": "ddof=0 (population SD via statistics.pstdev)",
        "seeds_s6n3": "4500:4800",
        "seeds_s7n4": "10000:10300",
        "seeds_s8n6": "10000:10300",
        "references": {
            "bc30": {
                "S6N3": summarize_group(bc30_s6),
                "S7N4": summarize_group(bc30_s7),
                "S8N6": summarize_group(bc30_s8),
            },
            "film_nwd": {
                "S6N3": summarize_group(film_s6),
                "S7N4": summarize_group(film_s7),
                "S8N6": summarize_group(film_s8),
                "paired_vs_bc30": {
                    "S6N3": paired_delta(film_s6, bc30_s6),
                    "S7N4": paired_delta(film_s7, bc30_s7),
                    "S8N6": paired_delta(film_s8, bc30_s8),
                },
            },
        },
        "variants": {},
        "runs": bc30_s8 + film_s8,
    }

    for v in ATTN_VARIANTS:
        r_s6 = [eval_run(f"runs/b2_attn_{v}_seed{s}", ENV_ID, TEST_SEEDS) for s in SEEDS]
        r_s7 = [eval_run(f"runs/b2_attn_{v}_seed{s}", ENV_ID_S7, TEST_SEEDS_S7) for s in SEEDS]
        r_s8 = [eval_run(f"runs/b2_attn_{v}_seed{s}", ENV_ID_S8, TEST_SEEDS_S7) for s in SEEDS]
        results["runs"].extend(r_s6 + r_s7 + r_s8)
        results["variants"][v] = {
            "S6N3": summarize_group(r_s6),
            "S7N4": summarize_group(r_s7),
            "S8N6": summarize_group(r_s8),
            "paired_vs_bc30": {
                "S6N3": paired_delta(r_s6, bc30_s6),
                "S7N4": paired_delta(r_s7, bc30_s7),
                "S8N6": paired_delta(r_s8, bc30_s8),
            },
            "paired_vs_film_nwd": {
                "S6N3": paired_delta(r_s6, film_s6),
                "S7N4": paired_delta(r_s7, film_s7),
                "S8N6": paired_delta(r_s8, film_s8),
            },
        }
        s6m = results["variants"][v]["S6N3"]["closed_loop_success"]
        s7m = results["variants"][v]["S7N4"]["closed_loop_success"]
        s8m = results["variants"][v]["S8N6"]["closed_loop_success"]
        print(f"  [{v:20s}] S6N3={s6m['mean']:.3f}±{s6m['sd']:.3f} | S7N4={s7m['mean']:.3f}±{s7m['sd']:.3f} | S8N6={s8m['mean']:.3f}±{s8m['sd']:.3f}")

    (ROOT / "reports/b2_attn_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def run_f020_eval() -> dict:
    print("== 2. Running 20% step-matched low-data evaluations ==")
    b5_s6 = json.loads((ROOT / "reports/b5_data_eff_eval.json").read_text())
    b5_s7 = json.loads((ROOT / "reports/b5_data_eff_s7n4.json").read_text())

    out = {
        "frac": 0.2,
        "n_episodes": 900,
        "step_matched_to": 16534,
        "references": {
            "bc_step_f020": {
                "S6N3": b5_s6["scaling_curves"]["bc_step"]["f020"],
                "S7N4": b5_s7["scaling_curves"]["bc_step"]["f020"],
            },
            "film_step_f020": {
                "S6N3": b5_s6["scaling_curves"]["film_step"]["f020"],
                "S7N4": b5_s7["scaling_curves"]["film_step"]["f020"],
            },
        },
        "variants": {},
        "runs": [],
    }

    for v in F020_VARIANTS:
        r_s6 = [eval_run(f"runs/b2_attn_{v}_f020_seed{s}", ENV_ID, TEST_SEEDS) for s in SEEDS]
        r_s7 = [eval_run(f"runs/b2_attn_{v}_f020_seed{s}", ENV_ID_S7, TEST_SEEDS_S7) for s in SEEDS]
        out["runs"].extend(r_s6 + r_s7)
        out["variants"][v] = {
            "S6N3": summarize_group(r_s6),
            "S7N4": summarize_group(r_s7),
        }
        s6m = out["variants"][v]["S6N3"]["closed_loop_success"]
        s7m = out["variants"][v]["S7N4"]["closed_loop_success"]
        print(f"  [{v:20s} f020] S6N3={s6m['mean']:.3f}±{s6m['sd']:.3f} | S7N4={s7m['mean']:.3f}±{s7m['sd']:.3f}")

    (ROOT / "reports/b2_attn_f020_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def probe_single(run_dir: str, states: list[dict]) -> dict:
    model, cfg, _ = load_run_universal(run_dir)
    policy = make_policy_fn(model)
    c_chg = c_tot = t_chg = t_tot = 0
    for s in states:
        with torch.no_grad():
            base = policy({"image": s["image"], "direction": s["direction"], "mission": s["mission"]})
            if s["alt_color_mission"]:
                c_tot += 1
                if policy({"image": s["image"], "direction": s["direction"], "mission": s["alt_color_mission"]}) != base:
                    c_chg += 1
            if s["alt_type_mission"]:
                t_tot += 1
                if policy({"image": s["image"], "direction": s["direction"], "mission": s["alt_type_mission"]}) != base:
                    t_chg += 1
    return {
        "run": run_dir,
        "variant": cfg.get("arch", {}).get("variant", "concat"),
        "train_seed": cfg.get("seed"),
        "color_total": c_tot,
        "color_changed": c_chg,
        "color_swap_sensitivity": round(c_chg / max(1, c_tot), 6),
        "type_total": t_tot,
        "type_changed": t_chg,
        "type_swap_sensitivity": round(t_chg / max(1, t_tot), 6),
        "combined_sensitivity": round((c_chg + t_chg) / max(1, c_tot + t_tot), 6),
    }


def run_probe() -> dict:
    print("== 3. Running 183-state counterfactual language probe ==")
    states = collect_states(600, scan=300)
    groups = {
        "bc30": [f"runs/b3_bc30_seed{s}" for s in SEEDS],
        "film_nwd": [f"runs/b2_film_nwd_seed{s}" for s in SEEDS],
        **{v: [f"runs/b2_attn_{v}_seed{s}" for s in SEEDS] for v in ATTN_VARIANTS},
    }
    out = {"n_states": len(states), "by_variant": {}, "runs": []}
    for name, rlist in groups.items():
        rs = [probe_single(r, states) for r in rlist]
        out["runs"].extend(rs)
        out["by_variant"][name] = {
            "color_swap_sensitivity": mean_sd([r["color_swap_sensitivity"] for r in rs]),
            "type_swap_sensitivity": mean_sd([r["type_swap_sensitivity"] for r in rs]),
            "combined_sensitivity": mean_sd([r["combined_sensitivity"] for r in rs]),
        }
        cs = out["by_variant"][name]["color_swap_sensitivity"]
        ts = out["by_variant"][name]["type_swap_sensitivity"]
        print(f"  [{name:20s}] color={cs['mean']:.3f}±{cs['sd']:.3f} | type={ts['mean']:.3f}±{ts['sd']:.3f}")
    (ROOT / "reports/b2_attn_probe.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def analyze_run_failures(run_dir: str) -> tuple[dict, dict]:
    model, cfg, _ = load_run_universal(run_dir)
    policy = make_policy_fn(model)
    env = make_env(ENV_ID)
    trajs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TEST_SEEDS:
            trajs.append(run_episode_with_policy(env, policy, seed))
    env.close()

    types = collections.Counter()
    groups = collections.Counter()
    cases = {}
    tmap = {t["seed"]: t for t in trajs}
    for t in trajs:
        if not t["success"]:
            ft, reason = classify_failure(t)
            types[ft] += 1
            groups[MECHANISM.get(ft, "other")] += 1
    for cs in CASE_SEEDS:
        t = tmap[cs]
        ft = None if t["success"] else classify_failure(t)[0]
        cases[str(cs)] = {
            "success": t["success"],
            "episode_len": t["episode_len"],
            "failure_type": ft,
            "mission": t["mission"],
            "final_pos": t["final_agent_pos"],
            "final_front_cell": t["final_front_cell"],
        }
    summary = {
        "run": run_dir,
        "variant": cfg.get("arch", {}).get("variant", "concat"),
        "train_seed": cfg.get("seed"),
        "n_success": sum(1 for t in trajs if t["success"]),
        "n_failure": sum(1 for t in trajs if not t["success"]),
        "failure_types": dict(types),
        "failure_groups": dict(groups),
    }
    return summary, cases


def run_failures() -> dict:
    print("== 4. Running 300-episode failure breakdown & 5-case audit ==")
    groups = {
        "bc30": [f"runs/b3_bc30_seed{s}" for s in SEEDS],
        "film_nwd": [f"runs/b2_film_nwd_seed{s}" for s in SEEDS],
        **{v: [f"runs/b2_attn_{v}_seed{s}" for s in SEEDS] for v in ATTN_VARIANTS},
    }
    out = {"env_id": ENV_ID, "seeds": "4500:4800", "case_seeds": CASE_SEEDS, "aggregate": {}, "cases": {}, "runs": []}
    for name, rlist in groups.items():
        rs, cs_list = [], []
        for r in rlist:
            s_res, c_res = analyze_run_failures(r)
            rs.append(s_res)
            cs_list.append(c_res)
            out["runs"].append(s_res)
        all_types = sorted({k for r in rs for k in r["failure_types"]})
        out["aggregate"][name] = {
            "n_failure": mean_sd([r["n_failure"] for r in rs]),
            "representation": mean_sd([r["failure_groups"].get("representation", 0) for r in rs]),
            "state_coverage": mean_sd([r["failure_groups"].get("state_coverage", 0) for r in rs]),
            "failure_types_mean": {
                k: round(sum(r["failure_types"].get(k, 0) for r in rs) / len(rs), 2) for k in all_types
            },
        }
        out["cases"][name] = {}
        for cs in CASE_SEEDS:
            k = str(cs)
            per_s = [c[k] for c in cs_list]
            out["cases"][name][k] = {
                "n_solved": sum(1 for x in per_s if x["success"]),
                "per_seed": per_s,
            }
        ag = out["aggregate"][name]
        print(
            f"  [{name:20s}] fail={ag['n_failure']['mean']:4.1f} "
            f"(repr={ag['representation']['mean']:4.1f}, cov={ag['state_coverage']['mean']:4.1f}) "
            + " ".join(f"s{cs}:{out['cases'][name][str(cs)]['n_solved']}/3" for cs in CASE_SEEDS)
        )
    (ROOT / "reports/b2_attn_failures.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def make_comparison_figure(ev: dict, f20: dict, pr: dict, fl: dict) -> None:
    print("== 5. Generating 300-DPI comparison figure ==")
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    order = [
        "bc30",
        "film_nwd",
        "spatial_attn_gate",
        "spatial_softmax_attn",
        "cross_attn_mha",
        "film_spatial_attn",
        "film_cross_mha",
    ]
    short = [
        "BC30\n(Concat)",
        "FiLM\n(Primary)",
        "Spatial\nGate",
        "Spatial\nSoftmax",
        "Cross\nMHA",
        "FiLM +\nSpatial",
        "FiLM +\nMHA",
    ]
    colors = ["#D97706", "#059669", "#2563EB", "#3B82F6", "#6366F1", "#0D9488", "#14B8A6"]

    def get_ms(key, env_tag):
        if key in ("bc30", "film_nwd"):
            d = ev["references"][key][env_tag]["closed_loop_success"]
        else:
            d = ev["variants"][key][env_tag]["closed_loop_success"]
        return d["mean"], d["sd"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=300)

    # Panel (a): S6N3, S7N4, S8N6
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.26
    for idx, (env_tag, hatch, alpha, lbl) in enumerate(
        [("S6N3", "", 0.92, "S6N3 (in-dist)"), ("S7N4", "///", 0.55, "S7N4 (zero-shot)"), ("S8N6", "...", 0.35, "S8N6 (8×8 N=6)")]
    ):
        means = [get_ms(k, env_tag)[0] for k in order]
        sds = [get_ms(k, env_tag)[1] for k in order]
        bars = ax.bar(
            x + (idx - 1) * w,
            means,
            w,
            yerr=sds,
            capsize=2.2,
            color=colors,
            alpha=alpha,
            hatch=hatch,
            edgecolor="#1F2937",
            linewidth=0.7,
            label=lbl,
        )
        if env_tag == "S6N3":
            for b, m in zip(bars, means):
                ax.text(b.get_x() + b.get_width() / 2, m + 0.018, f"{m:.3f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7.6)
    ax.set_ylim(0.35, 1.07)
    ax.set_ylabel("Closed-loop Success (mean ± SD, n=3)")
    ax.set_title("(a) In-Distribution & Cross-Scale Transfer")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.3, frameon=False, loc="lower left")

    # Panel (b): 20% Low-Data Efficiency (step-matched) & Counterfactual Color Sensitivity
    ax = axes[1]
    f20_keys = ["bc30", "film_nwd", "spatial_attn_gate", "cross_attn_mha", "film_spatial_attn"]
    f20_lbls = ["BC30\n(Concat)", "FiLM\n(Primary)", "Spatial\nGate", "Cross\nMHA", "FiLM +\nSpatial"]
    f20_cols = ["#D97706", "#059669", "#2563EB", "#6366F1", "#0D9488"]
    f20_means, f20_sds = [], []
    for k in f20_keys:
        if k == "bc30":
            d = f20["references"]["bc_step_f020"]["S6N3"]
            f20_means.append(d["success_mean"])
            f20_sds.append(d["success_sd"])
        elif k == "film_nwd":
            d = f20["references"]["film_step_f020"]["S6N3"]
            f20_means.append(d["success_mean"])
            f20_sds.append(d["success_sd"])
        else:
            d = f20["variants"][k]["S6N3"]["closed_loop_success"]
            f20_means.append(d["mean"])
            f20_sds.append(d["sd"])
    x2 = np.arange(len(f20_keys))
    bars2 = ax.bar(x2, f20_means, 0.52, yerr=f20_sds, capsize=3, color=f20_cols, edgecolor="#1F2937", linewidth=0.8)
    for b, m in zip(bars2, f20_means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.025, f"{m:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax.axhline(0.8144, color="#D97706", linestyle="--", linewidth=1.0, label="100% BC30 (0.814)")
    ax.set_xticks(x2)
    ax.set_xticklabels(f20_lbls, fontsize=8.0)
    ax.set_ylim(0.55, 0.96)
    ax.set_ylabel("S6N3 Success at 20% Data (900 eps)")
    ax.set_title("(b) Step-Matched 20% Low-Data Regime")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # Panel (c): Color-Swap Sensitivity vs Representation Failures
    ax = axes[2]
    cs_means = [pr["by_variant"][k]["color_swap_sensitivity"]["mean"] for k in order]
    cs_sds = [pr["by_variant"][k]["color_swap_sensitivity"]["sd"] for k in order]
    rf_means = [fl["aggregate"][k]["representation"]["mean"] for k in order]
    x3 = np.arange(len(order))
    b3 = ax.bar(x3 - 0.17, cs_means, 0.34, yerr=cs_sds, capsize=2.5, color=colors, edgecolor="#1F2937", linewidth=0.7, label="Color-swap sensitivity")
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.plot(x3 + 0.17, rf_means, "o--", color="#DC2626", linewidth=1.4, markersize=5, label="Repr. failures (/300)")
    for xi, m in zip(x3, cs_means):
        ax.text(xi - 0.17, m + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x3)
    ax.set_xticklabels(short, fontsize=7.6)
    ax.set_ylim(0, 0.36)
    ax_r.set_ylim(0, 48)
    ax.set_ylabel("Color-swap Action Flip Rate")
    ax_r.set_ylabel("Repr. Failures / 300 eps", color="#DC2626")
    ax.set_title("(c) Grounding Probe & Error Reduction")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/b2_attn/b2_attn_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_egocentric_grid_with_map(ax, img: np.ndarray, smap: np.ndarray, title: str, cmap_norm=None) -> None:
    """把 7×7 局部视野 (HWC) 与 7×7 空间注意力图 cell-for-cell 对齐绘制。"""
    # MiniGrid obs["image"] is [x_lr, y_td, 3], agent at (3, 6) facing up
    grid_types = img[:, :, 0].T  # (7 row top->bot, 7 col left->right)
    grid_cols = img[:, :, 1].T
    heat = smap.T

    rgb_map = {
        "red": "#EF4444",
        "green": "#10B981",
        "blue": "#3B82F6",
        "purple": "#9333EA",
        "yellow": "#EAB308",
        "grey": "#9CA3AF",
    }

    # Base cell tile colors
    bg = np.ones((7, 7, 3)) * 0.94
    for r in range(7):
        for c in range(7):
            t_id = int(grid_types[r, c])
            if t_id == 0:  # unseen
                bg[r, c] = [0.18, 0.19, 0.22]
            elif t_id == 2:  # wall
                bg[r, c] = [0.45, 0.48, 0.52]
    ax.imshow(bg, interpolation="nearest", extent=(-0.5, 6.5, 6.5, -0.5))

    # Overlay spatial heatmap cleanly
    if cmap_norm is None:
        im = ax.imshow(heat, cmap="YlOrRd", alpha=0.62, interpolation="nearest", extent=(-0.5, 6.5, 6.5, -0.5))
    else:
        im = ax.imshow(heat, cmap="YlOrRd", vmin=cmap_norm[0], vmax=cmap_norm[1], alpha=0.62, interpolation="nearest", extent=(-0.5, 6.5, 6.5, -0.5))

    # Draw objects and agent
    for r in range(7):
        for c in range(7):
            t_id = int(grid_types[r, c])
            c_id = int(grid_cols[r, c])
            tname = IDX_TO_OBJECT.get(t_id, "")
            cname = IDX_TO_COLOR.get(c_id, "grey")
            col = rgb_map.get(cname, "#6B7280")
            if tname == "ball":
                ax.scatter([c], [r], s=140, c=col, edgecolors="black", linewidths=1.0, zorder=4)
            elif tname == "box":
                ax.scatter([c], [r], s=140, marker="s", c=col, edgecolors="black", linewidths=1.0, zorder=4)
            elif tname == "key":
                ax.scatter([c], [r], s=140, marker="P", c=col, edgecolors="black", linewidths=1.0, zorder=4)
            if r == 6 and c == 3:
                ax.scatter([c], [r], s=120, marker="^", c="#DC2626", edgecolors="white", linewidths=0.8, zorder=5)
            # Small numeric weight on top objects / max cell
            if tname in ("ball", "box", "key") or heat[r, c] == heat.max():
                ax.text(c, r + 0.34, f"{heat[r, c]:.2f}", ha="center", va="center", fontsize=6.2, color="black",
                        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.72), zorder=6)

    for g in np.arange(-0.5, 7, 1):
        ax.axhline(g, color="#9CA3AF", linewidth=0.4, alpha=0.5)
        ax.axvline(g, color="#9CA3AF", linewidth=0.4, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.3)
    return im


def make_heatmap_figure() -> None:
    print("== 6. Generating 7x7 spatial attention heatmap figure ==")
    cols = [
        ("spatial_attn_gate", "Spatial-Gate (weight)"),
        ("spatial_softmax_attn", "Spatial-Softmax (49×α)"),
        ("cross_attn_mha", "Cross-MHA (|Δ|_2)"),
        ("film_spatial_attn", "FiLM+Spatial (weight)"),
    ]
    models = {v: load_run_universal(f"runs/b2_attn_{v}_seed0")[0] for v, _ in cols}
    env = make_env(ENV_ID)
    fig, axes = plt.subplots(3, 4, figsize=(12.2, 8.6), dpi=300)
    target_seeds = [4501, 4516, 4502]
    row_titles = [
        "Seed 4501 (t=2): go to the blue key",
        "Seed 4516 (t=2): go to the purple box",
        "Seed 4502 (t=2): go to the purple key",
    ]

    for ri, seed in enumerate(target_seeds):
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed)
            # Turn to bring room objects into the 7x7 egocentric view
            if seed == 4501:
                for a in (0, 2):  # left, forward -> yellow box ahead, blue key at (3,1)
                    obs, _, _, _, _ = env.step(a)
            elif seed in (4516, 4502):
                for a in (1, 1):  # turn around to face the room interior (West)
                    obs, _, _, _, _ = env.step(a)
        img, tok, d = encode_batch([obs])
        for ci, (v, vlabel) in enumerate(cols):
            sp = models[v].extract_spatial_map(img, tok, d)
            draw_egocentric_grid_with_map(axes[ri, ci], obs["image"], sp["spatial_map"], f"{vlabel}" if ri == 0 else "")
            if ci == 0:
                axes[ri, ci].set_ylabel(row_titles[ri], fontsize=8.5, fontweight="bold")
    env.close()
    fig.suptitle("7×7 Egocentric Spatial Attention Weights (Seed 0 Models; ▲=Agent, ●=Ball, ■=Box, ✚=Key)", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/b2_attn/b2_attn_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(ev: dict, f20: dict, pr: dict, fl: dict) -> None:
    print("== 7. Writing reports/b2_attn_summary.md ==")
    order = ["bc30", "film_nwd", "spatial_attn_gate", "spatial_softmax_attn", "cross_attn_mha", "film_spatial_attn", "film_cross_mha"]
    lines = [
        "# 支线 2 扩展消融 · FiLM vs Attention 与 FiLM+Attention 融合总结",
        "",
        "> 由 `branches/b2_attn/eval_and_analyze.py` 动态生成（`ddof=0` 总体标准差，n=3 初始化种子）。",
        "",
        "| 方法 | 参数量 | val 准确率 | **S6N3 Success** | **S7N4 迁移** | **S8N6 (10000:)** | **20% 数据 S6** | 颜色敏感度 | 表征+覆盖失败 (/300) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in order:
        if k in ("bc30", "film_nwd"):
            d6 = ev["references"][k]["S6N3"]
            d7 = ev["references"][k]["S7N4"]
            d8 = ev["references"][k]["S8N6"]
            f20_k = "bc_step_f020" if k == "bc30" else "film_step_f020"
            s20 = f"{f20['references'][f20_k]['S6N3']['success_mean']:.3f} ± {f20['references'][f20_k]['S6N3']['success_sd']:.3f}"
        else:
            d6 = ev["variants"][k]["S6N3"]
            d7 = ev["variants"][k]["S7N4"]
            d8 = ev["variants"][k]["S8N6"]
            if k in f20["variants"]:
                m20 = f20["variants"][k]["S6N3"]["closed_loop_success"]
                s20 = f"{m20['mean']:.3f} ± {m20['sd']:.3f}"
            else:
                s20 = "—"
        cs = pr["by_variant"][k]["color_swap_sensitivity"]
        fa = fl["aggregate"][k]
        lines.append(
            f"| `{k}` | {d6['n_params']:,} | {d6['val_action_acc']['mean']:.3f} ± {d6['val_action_acc']['sd']:.3f} | "
            f"**{d6['closed_loop_success']['mean']:.3f} ± {d6['closed_loop_success']['sd']:.3f}** | "
            f"{d7['closed_loop_success']['mean']:.3f} ± {d7['closed_loop_success']['sd']:.3f} | "
            f"{d8['closed_loop_success']['mean']:.3f} ± {d8['closed_loop_success']['sd']:.3f} | "
            f"{s20} | {cs['mean']:.3f} ± {cs['sd']:.3f} | "
            f"{fa['n_failure']['mean']:.1f} ({fa['representation']['mean']:.1f}+{fa['state_coverage']['mean']:.1f}) |"
        )

    lines.extend(["", "## 个案修复对比（3 seeds 通过数：4501, 4516, 4560, 4568, 4502）", "", "| 方法 | 4501 (绕障) | 4516 (紫盒) | 4560 (绿球) | 4568 (盲区) | 4502 (拥挤紫匙) |", "|---|---|---|---|---|---|"])
    for k in order:
        row = [f"`{k}`"]
        for cs in CASE_SEEDS:
            c = fl["cases"][k][str(cs)]
            lens = [str(x["episode_len"]) if x["success"] else "T" for x in c["per_seed"]]
            row.append(f"{c['n_solved']}/3 ({'/'.join(lens)})")
        lines.append("| " + " | ".join(row) + " |")

    (ROOT / "reports/b2_attn_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ev = run_main_eval()
    f20 = run_f020_eval()
    pr = run_probe()
    fl = run_failures()
    make_comparison_figure(ev, f20, pr, fl)
    make_heatmap_figure()
    write_summary_md(ev, f20, pr, fl)
    print("ALL B2_ATTN EVALUATIONS & FIGURES COMPLETE.")


if __name__ == "__main__":
    main()
