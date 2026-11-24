"""Build Paradigm 1 Markdown Summary (`reports/b6_georel_summary.md`) & Figure from `reports/b6_georel_results.json`."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

LADDER_MODELS = [
    ("bc30", "0. BC30 (Concat)"),
    ("film_nwd", "0'. Channel FiLM"),
    ("spatial_softmax_attn", "1. +A: 1H Spatial-SS"),
    ("s2_map1h", "2. +A+B: 1H Map-Inject"),
    ("s3_map1h_relbias", "3. +A+B+C: Map1H+RelBias"),
    ("s3_res1h_relbias", "3'. +A+B'+C: Res1H+RelBias"),
    ("mh_spatial_relational", "ref: v2 4H MH-Rel"),
    ("bc30_gru", "ctrl: BC30 + GRU"),
    ("s4_georel_gru", "4. +A+B+C+D: GeoRel-GRU"),
    ("s4_res1h_gru", "4'. +A+B'+C+D: Res1H-GRU"),
]

NEW_VARIANTS = (
    "s2_map1h",
    "s3_map1h_relbias",
    "s3_res1h_relbias",
    "bc30_gru",
    "s4_georel_gru",
    "s4_res1h_gru",
)


def main() -> None:
    res = json.loads((ROOT / "reports" / "b6_georel_results.json").read_text())
    prev_ev2 = json.loads((ROOT / "reports" / "b2_attn_v2_eval.json").read_text())
    prev_f020 = json.loads((ROOT / "reports" / "b2_attn_v2_f020_eval.json").read_text())
    prev_fail = json.loads((ROOT / "reports" / "b2_attn_v2_failures.json").read_text())
    prev_probe = json.loads((ROOT / "reports" / "b2_attn_v2_probe.json").read_text())
    prev_feat = json.loads((ROOT / "reports" / "b2_attn_v2_feature_mechanics.json").read_text())

    eval_summary = dict(res["eval"])
    if "spatial_softmax_attn" not in eval_summary and "v1_spatial_softmax_attn" in prev_ev2.get("references", {}):
        eval_summary["spatial_softmax_attn"] = prev_ev2["references"]["v1_spatial_softmax_attn"]

    f020_summary = dict(res["f020"])
    if "spatial_softmax_attn" not in f020_summary:
        raw_ss = prev_f020.get("references", {}).get("v1_spatial_softmax_attn") or prev_f020.get("variants", {}).get("v1_spatial_softmax_attn")
        if raw_ss:
            f020_summary["spatial_softmax_attn"] = raw_ss

    fail_summary = dict(res["failures"])
    ref_fail_agg = prev_fail.get("references", {}).get("aggregate", {})
    var_fail_agg = prev_fail.get("aggregate", {})
    for rk, src_key in [
        ("spatial_softmax_attn", "v1_spatial_softmax_attn"),
        ("mh_spatial_relational", "mh_spatial_relational"),
    ]:
        if rk not in fail_summary:
            pf = ref_fail_agg.get(src_key) or var_fail_agg.get(src_key)
            if pf:
                fail_summary[rk] = {
                    "total_failures": pf["n_failure"],
                    "repr_failures": pf["representation"],
                    "cov_failures": pf["state_coverage"],
                    "cases": {},
                }

    # Fill verified reference case pass counts (from reports/b2_attn_v2_failures.json / README)
    ref_case_lookup = {
        "bc30": {"4501": 1, "4502": 0, "4568": 0},
        "film_nwd": {"4501": 2, "4502": 0, "4568": 0},
        "spatial_softmax_attn": {"4501": 1, "4502": 3, "4568": 0},
        "mh_spatial_relational": {"4501": 3, "4502": 3, "4568": 0},
    }
    for rk, cmap in ref_case_lookup.items():
        if rk in fail_summary and not fail_summary[rk].get("cases"):
            fail_summary[rk]["cases"] = {k: {"solved": v, "total": 3} for k, v in cmap.items()}

    # Build figure
    keys = [k for k, _ in LADDER_MODELS if k in eval_summary]
    labels = [lbl for k, lbl in LADDER_MODELS if k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), dpi=220)
    x = np.arange(len(keys))

    # Panel A
    ax = axes[0]
    w = 0.26
    for idx, (env_k, col, hatch, name) in enumerate([
        ("S6N3", "#2563EB", "", "S6N3 (6x6, ID)"),
        ("S7N4", "#0F766E", "//", "S7N4 (7x7, OOD)"),
        ("S8N6", "#D97706", "..", "S8N6 (8x8, OOD)"),
    ]):
        means = [eval_summary[k][env_k]["closed_loop_success"]["mean"] for k in keys]
        sds = [eval_summary[k][env_k]["closed_loop_success"]["sd"] for k in keys]
        ax.bar(
            x + (idx - 1) * w, means, width=w, yerr=sds, capsize=2.2,
            color=col, alpha=0.90 if idx == 0 else 0.80, edgecolor="#0F172A",
            linewidth=0.6, hatch=hatch, label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8.2)
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("Closed-Loop Success (300 unseen eps)")
    ax.set_title("(a) Paradigm 1 Composable Ladder: S6N3 / S7N4 / S8N6", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right")

    # Panel B
    ax = axes[1]
    fkeys = [k for k in keys if k in f020_summary]
    flabels = [lbl for k, lbl in LADDER_MODELS if k in fkeys]
    fx = np.arange(len(fkeys))
    w2 = 0.34
    s6m = [f020_summary[k]["S6N3"]["closed_loop_success"]["mean"] for k in fkeys]
    s6s = [f020_summary[k]["S6N3"]["closed_loop_success"]["sd"] for k in fkeys]
    s7m = [f020_summary[k]["S7N4"]["closed_loop_success"]["mean"] for k in fkeys]
    s7s = [f020_summary[k]["S7N4"]["closed_loop_success"]["sd"] for k in fkeys]
    ax.bar(fx - w2 / 2, s6m, width=w2, yerr=s6s, capsize=2.5, color="#1D4ED8", edgecolor="#0F172A", linewidth=0.6, label="20% S6N3")
    ax.bar(fx + w2 / 2, s7m, width=w2, yerr=s7s, capsize=2.5, color="#0D9488", edgecolor="#0F172A", linewidth=0.6, hatch="//", label="20% S7N4 (OOD)")
    ax.set_xticks(fx)
    ax.set_xticklabels(flabels, rotation=32, ha="right", fontsize=8.2)
    ax.set_ylim(0.55, 0.98)
    ax.set_ylabel("Closed-Loop Success (20% = 900 eps)")
    ax.set_title("(b) Step-Matched 20% Low-Data Generalization", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.2, loc="lower right")

    # Panel C
    ax = axes[2]
    tkeys = [k for k in keys if k in fail_summary]
    tlabels = [lbl for k, lbl in LADDER_MODELS if k in tkeys]
    tx = np.arange(len(tkeys))
    repr_m = [fail_summary[k]["repr_failures"]["mean"] for k in tkeys]
    cov_m = [fail_summary[k]["cov_failures"]["mean"] for k in tkeys]
    ax.bar(tx, repr_m, width=0.58, color="#DC2626", alpha=0.88, edgecolor="#0F172A", linewidth=0.6, label="Repr. Grounding")
    ax.bar(tx, cov_m, bottom=repr_m, width=0.58, color="#3B82F6", alpha=0.88, edgecolor="#0F172A", linewidth=0.6, label="POMDP Coverage Loops")
    for i, (r, c) in enumerate(zip(repr_m, cov_m)):
        ax.text(i, r + c + 0.8, f"{r+c:.1f}\n({r:.1f}+{c:.1f})", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax.set_xticks(tx)
    ax.set_xticklabels(tlabels, rotation=32, ha="right", fontsize=8.2)
    ax.set_ylim(0, max(r + c for r, c in zip(repr_m, cov_m)) * 1.18 + 3)
    ax.set_ylabel("Mean Failures / 300 unseen S6N3 eps")
    ax.set_title("(c) 300-Ep Failure Funnel: Grounding + Blind-Spot Loops", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.2, loc="upper right")

    fig.tight_layout()
    fig_path = ROOT / "reports" / "figures" / "b6_georel_unified" / "b6_paradigm1_ladder.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # Markdown report
    lines = [
        "# Branch 6 · Paradigm 1 Unified Composable Architecture (`GeoRel-BCNet` & `GeoRel-GRU`)",
        "",
        "All models trained strictly on frozen `data/expert_5000.pkl` (4,500 train + 500 val, **zero extra expert queries**),",
        "`30` epochs, `Adam lr=1e-3, wd=0`, checkpoint strictly by offline `val_action_acc`, evaluated on `300` unseen seeds (`mean ± SD`, `ddof=0`, `n = 3`).",
        "",
        "## 1. Paradigm 1 Composable Ablation Ladder (Multi-Scale, 20% Low-Data, Failures & Key Cases)",
        "",
        "| Stage / Model | Params | `val_acc` | `S6N3` (ID) | `S7N4` (OOD) | `S8N6` (OOD) | `20%` S6 / S7 | Fails /300 (`Repr+Cov`) | `4501` / `4502` / `4568` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, lbl in LADDER_MODELS:
        if k not in eval_summary:
            continue
        ev = eval_summary[k]
        p = ev["S6N3"]["n_params"]
        va = ev["S6N3"]["val_action_acc"]["mean"]
        s6 = ev["S6N3"]["closed_loop_success"]
        s7 = ev["S7N4"]["closed_loop_success"]
        s8 = ev["S8N6"]["closed_loop_success"]
        if k in f020_summary:
            f6 = f020_summary[k]["S6N3"]["closed_loop_success"]["mean"]
            f7 = f020_summary[k]["S7N4"]["closed_loop_success"]["mean"]
            fstr = f"{f6:.3f} / {f7:.3f}"
        else:
            fstr = "—"
        fs = fail_summary[k]
        f_tot = fs["total_failures"]["mean"]
        f_r = fs["repr_failures"]["mean"]
        f_c = fs["cov_failures"]["mean"]
        fail_str = f"{f_tot:.1f} (`{f_r:.1f}+{f_c:.1f}`)"
        cs = fs.get("cases", {})
        c4501 = cs.get("4501", {}).get("solved", "—")
        c4502 = cs.get("4502", {}).get("solved", "—")
        c4568 = cs.get("4568", {}).get("solved", "—")
        case_str = f"{c4501}/3 · {c4502}/3 · {c4568}/3"
        lines.append(
            f"| {lbl} | {p:,} | {va:.3f} | {s6['mean']:.3f} ± {s6['sd']:.3f} | {s7['mean']:.3f} ± {s7['sd']:.3f} | "
            f"{s8['mean']:.3f} ± {s8['sd']:.3f} | {fstr} | {fail_str} | {case_str} |"
        )

    p1 = res["probe1_counterfactual"]
    p2 = res["probe2_feature_geometry"]
    lines += [
        "",
        "## 2. Mechanistic Probes (Probe 1 Counterfactual Swap & Probe 2 Feature Geometry)",
        "",
        "| Model | Color Swap | Type Swap | S6 `h_mod T/maxD` | S8 `h_mod T/maxD` | S8 `h2 Partial SNR` | S8 Selectivity | S8 Bg Drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for var in NEW_VARIANTS:
        if var not in p1:
            continue
        r1 = p1[var]
        r2s6 = p2["S6N3"][var]
        r2s8 = p2["S8N6"][var]
        lines.append(
            f"| `{var}` | {r1['color_swap_rate']['mean']:.3f} | {r1['type_swap_rate']['mean']:.3f} | "
            f"{r2s6['h_mod_T_over_maxD']['mean']:.3f}× | {r2s8['h_mod_T_over_maxD']['mean']:.3f}× | "
            f"{r2s8['h2_T_over_partialD']['mean']:.3f}× | {r2s8['spatial_selectivity']['mean']:.2f}× | "
            f"{r2s8['background_drift']['mean']:.3f} |"
        )

    out_md = ROOT / "reports" / "b6_georel_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
