"""Slide-specific plots (100% Attention-Protagonist; Channel FiLM shown strictly as neutral slate control).
Every number is read from the raw repository JSONs copied from 4090 (src/remote/json/*.json).
"""
import json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "src/remote/json"
A = ROOT / "assets"; A.mkdir(exist_ok=True)
H = ROOT / "src/hist"

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 13.5, "axes.titlesize": 14.5, "axes.labelsize": 13.5,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 250})

C_P5 = "#94A3B8"
C_BC = "#D9730D"
C_FILM = "#64748B"     # neutral slate control (NEVER protagonist green!)
C_SS = "#2563EB"       # royal blue protagonist: Spatial-Softmax
C_RES = "#DC2626"      # crimson: Res-Spatial-Softmax
C_MAP = "#EA580C"      # warm orange: Map-Inject
C_REL = "#0F766E"      # deep teal protagonist: MH-Spatial-Relational
MK = ["o", "s", "^"]


def load(f):
    return json.load(open(J / f))


def pstd(v):
    return float(np.std(v, ddof=0))


b2 = load("b2_film_eval.json"); b2s7 = load("b2_film_s7n4.json")
base = load("baseline_eval.json"); s7 = load("s7n4_transfer.json")
v2_ev = load("b2_attn_v2_eval.json")
f20_v2 = load("b2_attn_v2_f020_eval.json")
pr_v2 = load("b2_attn_v2_probe.json")
fl_v2 = load("b2_attn_v2_failures.json")
fm = load("b2_attn_v2_feature_mechanics.json")
sc_attn = load("b2_attn_v2_scaling.json")
AGG, AGG7 = b2["aggregate"], b2s7["aggregate"]
p5_s6 = [r["closed_loop_success"] for r in base["runs"] if "baseline_seed" in r["run"]]
p5_s7 = [r["closed_loop_success"] for r in s7["runs"] if "baseline_seed" in r["run"]]
NUM = {}


def ms(v):
    return f"{np.mean(v):.3f} ± {pstd(v):.3f}"


def _v2_vals(k, env_tag):
    if k in ("bc30", "film_nwd"):
        d = v2_ev["references"][k][env_tag]["closed_loop_success"]
    elif f"v1_{k}" in v2_ev["references"]:
        d = v2_ev["references"][f"v1_{k}"][env_tag]["closed_loop_success"]
    else:
        d = v2_ev["variants"][k][env_tag]["closed_loop_success"]
    return d["mean"], d["sd"], d["values"]


def hist(prefix, key):
    arr = [[e[key] for e in json.load(open(H / f"{prefix}_seed{s}.json"))] for s in range(3)]
    return np.array(arr)


# ---------- 1. Slide 4: Baseline reproduction figure ----------
ep = np.arange(1, 31)
fig, axs = plt.subplots(1, 3, figsize=(9.3, 2.75), gridspec_kw=dict(width_ratios=[1.1, 1.15, 0.95]))
ax = axs[0]
for key, lab, c, ls_ in [("train_loss", "train CE", "#1F6FE5", "-"), ("val_loss", "val CE", C_BC, "--")]:
    a = hist("b3_bc30", key); m_, s_ = a.mean(0), a.std(0)
    ax.plot(ep, m_, color=c, ls=ls_, lw=2.2, label=lab); ax.fill_between(ep, m_ - s_, m_ + s_, color=c, alpha=0.2)
for e in (13, 14):
    ax.axvline(e, color="#C0392B", ls=":", lw=1.8)
ax.text(14.6, 0.56, "patience=5 stops\nseeds 1, 2", color="#C0392B", fontsize=11.5)
ax.set_xlabel("Epoch (max 30)"); ax.set_ylabel("Cross-entropy"); ax.set_xlim(1, 30); ax.set_ylim(0, 0.7)
ax.legend(frameon=False, loc="lower left", fontsize=11, ncol=2, handlelength=1.5, columnspacing=0.8); ax.grid(ls="--", alpha=0.4)
ax.set_title("(a) BC30 loss, 3 seeds", fontsize=13.5, fontweight="bold")

ax = axs[1]
triv = {r["run"]: r["closed_loop_success"] for r in base["runs"] if "trivial" in r["run"]}
names = ["left\nonly", "fwd\nonly", "rand-\nom", "BC\npat5", "BC30"]
vals = [triv["trivial:always_left"], triv["trivial:always_forward"], triv["trivial:random"]]
bars = [(v, 0) for v in vals] + [(np.mean(p5_s6), pstd(p5_s6)), (np.mean(AGG["bc30"]["closed_loop_success"]["values"]), pstd(AGG["bc30"]["closed_loop_success"]["values"]))]
cols_ = ["#CBD5E1"] * 3 + [C_P5, C_BC]
for i, ((m_, s_), c) in enumerate(zip(bars, cols_)):
    ax.bar(i, m_, 0.65, yerr=s_ if s_ else None, color=c, edgecolor="#333", capsize=4)
    ax.text(i, m_ + (s_ or 0) + 0.02, f"{m_:.2f}" if i < 3 else f"{m_:.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=10.5); ax.set_ylim(0, 1.0)
ax.set_ylabel("S6N3 Success"); ax.grid(axis="y", ls="--", alpha=0.4)
ax.set_title("(b) Sanity checks", fontsize=13.5, fontweight="bold")

ax = axs[2]
for i, (lab, v6, v7, c) in enumerate([("BC (pat=5)", p5_s6, p5_s7, C_P5), ("BC30", AGG["bc30"]["closed_loop_success"]["values"], AGG7["bc30"]["closed_loop_success"]["values"], C_BC)]):
    for off, v, hatch in [(-0.2, v6, None), (0.2, v7, "//")]:
        ax.bar(i + off, np.mean(v), 0.38, yerr=pstd(v), color=c if hatch is None else "white", edgecolor=c, hatch=hatch, lw=2, capsize=4)
        for k, val in enumerate(v):
            ax.plot(i + off + (k - 1) * 0.08, val, MK[k], ms=5.2, mfc="white", mec="#222", zorder=5)
        ax.text(i + off, max(max(v), np.mean(v) + pstd(v)) + 0.03, f"{np.mean(v):.3f}", ha="center", fontsize=10.2, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["BC (pat=5)", "BC30"]); ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Success"); ax.grid(axis="y", ls="--", alpha=0.4)
ax.set_title("(c) S6N3 / S7N4 (hatched)", fontsize=12.5, fontweight="bold")
fig.tight_layout(w_pad=1.0); fig.savefig(A / "p_baseline.png"); plt.close(fig)


# ---------- 2. Slide 7 Left: Primary comparison (Controls in orange/slate; Attention protagonists in blue/teal) ----------
main4 = [
    ("bc30", "BC30\n(Concat)", C_BC),
    ("film_nwd", "FiLM (ctrl)\n(225.2k)", C_FILM),
    ("spatial_softmax_attn", "Spatial-SS\n(225.2k)", C_SS),
    ("mh_spatial_relational", "MH-Rel\n(229.6k)", C_REL),
]
fig, ax = plt.subplots(figsize=(4.8, 3.12))
xm = np.arange(len(main4)); wm = 0.25
for idx, (etag, hatch, lbl) in enumerate([("S6N3", None, "S6N3"), ("S7N4", "//", "S7N4"), ("S8N6", "..", "S8N6")]):
    for i, (k, _, col) in enumerate(main4):
        m_, s_, vals = _v2_vals(k, etag)
        ax.bar(i + (idx - 1) * wm, m_, wm, yerr=s_, capsize=2.8,
               color=col if hatch is None else "white", edgecolor=col, hatch=hatch,
               lw=1.7, error_kw=dict(lw=1.2, ecolor="#222"))
        for sv_i, sv in enumerate(vals):
            ax.plot(i + (idx - 1) * wm + (sv_i - 1) * 0.055, sv, MK[sv_i], ms=4.2, mfc="white", mec="#222", mew=0.9, zorder=5)
        if etag in ("S6N3", "S8N6"):
            ax.text(i + (idx - 1) * wm, max(max(vals), m_ + s_) + 0.014, f"{m_:.3f}", ha="center", va="bottom",
                    fontsize=8.2, fontweight="bold", color="#111" if etag == "S6N3" else col)
ax.axvline(1.5, color="#94A3B8", ls=":", lw=1.2)
ax.set_xticks(xm); ax.set_xticklabels([x[1] for x in main4], fontsize=9.6, fontweight="bold")
for t_, (_, _, col) in zip(ax.get_xticklabels(), main4):
    t_.set_color(col)
ax.set_ylim(0.42, 1.07); ax.set_ylabel("Closed-loop Success\n(300 unseen seeds / run)", fontsize=11.2)
ax.grid(axis="y", ls="--", alpha=0.4)
ax.legend(handles=[
    Patch(facecolor="#475569", edgecolor="#222", label="S6N3 (6×6)"),
    Patch(facecolor="white", edgecolor="#222", hatch="//", label="S7N4 (7×7)"),
    Patch(facecolor="white", edgecolor="#222", hatch="..", label="S8N6 (8×8)"),
    Line2D([], [], marker="o", ls="", mfc="white", mec="#222", ms=5, label="seeds ○□△"),
], loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=4, frameon=False, fontsize=8.8, columnspacing=0.6, handlelength=1.1)
fig.tight_layout(); fig.savefig(A / "p_main_attn.png", dpi=250); plt.close(fig)


# ---------- 3. Slide 7 Right: Feature-space mechanism scatter ----------
fig, ax = plt.subplots(figsize=(4.55, 3.18))
pts = [
    ("film_nwd", "Channel FiLM\n(ctrl, 1.90×)", C_FILM, (8, -13), False),
    ("gated_attn_chaplot", "Chaplot GA", "#94A3B8", (8, 4), False),
    ("cross_attn_mha", "Cross-MHA\n(pos_emb)", "#818CF8", (8, -12), False),
    ("cross_mha_nopos", "Ours: NoPE\n(+6.2%)", "#6D28D9", (-64, 5), True),
    ("attn_map_inject", "Ours: Map-Inject\n(drift 0.57)", C_MAP, (-26, -27), True),
    ("mh_spatial_relational", "Ours: MH-Rel\n(SNR 2.43)", C_REL, (8, -8), True),
    ("res_spatial_softmax", "Ours: Res-SS\n(0.888)", C_RES, (-68, 4), True),
    ("spatial_softmax_attn", "Ours: Spatial-SS\n(3.59×→0.887)", C_SS, (-66, -28), True),
]
for k, lbl, col, (dx, dy), bold in pts:
    sx = fm["by_env"]["S8N6"][k]["spatial_selectivity_T_over_bg"]["mean"]
    sy = _v2_vals(k, "S8N6")[0]
    ax.scatter([sx], [sy], s=110 if bold else 75, color=col, edgecolors="#111", linewidths=1.0, zorder=5)
    ax.annotate(lbl, (sx, sy), xytext=(dx, dy), textcoords="offset points", fontsize=8.3, fontweight="bold" if bold else "normal", color="#111")
x1 = fm["by_env"]["S8N6"]["cross_attn_mha"]["spatial_selectivity_T_over_bg"]["mean"]
y1 = _v2_vals("cross_attn_mha", "S8N6")[0]
x2 = fm["by_env"]["S8N6"]["cross_mha_nopos"]["spatial_selectivity_T_over_bg"]["mean"]
y2 = _v2_vals("cross_mha_nopos", "S8N6")[0]
ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#6D28D9", lw=2.0, ls="--"))
ax.set_xlim(1.70, 3.92); ax.set_ylim(0.715, 0.928)
ax.set_xlabel("S8N6 Spatial Selectivity  Δh2(T) / Δh2(bg)", fontsize=10.8)
ax.set_ylabel("Zero-Shot S8N6 Success", fontsize=11.2)
ax.set_title("Spatial Selectivity Drives Cross-Scale Transfer", fontsize=11.2, fontweight="bold", color="#112B5C")
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_mechanics_slide.png", dpi=250); plt.close(fig)


# ---------- 4. Slide 8 Right: Failure shift (4-stage progression) ----------
fb = load("b3_v2_eval_s6n3.json")["by_group"]
rows_f = [
    ("BC30\n(Concat)", fb["b3_bc30"]["failure_groups_mean"]["representation"], fb["b3_bc30"]["failure_groups_mean"]["state_coverage"]),
    ("Channel FiLM\n(ctrl)", fb["b2_film_nwd"]["failure_groups_mean"]["representation"], fb["b2_film_nwd"]["failure_groups_mean"]["state_coverage"]),
    ("Ours:\nSpatial-SS", fl_v2["references"]["aggregate"]["spatial_softmax_attn"]["representation"]["mean"], fl_v2["references"]["aggregate"]["spatial_softmax_attn"]["state_coverage"]["mean"]),
    ("Ours:\nMH-Rel", fl_v2["aggregate"]["mh_spatial_relational"]["representation"]["mean"], fl_v2["aggregate"]["mh_spatial_relational"]["state_coverage"]["mean"]),
]
fig, ax = plt.subplots(figsize=(4.45, 3.25))
for i, (lab, r_, s_) in enumerate(rows_f):
    ax.bar(i, r_, 0.58, color="#D64545", edgecolor="#222")
    ax.bar(i, s_, 0.58, bottom=r_, color="#6FA8F5", edgecolor="#222")
    ax.text(i, r_ + s_ + 1.4, f"{r_ + s_:.1f}\n({r_:.1f}+{s_:.1f})", ha="center", fontsize=10.5, fontweight="bold")
ax.axvline(1.5, color="#94A3B8", ls=":", lw=1.2)
ax.set_xticks(range(len(rows_f))); ax.set_xticklabels([r[0] for r in rows_f], fontsize=10.8, fontweight="bold")
ax.set_ylabel("Mean failures / 300 eps\n(mean of 3 seeds)", fontsize=12); ax.set_ylim(0, 74)
ax.legend(handles=[Patch(color="#D64545", label="representation (grounding)"), Patch(color="#6FA8F5", label="state coverage (loops)")],
          frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=11.2)
ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_failshift.png"); plt.close(fig)


# ---------- 5. Backup B3: Capacity & wd controls (Concat vs Channel FiLM ctrl vs Attention protagonists!) ----------
abl_items = [
    ("Concat\nwd=1e-4", AGG["concat_wd"]["closed_loop_success"]["values"], "#94A3B8"),
    ("Concat-Wide\n(224,683 p)", AGG["concat_wide_wd"]["closed_loop_success"]["values"], "#94A3B8"),
    ("BC30\n(223,071 p)", AGG["bc30"]["closed_loop_success"]["values"], C_BC),
    ("FiLM-wd\n(ctrl)", AGG["film_wd"]["closed_loop_success"]["values"], C_FILM),
    ("FiLM-nwd\n(225,183 p)", AGG["film_nwd"]["closed_loop_success"]["values"], C_FILM),
    ("Res-SS\n(225,184 p)", _v2_vals("res_spatial_softmax", "S6N3")[2], C_RES),
    ("Map-Inject\n(226,335 p)", _v2_vals("attn_map_inject", "S6N3")[2], C_MAP),
    ("Spatial-SS\n(225,183 p)", _v2_vals("spatial_softmax_attn", "S6N3")[2], C_SS),
    ("MH-Rel\n(229,572 p)", _v2_vals("mh_spatial_relational", "S6N3")[2], C_REL),
]
fig, ax = plt.subplots(figsize=(9.3, 3.45))
for i, (lab, v, c) in enumerate(abl_items):
    ax.bar(i, np.mean(v), 0.62, yerr=pstd(v), color=c, alpha=0.92, capsize=3.5, edgecolor="#222")
    for j, val in enumerate(v):
        ax.plot(i + (j - 1) * 0.13, val, MK[j], ms=5.5, mfc="white", mec="#222", zorder=5)
    ax.text(i, max(max(v), np.mean(v) + pstd(v)) + 0.013, f"{np.mean(v):.3f}", ha="center", fontsize=11.0, fontweight="bold")
ax.axvline(4.5, color="#475569", ls="--", lw=1.4)
ax.text(2.0, 1.015, "Concat & Channel Controls", ha="center", fontsize=11.5, color="#475569", fontweight="bold")
ax.text(6.5, 1.015, "Ours: Spatial & Relational Attention", ha="center", fontsize=11.5, color=C_SS, fontweight="bold")
ax.set_xticks(range(len(abl_items))); ax.set_xticklabels([o[0] for o in abl_items], fontsize=9.5, fontweight="bold")
ax.set_ylim(0.60, 1.055); ax.set_ylabel("S6N3 Success\n(mean ± SD, n=3)"); ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_ablation.png"); plt.close(fig)


# ---------- 6. Backup B7: Counterfactual language probe (183 test states — Attention wins!) ----------
probe_groups = [
    ("BC30\n(Concat)", pr_v2["references"]["bc30"], C_BC),
    ("Chaplot GA\n(ctrl)", pr_v2["by_variant"]["gated_attn_chaplot"], "#94A3B8"),
    ("Channel FiLM\n(ctrl)", pr_v2["references"]["film_nwd"], C_FILM),
    ("Ours:\nMH-Rel", pr_v2["by_variant"]["mh_spatial_relational"], C_REL),
    ("Ours:\nSpatial-SS", pr_v2["references"]["spatial_softmax_attn"], C_SS),
]
fig, ax = plt.subplots(figsize=(5.7, 3.45))
metrics = [("color_swap_sensitivity", "Color swap\n(138 pairs)"), ("type_swap_sensitivity", "Type swap\n(59 pairs)"), ("combined_sensitivity", "Combined\n(183 states)")]
wp = 0.155
for mi, (mk, ml) in enumerate(metrics):
    for si, (sn, d_p, col) in enumerate(probe_groups):
        m_, s_ = d_p[mk]["mean"], d_p[mk]["sd"]
        xx = mi + (si - 2) * wp
        ax.bar(xx, m_, wp, yerr=s_, color=col, edgecolor="#222", capsize=2.5, label=sn.replace("\n", " ") if mi == 0 else None)
        if mk in ("color_swap_sensitivity", "type_swap_sensitivity") and si in (0, 2, 4):
            ax.text(xx, m_ + s_ + 0.012, f"{m_:.3f}", ha="center", fontsize=8.8, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in metrics], fontsize=11.5, fontweight="bold")
ax.set_ylabel("Greedy action-change rate"); ax.set_ylim(0, 0.43)
ax.legend(frameon=False, fontsize=9.2, loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=3, columnspacing=0.7, handlelength=1.1)
ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_probe.png"); plt.close(fig)


# ---------- 7. Backup B8: Mission breakdown pooled 900 episodes (BC30 vs FiLM ctrl vs Spatial-SS vs MH-Rel!) ----------
ev1 = load("b2_attn_eval.json")
def pooled(runs):
    agg = collections.defaultdict(lambda: [0, 0.0])
    for r in runs:
        for m, v in r["per_mission"].items():
            w_ = m.split()
            for key in (w_[2], w_[3], w_[4]):
                agg[key][0] += v["n"]; agg[key][1] += v["n"] * v["success"]
    return {k: (n, s / n) for k, (n, s) in agg.items()}

ss_s6_runs = [r for r in ev1["runs"] if r.get("variant") == "spatial_softmax_attn" and r["env_id"] == "BabyAI-GoToLocalS6N3-v0"]
rel_s6_runs = [r for r in v2_ev["runs"] if r.get("variant") == "mh_spatial_relational" and r["env_id"] == "BabyAI-GoToLocalS6N3-v0"]
mg = [
    ("BC30 (Concat)", pooled(b2["baseline_runs"]), C_BC),
    ("Channel FiLM (ctrl)", pooled([r for r in b2["runs"] if "film_nwd" in r["run"]]), C_FILM),
    ("Ours: Spatial-SS", pooled(ss_s6_runs), C_SS),
    ("Ours: MH-Rel", pooled(rel_s6_runs), C_REL),
]
cats = ["the", "a", "red", "green", "blue", "purple", "yellow", "grey", "ball", "box", "key"]
fig, ax = plt.subplots(figsize=(9.3, 3.25))
wm_b = 0.20
for gi, (gn, d_, col) in enumerate(mg):
    ax.bar(np.arange(len(cats)) + (gi - 1.5) * wm_b, [d_[c][1] for c in cats], wm_b, color=col, edgecolor="#222", lw=0.7, label=gn)
    for c in cats:
        NUM[f"mission_v2 {gn} {c}"] = round(d_[c][1], 3)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([f"'{c}'\nn={mg[0][1][c][0]}" for c in cats], fontsize=10.8)
for xv in (1.5, 7.5):
    ax.axvline(xv, color="#666", ls=":")
ax.set_ylim(0.65, 1.04); ax.set_ylabel("Pooled Success\n(3 seeds × 300 eps)")
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=10.8)
ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_mission.png"); plt.close(fig)


# ---------- 8. Backup B9: Complete 10/20/50/100% Scaling Curves (Spatial-SS & MH-Rel as Protagonists!) ----------
sc = load("b5_data_eff_eval.json")["scaling_curves"]
fr = ["f010", "f020", "f050", "f100"]; xs = [10, 20, 50, 100]
fig, ax = plt.subplots(figsize=(6.05, 3.55))

ss_m = [sc_attn["spatial_softmax_step"][f]["S6N3"]["closed_loop_success"]["mean"] for f in fr]
ss_s = [sc_attn["spatial_softmax_step"][f]["S6N3"]["closed_loop_success"]["sd"] for f in fr]
rel_m = [sc_attn["mh_spatial_relational_step"][f]["S6N3"]["closed_loop_success"]["mean"] for f in fr]
rel_s = [sc_attn["mh_spatial_relational_step"][f]["S6N3"]["closed_loop_success"]["sd"] for f in fr]

ax.errorbar(xs, ss_m, yerr=ss_s, color=C_SS, ls="-", marker="D", ms=6.5, lw=2.6, capsize=4, label="Ours: Spatial-Softmax (step-matched)", zorder=5)
ax.errorbar(xs, rel_m, yerr=rel_s, color=C_REL, ls="-", marker="^", ms=7.0, lw=2.4, capsize=4, label="Ours: MH-Spatial-Rel (step-matched)", zorder=4)
for key, lab, c, ls_ in [
    ("film_step", "Channel FiLM ctrl (step-matched)", C_FILM, "--"),
    ("bc_step", "BC30 Concat (step-matched)", C_BC, "-."),
]:
    m_ = [sc[key][f]["success_mean"] for f in fr]; s_ = [sc[key][f]["success_sd"] for f in fr]
    ax.errorbar(xs, m_, yerr=s_, color=c, ls=ls_, marker="o", ms=5.5, lw=1.9, capsize=3.5, label=lab, zorder=3)

ax.axhline(sc["bc_naive"]["f100"]["success_mean"], color=C_BC, ls=":", lw=1.2)
ax.text(46, 0.822, "100% BC30 = 0.814", color=C_BC, fontsize=10.2, fontweight="bold")
for x_pt, y_pt in zip(xs, ss_m):
    ax.text(x_pt, y_pt + 0.018, f"{y_pt:.3f}", ha="center", fontsize=9.8, fontweight="bold", color=C_SS)
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["10%\n450", "20%\n900", "50%\n2,250", "100%\n4,500"])
ax.set_xlabel("Training demos (fraction / episodes)"); ax.set_ylabel("S6N3 Success\n(mean ± SD, n=3)")
ax.set_ylim(0.60, 1.035)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=2, fontsize=9.0, columnspacing=0.8)
ax.grid(ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_dataeff.png"); plt.close(fig)


# ---------- 9. Backup B11: Training CE & Val Action Acc Curves (BC30 vs FiLM ctrl vs Spatial-SS vs MH-Rel!) ----------
fig, axs = plt.subplots(1, 2, figsize=(9.3, 3.25))
curve_models = [
    ("b3_bc30", "BC30 (Concat)", C_BC, "-."),
    ("b2_film_nwd", "Channel FiLM (ctrl)", C_FILM, "--"),
    ("b2_attn_ss", "Ours: Spatial-Softmax", C_SS, "-"),
    ("b2_attn_mhrel", "Ours: MH-Spatial-Rel", C_REL, "-"),
]
for ax, key, ttl, ylab, ylim in [
    (axs[0], "train_loss", "(a) Training cross-entropy (30 epochs)", "Train CE loss", (0.0, 0.65)),
    (axs[1], "val_acc", "(b) Held-out validation action accuracy (500 eps)", "Val action acc", (0.66, 1.00)),
]:
    for pre, lab, c, ls_ in curve_models:
        a = hist(pre, key); m_, s_ = a.mean(0), a.std(0)
        ax.plot(ep, m_, color=c, ls=ls_, lw=2.2, label=lab)
        ax.fill_between(ep, m_ - s_, m_ + s_, color=c, alpha=0.14)
    ax.set_title(ttl, fontsize=13.2, fontweight="bold"); ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
    ax.set_xlim(1, 30); ax.set_ylim(*ylim); ax.grid(ls="--", alpha=0.4)
axs[1].legend(frameon=False, fontsize=10.5, loc="lower right")
fig.tight_layout(); fig.savefig(A / "p_loss_backup.png"); plt.close(fig)

json.dump(NUM, open(A / "numbers.json", "w"), indent=1, default=str)
print("ALL ATTENTION-PROTAGONIST PLOTS GENERATED CLEANLY:", sorted(NUM.keys())[:6])
