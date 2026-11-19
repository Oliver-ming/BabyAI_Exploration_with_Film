"""Slide-specific plots. Every number is read from the raw repository JSONs copied from 4090
(src/remote/json/*.json); nothing is typed by hand except labels."""
import json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "src/remote/json"
A = ROOT / "assets"; A.mkdir(exist_ok=True)
FIG = ROOT / "src/remote/figures"

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 250})
C_P5, C_BC, C_FILM, C_GREY = "#8C9BB0", "#D9730D", "#0E9664", "#5A6B80"
MK = ["o", "s", "^"]


def load(f):
    return json.load(open(J / f))


def pstd(v):
    return float(np.std(v, ddof=0))


b2 = load("b2_film_eval.json"); b2s7 = load("b2_film_s7n4.json")
base = load("baseline_eval.json"); s7 = load("s7n4_transfer.json")
AGG, AGG7 = b2["aggregate"], b2s7["aggregate"]
p5_s6 = [r["closed_loop_success"] for r in base["runs"] if "baseline_seed" in r["run"]]
p5_s7 = [r["closed_loop_success"] for r in s7["runs"] if "baseline_seed" in r["run"]]
NUM = {}  # every number used on slides, dumped to assets/numbers.json for audit


def ms(v):
    return f"{np.mean(v):.3f} ± {pstd(v):.3f}"


# ---------- 1. primary comparison (S6N3 + S7N4, all seeds) ----------
groups = [("BC\n(patience=5)", p5_s6, p5_s7, C_P5),
          ("BC30\n(controlled)", AGG["bc30"]["closed_loop_success"]["values"], AGG7["bc30"]["closed_loop_success"]["values"], C_BC),
          ("FiLM-nwd\n(ours)", AGG["film_nwd"]["closed_loop_success"]["values"], AGG7["film_nwd"]["closed_loop_success"]["values"], C_FILM)]
fig, ax = plt.subplots(figsize=(4.6, 3.5))
x = np.arange(3); w = 0.38
for i, (lab, v6, v7, c) in enumerate(groups):
    for off, v, hatch in [(-w / 2, v6, None), (w / 2, v7, "//")]:
        ax.bar(i + off, np.mean(v), w, yerr=pstd(v), color=c if hatch is None else "white", edgecolor=c, hatch=hatch,
               linewidth=2, capsize=4, error_kw=dict(lw=1.5, ecolor="#222"))
        for k, val in enumerate(v):
            ax.plot(i + off + (k - 1) * 0.09, val, MK[k], ms=5.5, mfc="white", mec="#222", mew=1.2, zorder=5)
    NUM[f"S6N3 {lab.replace(chr(10), ' ')}"] = ms(v6); NUM[f"S7N4 {lab.replace(chr(10), ' ')}"] = ms(v7)
ax.set_xticks(x)
ax.set_xticklabels([f"{nm}\n{np.mean(g[1]):.3f} | {np.mean(g[2]):.3f}" for nm, g in zip(["BC pat=5", "BC30", "FiLM-nwd"], groups)], fontsize=12.5)
for t_, g in zip(ax.get_xticklabels(), groups):
    t_.set_color(g[3] if g[3] != C_P5 else "#4A5568"); t_.set_fontweight("bold")
ax.tick_params(axis="y", labelsize=12.5)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("Closed-loop Success\n(300 unseen seeds / run)", fontsize=13)
ax.grid(axis="y", ls="--", alpha=0.4)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(facecolor="#999", edgecolor="#555", label="S6N3"),
                   Patch(facecolor="white", edgecolor="#555", hatch="//", label="S7N4 zero-shot"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec="#222", label="seeds ○ □ △"),
                   Line2D([], [], color="#222", lw=1.5, label="±1 SD (ddof=0)")],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=11.5, columnspacing=0.8, handlelength=1.4)
fig.tight_layout(); fig.savefig(A / "p_main.png"); plt.close(fig)

# paired deltas
pr = b2["paired"]["film_nwd"]; pr7 = b2s7["paired"]["film_nwd"]
NUM["paired S6N3 FiLM-nwd - BC30"] = (pr["delta_mean"], pr["delta_sd"], [r["delta"] for r in pr["rows"]], pr["n_positive"])
NUM["paired S7N4 FiLM-nwd - BC30"] = (pr7["delta_mean"], pr7["delta_sd"], [r["delta"] for r in pr7["rows"]], pr7["n_positive"])
for k in ["mean_return", "val_action_acc", "n_timeout", "median_len_success"]:
    for g in ["bc30", "film_nwd"]:
        NUM[f"{k} {g}"] = (AGG[g][k]["mean"], AGG[g][k]["sd"], AGG[g][k]["values"])
NUM["pat5 val_acc"] = ms([r["val_action_acc"] for r in base["runs"] if "baseline_seed" in r["run"]])
NUM["pat5 return"] = ms([r["mean_return"] for r in base["runs"] if "baseline_seed" in r["run"]])
NUM["pat5 best_epoch"] = [r["best_epoch"] for r in base["runs"] if "baseline_seed" in r["run"]]
NUM["trivial"] = {r["run"]: r["closed_loop_success"] for r in base["runs"] if "trivial" in r["run"]}

# ---------- 2. failure shift (representation vs state coverage) ----------
fb = load("b3_v2_eval_s6n3.json")["by_group"]
rows = [("BC\n(pat=5)", fb["baseline"], C_P5), ("BC30", fb["b3_bc30"], C_BC), ("FiLM-nwd", fb["b2_film_nwd"], C_FILM)]
fig, ax = plt.subplots(figsize=(4.3, 3.45))
for i, (lab, g, c) in enumerate(rows):
    r_, s_ = g["failure_groups_mean"]["representation"], g["failure_groups_mean"]["state_coverage"]
    ax.bar(i, r_, 0.6, color="#D64545", edgecolor="#222")
    ax.bar(i, s_, 0.6, bottom=r_, color="#6FA8F5", edgecolor="#222")
    ax.text(i, r_ + s_ + 1.5, f"{r_ + s_:.1f}\n({r_:.1f}+{s_:.1f})", ha="center", fontsize=12, fontweight="bold")
    NUM[f"failures {lab.replace(chr(10), ' ')}"] = (r_, s_, g["failure_types_mean"])
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel("Mean failures / 300 eps\n(mean of 3 seeds)", fontsize=14); ax.set_ylim(0, 92)
ax.legend(handles=[Patch(color="#D64545", label="representation (grounding)"), Patch(color="#6FA8F5", label="state coverage (loops)")],
          frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=1, fontsize=13)
ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_failshift.png"); plt.close(fig)

# ---------- 3. ablations / capacity control (backup) ----------
order = [("concat_wd", "Concat\nwd=1e-4"), ("concat_wide_wd", "Concat-Wide\n(224,683 p)"), ("bc30", "BC30\n(concat)"),
         ("concat_nwd", "Concat-nwd\n(re-impl.)"), ("film_res_wd", "FiLM-res\nwd"), ("film_wd", "FiLM\nwd=1e-4"),
         ("film_only_wd", "FiLM-only\n(221,087 p)"), ("film_nwd", "FiLM-nwd\n(primary)")]
fig, ax = plt.subplots(figsize=(9.2, 3.5))
for i, (k, lab) in enumerate(order):
    v = AGG[k]["closed_loop_success"]["values"]
    c = C_FILM if "film" in k else (C_BC if k == "bc30" else C_GREY)
    ax.bar(i, np.mean(v), 0.62, yerr=pstd(v), color=c, alpha=0.9, capsize=4, edgecolor="#222")
    for j, val in enumerate(v):
        ax.plot(i + (j - 1) * 0.14, val, MK[j], ms=6, mfc="white", mec="#222", zorder=5)
    ax.text(i, max(max(v), np.mean(v) + pstd(v)) + 0.015, f"{np.mean(v):.3f}", ha="center", fontsize=12.5, fontweight="bold")
    NUM[f"abl {k}"] = ms(v)
ax.set_xticks(range(len(order))); ax.set_xticklabels([o[1] for o in order], fontsize=12)
ax.set_ylim(0.6, 1.03); ax.set_ylabel("S6N3 Success\n(mean ± SD, n=3)"); ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_ablation.png"); plt.close(fig)

# ---------- 4. counterfactual language probe (backup) ----------
ls = load("b2_language_sensitivity.json")["runs"]
sets = {"BC30 (n=3)": [r for r in ls if "b3_bc30" in r["run"]],
        "All concat (n=6)": [r for r in ls if "bc30" in r["run"] or "concat_nwd" in r["run"]],
        "FiLM-nwd (n=3)": [r for r in ls if "film_nwd" in r["run"]]}
cols = [C_BC, "#C7D0DC", C_FILM]
fig, ax = plt.subplots(figsize=(5.4, 3.5))
metrics = [("color_swap_sensitivity", "Color swap\n(138 pairs)"), ("type_swap_sensitivity", "Type swap\n(59 pairs)"), ("combined_sensitivity", "Combined\n(183 states)")]
for mi, (mk, ml) in enumerate(metrics):
    for si, (sn, rs) in enumerate(sets.items()):
        v = [r[mk] for r in rs]
        xx = mi + (si - 1) * 0.26
        ax.bar(xx, np.mean(v), 0.26, yerr=pstd(v), color=cols[si], edgecolor="#333", capsize=3, label=sn if mi == 0 else None)
        if mk == "color_swap_sensitivity":
            ax.text(xx, np.mean(v) + pstd(v) + 0.01, f"{np.mean(v):.3f}", ha="center", fontsize=11.5, fontweight="bold")
        NUM[f"probe {mk} {sn}"] = ms(v)
ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in metrics])
ax.set_ylabel("Greedy action-change rate"); ax.set_ylim(0, 0.36)
ax.legend(frameon=False, fontsize=11.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handlelength=1.2)
ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_probe.png"); plt.close(fig)

# ---------- 5. mission breakdown pooled 900 episodes (backup) ----------
def pooled(runs):
    agg = collections.defaultdict(lambda: [0, 0.0])
    for r in runs:
        for m, v in r["per_mission"].items():
            w_ = m.split()
            for key in (w_[2], w_[3], w_[4]):
                agg[key][0] += v["n"]; agg[key][1] += v["n"] * v["success"]
    return {k: (n, s / n) for k, (n, s) in agg.items()}
mg = {"BC (pat=5)": pooled([r for r in base["runs"] if "baseline_seed" in r["run"]]),
      "BC30": pooled(b2["baseline_runs"]),
      "FiLM-nwd": pooled([r for r in b2["runs"] if "film_nwd" in r["run"]])}
cats = ["the", "a", "red", "green", "blue", "purple", "yellow", "grey", "ball", "box", "key"]
fig, ax = plt.subplots(figsize=(9.2, 3.4))
for gi, (gn, d_) in enumerate(mg.items()):
    ax.bar(np.arange(len(cats)) + (gi - 1) * 0.27, [d_[c][1] for c in cats], 0.27, color=[C_P5, C_BC, C_FILM][gi], edgecolor="#333", label=gn)
    NUM[f"mission {gn}"] = {c: (d_[c][0], round(d_[c][1], 3)) for c in cats}
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([f"'{c}'\nn={mg['BC30'][c][0]}" for c in cats], fontsize=11.5)
for xv in (1.5, 7.5):
    ax.axvline(xv, color="#666", ls=":")
ax.set_ylim(0.6, 1.04); ax.set_ylabel("Pooled Success\n(3 seeds × 300 eps)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=12)
ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_mission.png"); plt.close(fig)

# ---------- 6. data efficiency (backup) ----------
sc = load("b5_data_eff_eval.json")["scaling_curves"]
fr = ["f010", "f020", "f050", "f100"]; xs = [10, 20, 50, 100]
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for key, lab, c, ls_ in [("film_step", "FiLM (3,900-step matched)", C_FILM, "-"), ("bc_step", "BC (3,900-step matched)", C_BC, "-"),
                          ("bc_naive", "BC (30 epochs, naive)", C_GREY, ":")]:
    m_ = [sc[key][f]["success_mean"] for f in fr]; s_ = [sc[key][f]["success_sd"] for f in fr]
    ax.errorbar(xs, m_, yerr=s_, color=c, ls=ls_, marker="o", lw=2.2, capsize=4, label=lab)
    NUM[f"dataeff {key}"] = {f: (sc[key][f]["success_mean"], sc[key][f]["success_sd"]) for f in fr}
ax.axhline(sc["bc_naive"]["f100"]["success_mean"], color=C_BC, ls="--", lw=1)
ax.text(23, sc["bc_naive"]["f100"]["success_mean"] - 0.03, "100%-data BC30 = 0.814", color=C_BC, fontsize=12)
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["10%\n450", "20%\n900", "50%\n2,250", "100%\n4,500"])
ax.set_xlabel("Training demos (fraction / episodes)"); ax.set_ylabel("S6N3 Success\n(mean ± SD, n=3)")
ax.set_ylim(0.62, 1.0); ax.legend(frameon=False, loc="upper left", fontsize=11.5); ax.grid(ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(A / "p_dataeff.png"); plt.close(fig)
NUM["pat5 data eff csv"] = open(J / "data_efficiency.csv").read()

# ---------- 7. crops of approved composite figures ----------
def crop(src, box, dst):
    Image.open(FIG / src).crop(box).save(A / dst)
crop("final_report/fig1_baseline_reproduction.png", (0, 0, 660, 606), "c_loss.png")
crop("final_report/fig1_baseline_reproduction.png", (660, 0, 1310, 606), "c_sanity.png")
crop("final_report/fig1_baseline_reproduction.png", (1310, 0, 1950, 606), "c_perseed.png")
crop("final_report/fig2_film_main_results.png", (0, 0, 640, 624), "c_valacc.png")

json.dump(NUM, open(A / "numbers.json", "w"), indent=1, default=str)
for k, v in NUM.items():
    if "csv" not in k:
        print(k, v)

# ---------- 8. learning curves from runs/*/history.json (streamed read-only from 4090) ----------
H = ROOT / "src/hist"
def hist(prefix, key):
    arr = [[e[key] for e in json.load(open(H / f"{prefix}_seed{s}.json"))] for s in range(3)]
    return np.array(arr)
ep = np.arange(1, 31)
fig, ax = plt.subplots(figsize=(4.3, 2.35))
for pre, lab, c in [("b3_bc30", "BC30", C_BC), ("b2_film_nwd", "FiLM-nwd", C_FILM)]:
    a = hist(pre, "val_acc"); m_, s_ = a.mean(0), a.std(0)
    ax.plot(ep, m_, color=c, lw=2.2, label=lab); ax.fill_between(ep, m_ - s_, m_ + s_, color=c, alpha=0.2)
    NUM[f"curve best val_acc {pre}"] = [round(float(x), 6) for x in a.max(1)]
ax.set_xlabel("Epoch (≤30, same schedule)"); ax.set_ylabel("Val action acc")
ax.set_xlim(1, 30); ax.set_ylim(0.68, 1.0); ax.grid(ls="--", alpha=0.4); ax.legend(frameon=False, loc="lower right", fontsize=12, ncol=2, columnspacing=0.8, handlelength=1.4)
fig.tight_layout(); fig.savefig(A / "p_curves.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3))
for ax, key, ttl in [(axs[0], "train_loss", "Training cross-entropy"), (axs[1], "val_loss", "Validation cross-entropy (500 held-out eps)")]:
    for pre, lab, c in [("b3_bc30", "BC30", C_BC), ("b2_film_nwd", "FiLM-nwd", C_FILM)]:
        a = hist(pre, key); m_, s_ = a.mean(0), a.std(0)
        ax.plot(ep, m_, color=c, lw=2.2, label=lab); ax.fill_between(ep, m_ - s_, m_ + s_, color=c, alpha=0.2)
    ax.set_title(ttl, fontsize=14); ax.set_xlabel("Epoch"); ax.set_ylabel("CE loss"); ax.grid(ls="--", alpha=0.4); ax.set_xlim(1, 30)
for s, e in zip([1, 2], [13, 14]):
    axs[1].axvline(e, color=C_P5, ls=":", lw=1.8)
axs[1].text(14.5, axs[1].get_ylim()[1] * 0.97, "patience=5 stops\n(seeds 1, 2: ep 13, 14)", color=C_GREY, fontsize=11.5, va="top")
axs[0].legend(frameon=False, fontsize=11.5, loc="upper right")
fig.tight_layout(); fig.savefig(A / "p_loss_backup.png"); plt.close(fig)
for s in range(3):
    NUM[f"pat5 history len seed{s}"] = len(json.load(open(H / f"baseline_seed{s}.json")))
json.dump(NUM, open(A / "numbers.json", "w"), indent=1, default=str)
print({k: v for k, v in NUM.items() if "curve" in k or "history" in k})

# ---------- 9. slide-sized baseline reproduction figure (replaces dense composite on slide 4) ----------
fig, axs = plt.subplots(1, 3, figsize=(9.3, 2.75), gridspec_kw=dict(width_ratios=[1.1, 1.15, 0.95]))
ax = axs[0]
for key, lab, c, ls_ in [("train_loss", "train CE", "#1F6FE5", "-"), ("val_loss", "val CE", C_BC, "--")]:
    a = hist("b3_bc30", key); m_, s_ = a.mean(0), a.std(0)
    ax.plot(ep, m_, color=c, ls=ls_, lw=2.2, label=lab); ax.fill_between(ep, m_ - s_, m_ + s_, color=c, alpha=0.2)
for e in (13, 14):
    ax.axvline(e, color="#C0392B", ls=":", lw=1.8)
ax.text(14.6, 0.56, "patience=5 stops\nseeds 1, 2", color="#C0392B", fontsize=12)
ax.set_xlabel("Epoch (max 30)"); ax.set_ylabel("Cross-entropy"); ax.set_xlim(1, 30); ax.set_ylim(0, 0.7)
ax.legend(frameon=False, loc="lower left", fontsize=11.5, ncol=2, handlelength=1.5, columnspacing=0.8); ax.grid(ls="--", alpha=0.4)
ax.set_title("(a) BC30 loss, 3 seeds", fontsize=14, fontweight="bold")
ax = axs[1]
triv = {r["run"]: r["closed_loop_success"] for r in base["runs"] if "trivial" in r["run"]}
names = ["left\nonly", "fwd\nonly", "rand-\nom", "BC\npat5", "BC30"]
vals = [triv["trivial:always_left"], triv["trivial:always_forward"], triv["trivial:random"]]
bars = [(v, 0) for v in vals] + [(np.mean(p5_s6), pstd(p5_s6)), (np.mean(AGG["bc30"]["closed_loop_success"]["values"]), pstd(AGG["bc30"]["closed_loop_success"]["values"]))]
cols_ = ["#B0B8C4"] * 3 + [C_P5, C_BC]
for i, ((m_, s_), c) in enumerate(zip(bars, cols_)):
    ax.bar(i, m_, 0.65, yerr=s_ if s_ else None, color=c, edgecolor="#333", capsize=4)
    ax.text(i, m_ + (s_ or 0) + 0.02, f"{m_:.2f}" if i < 3 else f"{m_:.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=10.5); ax.set_ylim(0, 1.0)
ax.set_ylabel("S6N3 Success"); ax.grid(axis="y", ls="--", alpha=0.4)
ax.set_title("(b) Sanity checks", fontsize=14, fontweight="bold")
ax = axs[2]
for i, (lab, v6, v7, c) in enumerate(groups[:2]):
    for off, v, hatch in [(-0.2, v6, None), (0.2, v7, "//")]:
        ax.bar(i + off, np.mean(v), 0.38, yerr=pstd(v), color=c if hatch is None else "white", edgecolor=c, hatch=hatch, lw=2, capsize=4)
        for k, val in enumerate(v):
            ax.plot(i + off + (k - 1) * 0.08, val, MK[k], ms=5.5, mfc="white", mec="#222", zorder=5)
        ax.text(i + off, max(max(v), np.mean(v) + pstd(v)) + 0.03, f"{np.mean(v):.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["BC (pat=5)", "BC30"]); ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Success"); ax.grid(axis="y", ls="--", alpha=0.4)
ax.set_title("(c) S6N3 / S7N4 (hatched)", fontsize=12.5, fontweight="bold")
fig.tight_layout(w_pad=1.0); fig.savefig(A / "p_baseline.png"); plt.close(fig)


# ---------- 10. FiLM vs Attention & FiLM+Attention ablation (backup B4) ----------
a_ev = load("b2_attn_eval.json")
a_f20 = load("b2_attn_f020_eval.json")
a_pr = load("b2_attn_probe.json")
a_fl = load("b2_attn_failures.json")

a_order = [
    ("bc30", "BC30\n(Concat)", C_BC),
    ("film_nwd", "FiLM\n(Primary)", C_FILM),
    ("spatial_attn_gate", "Spatial\nGate", "#2563EB"),
    ("spatial_softmax_attn", "Spatial\nSoftmax", "#1D4ED8"),
    ("cross_attn_mha", "Cross\nMHA", "#6366F1"),
    ("film_spatial_attn", "FiLM+\nSpatial", "#0D9488"),
    ("film_cross_mha", "FiLM+\nMHA", "#0F766E"),
]

def _attn_stat(k, env_tag):
    if k in ("bc30", "film_nwd"):
        d = a_ev["references"][k][env_tag]["closed_loop_success"]
    else:
        d = a_ev["variants"][k][env_tag]["closed_loop_success"]
    return d["mean"], d["sd"], d["values"]

fig, ax = plt.subplots(figsize=(5.65, 3.35))
xa = np.arange(len(a_order))
wa = 0.25
for idx, (env_tag, hatch, alpha, lbl) in enumerate([
    ("S6N3", None, 0.92, "S6N3"),
    ("S7N4", "//", 0.55, "S7N4 zero-shot"),
    ("S8N6", "..", 0.35, "S8N6 (10000:)"),
]):
    for i, (k, _, col) in enumerate(a_order):
        m_, s_, _ = _attn_stat(k, env_tag)
        ax.bar(i + (idx - 1) * wa, m_, wa, yerr=s_, capsize=2.2,
               color=col if hatch is None else "white", edgecolor=col, hatch=hatch,
               lw=1.5, alpha=alpha if hatch is None else 1.0,
               error_kw=dict(lw=1.1, ecolor="#222"))
        if env_tag == "S6N3":
            ax.text(i + (idx - 1) * wa, m_ + s_ + 0.015, f"{m_:.3f}",
                    ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#111")
        NUM[f"b2_attn {k} {env_tag}"] = f"{m_:.3f} ± {s_:.3f}"
for k in ("spatial_attn_gate", "cross_attn_mha", "film_spatial_attn"):
    d20 = a_f20["variants"][k]["S6N3"]["closed_loop_success"]
    NUM[f"b2_attn_f020 {k}"] = f"{d20['mean']:.3f} ± {d20['sd']:.3f}"
for k, _, _ in a_order:
    NUM[f"b2_attn_probe {k}"] = round(a_pr["by_variant"][k]["color_swap_sensitivity"]["mean"], 3)
    NUM[f"b2_attn_4502 {k}"] = a_fl["cases"][k]["4502"]["n_solved"]
    NUM[f"b2_attn_4501 {k}"] = a_fl["cases"][k]["4501"]["n_solved"]

ax.set_xticks(xa); ax.set_xticklabels([x[1] for x in a_order], fontsize=9.8, fontweight="bold")
ax.set_ylim(0.40, 1.08); ax.set_ylabel("Closed-loop Success\n(mean ± SD, ddof=0)", fontsize=11.5)
ax.grid(axis="y", ls="--", alpha=0.4)
ax.legend(handles=[
    Patch(facecolor="#555", edgecolor="#222", label="S6N3 (in-dist)"),
    Patch(facecolor="white", edgecolor="#222", hatch="//", label="S7N4 (zero-shot)"),
    Patch(facecolor="white", edgecolor="#222", hatch="..", label="S8N6 (8×8 N=6)"),
], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=9.8, columnspacing=0.9, handlelength=1.4)
fig.tight_layout(); fig.savefig(A / "p_attn_ablation.png", dpi=250); plt.close(fig)

# Crop row-labels + Spatial-Softmax + FiLM+Spatial columns from b2_attn_heatmap.png
hm = Image.open(FIG / "b2_attn/b2_attn_heatmap.png")
W_hm, H_hm = hm.size
left_strip = hm.crop((0, 75, 72, H_hm))
col_soft = hm.crop((918, 75, 1762, H_hm))
col_fus = hm.crop((2685, 75, W_hm, H_hm))
canvas = Image.new("RGB", (left_strip.width + col_soft.width + col_fus.width, H_hm - 75), "white")
canvas.paste(left_strip, (0, 0))
canvas.paste(col_soft, (left_strip.width, 0))
canvas.paste(col_fus, (left_strip.width + col_soft.width, 0))
canvas.save(A / "p_attn_heatmap.png")

json.dump(NUM, open(A / "numbers.json", "w"), indent=1, default=str)
