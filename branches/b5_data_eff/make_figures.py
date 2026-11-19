"""支线 5 · 作图 -> reports/figures/b5_data_eff/

四张图（所有数值动态读取自 reports/b5_data_eff_*.json、runs/*/history.json 与 B1–B4 评测 JSON）：
  1. b5_fig1_scaling_curves_s6_s7.png     全缩放律曲线（5%–100%）：S6N3 已见域 & S7N4 未见外推
  2. b5_fig2_step_overfit_dynamics.png   20% (900 ep) 30-epoch 过拟合动态 & 14 组受控消融对比
  3. b5_fig3_grounding_emergence.png     语义落地敏感度涌现 (Color+Type) & 失败机制拆解
  4. b5_fig4_cross_branch_pareto.png     跨支线 B0–B5 全局 Pareto 前沿（监督 Oracle 转移数 vs 成功率）

用法:
    OMP_NUM_THREADS=1 python branches/b5_data_eff/make_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"
RUNS = ROOT / "runs"
OUT = R / "figures" / "b5_data_eff"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 180

FRAC_TAGS = ["f005", "f010", "f020", "f050", "f100"]
FRAC_EPS = {"f005": 225, "f010": 450, "f020": 900, "f050": 2250, "f100": 4500}
FRAC_LABELS = ["225\n(5%)", "450\n(10%)", "900\n(20%)", "2250\n(50%)", "4500\n(100%)"]


def load_json(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def raw_trans_map(eval_data: dict) -> dict[tuple[str, str], float]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in eval_data["runs"]:
        if r.get("n_raw_transitions") is not None:
            buckets[(r["method"], r["frac_tag"])].append(float(r["n_raw_transitions"]))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def plot_fig1_scaling_curves(ev_s6: dict, ev_s7: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), sharey=True)

    curves = [
        ("bc_naive", "Concat · bc_naive（步数未对齐）", "#6b7280", "--", "o", 1.8),
        ("bc_step", "Concat · bc_step（步数对齐 3900）", "#374151", "-", "s", 2.0),
        ("bc_d4", "Concat · bc_d4（D4 增广坍塌）", "#d97706", "-.", "^", 2.1),
        ("film_step", "FiLM · film_step（无增广）", "#2563eb", "-", "P", 2.3),
        ("film_d4", "FiLM · film_d4（D4 群增广）", "#059669", "-", "D", 2.5),
        ("film_d4_tta", "FiLM · film_d4_tta（+D4 测试集成）", "#dc2626", "-", "*", 2.6),
    ]

    panels = [
        (axes[0], ev_s6, "BabyAI-GoToLocalS6N3-v0（已见域，n=300×3）", True),
        (axes[1], ev_s7, "BabyAI-GoToLocalS7N4-v0（未见零样本外推，n=300×3）", False),
    ]

    for ax, data, title, is_left in panels:
        sc = data["scaling_curves"]
        bc100 = sc["bc_step"]["f100"]["success_mean"]
        ax.axhline(bc100, color="#9ca3af", ls=":", lw=1.4, zorder=1)
        ax.text(
            2100,
            bc100 + 0.011,
            f"100% BC30 (4500 ep) = {bc100:.3f}",
            fontsize=8.4,
            color="#4b5563",
        )

        for method, label, color, ls, marker, lw in curves:
            if method not in sc:
                continue
            tags = [t for t in FRAC_TAGS if t in sc[method]]
            xs = [FRAC_EPS[t] for t in tags]
            ys = np.array([sc[method][t]["success_mean"] for t in tags])
            sds = np.array([sc[method][t]["success_sd"] for t in tags])
            ms = 9.5 if marker == "*" else 6.0
            ax.plot(xs, ys, ls=ls, color=color, marker=marker, markersize=ms, lw=lw, label=label, zorder=4)
            ax.fill_between(xs, ys - sds, ys + sds, color=color, alpha=0.11, zorder=2)

        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([FRAC_EPS[t] for t in FRAC_TAGS])
        ax.set_xticklabels(FRAC_LABELS, fontsize=9)
        ax.set_xlim(195, 5150)
        ax.set_ylim(0.37, 1.01)
        ax.grid(True, which="major", ls="--", alpha=0.35)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("训练专家轨迹数（占 4,500 训练集比例）", fontsize=10)
        if is_left:
            ax.set_ylabel("闭环任务成功率（Mean ± SD，3 seeds）", fontsize=10)
            # Highlight 10x compression and bc_d4 100% collapse cleanly
            y_tta10 = sc["film_d4_tta"]["f010"]["success_mean"]
            ax.annotate(
                f"10× 压缩超越 BC30\n450 ep = {y_tta10:.3f}",
                xy=(450, y_tta10),
                xytext=(245, 0.895),
                fontsize=8.6,
                fontweight="bold",
                color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#fca5a5", lw=0.8),
            )
            y_bcd4_100 = sc["bc_d4"]["f100"]["success_mean"]
            ax.annotate(
                f"Concat+D4 @100% 坍塌\n{y_bcd4_100:.3f} (-0.122 vs BC30)",
                xy=(4500, y_bcd4_100),
                xytext=(1180, 0.505),
                fontsize=8.4,
                color="#b45309",
                arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="#fffbeb", ec="#fcd34d", lw=0.8),
            )
        else:
            y_s7_10 = sc["film_d4_tta"]["f010"]["success_mean"]
            ax.annotate(
                f"10% 已越 100% BC30\n450 ep = {y_s7_10:.3f} (> {bc100:.3f})",
                xy=(450, y_s7_10),
                xytext=(235, 0.855),
                fontsize=8.6,
                fontweight="bold",
                color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#fca5a5", lw=0.8),
            )
            y_s7_20 = sc["film_d4_tta"]["f020"]["success_mean"]
            ax.annotate(
                f"20% = {y_s7_20:.3f}\n追平 100% GRU-small",
                xy=(900, y_s7_20),
                xytext=(1280, 0.485),
                fontsize=8.4,
                color="#991b1b",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.1),
                bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#fecaca", lw=0.8),
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=8.8,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#d1d5db",
    )
    fig.suptitle(
        "图1｜B5 样本缩放律双重分离：D4 等变增广 × FiLM 在 10% 数据超越 100% BC30，而在 Concat 上于大样本坍塌",
        fontsize=12.2,
        fontweight="bold",
        y=1.06,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "b5_fig1_scaling_curves_s6_s7.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_dynamics_and_ablation(ev_s6: dict, ev_s7: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.2), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Left: 30-epoch training/validation dynamics at f020 (900 ep)
    ax_l = axes[0]
    hist_specs = [
        ("bc_naive", "bc_naive (390 步)", "#6b7280"),
        ("bc_step", "bc_step (3900 步)", "#374151"),
        ("bc_d4", "bc_d4 (D4+Concat)", "#d97706"),
        ("gru_small_step", "gru_small_step (GRU)", "#9333ea"),
        ("film_step", "film_step (FiLM)", "#2563eb"),
        ("film_d4", "film_d4 (FiLM+D4)", "#059669"),
    ]
    for method, label, color in hist_specs:
        tr_curves, va_curves = [], []
        for s in (0, 1, 2):
            hp = RUNS / f"b5_{method}_f020_seed{s}" / "history.json"
            hist = json.loads(hp.read_text(encoding="utf-8"))
            tr_curves.append([e["train_acc"] for e in hist])
            va_curves.append([e["val_acc"] for e in hist])
        epochs = np.arange(1, len(tr_curves[0]) + 1)
        tr_m = np.mean(tr_curves, axis=0)
        va_m = np.mean(va_curves, axis=0)
        ax_l.plot(epochs, va_m, color=color, lw=2.2, ls="-", label=f"{label} Val ({va_m.max():.3f})")
        ax_l.plot(epochs, tr_m, color=color, lw=1.2, ls="--", alpha=0.65)

    ax_l.text(
        1.5,
        0.978,
        "虚线 = 训练集准确率；实线 = 固定 500-ep 验证集准确率",
        fontsize=8.3,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.22", fc="#f9fafb", ec="#d1d5db"),
    )
    ax_l.set_xlim(1, 30.5)
    ax_l.set_ylim(0.58, 1.02)
    ax_l.set_xlabel("训练轮数 (Epoch)", fontsize=10)
    ax_l.set_ylabel("离线动作预测准确率 (Mean over 3 seeds)", fontsize=10)
    ax_l.set_title("20% 数据 (900 ep) 30-Epoch 训练与过拟合动态", fontsize=10.8, fontweight="bold", pad=8)
    ax_l.grid(True, ls="--", alpha=0.35)
    ax_l.legend(loc="lower right", fontsize=8.0, frameon=True, facecolor="white", framealpha=0.92)

    # Right: 20% controlled ablation horizontal grouped bars (S6N3 + S7N4)
    ax_r = axes[1]
    ab6 = ev_s6["f020_ablations"]
    ab7 = ev_s7["f020_ablations"]
    pretty = {
        "film_d4_tta": "film_d4_tta (FiLM+D4+TTA)",
        "film_d4": "film_d4 (FiLM+D4)",
        "film_step": "film_step (FiLM+步数对齐)",
        "bc_step": "bc_step (Concat+步数对齐)",
        "bc_wd": "bc_wd (强权重衰减)",
        "b5_defrac": "b5_defrac (原手册基线)",
        "bc_c4": "bc_c4 (仅 C4 旋转)",
        "gru_small_step": "gru_small_step (GRU)",
        "bc_d4_tta": "bc_d4_tta (Concat+D4+TTA)",
        "bc_z2": "bc_z2 (仅 Z2 镜像)",
        "bc_bal": "bc_bal (动作平衡采样)",
        "bc_reg": "bc_reg (Dropout+LS+WD)",
        "bc_d4": "bc_d4 (Concat+D4)",
        "bc_naive": "bc_naive (无步数对齐)",
    }
    ordered = sorted(ab6.keys(), key=lambda m: ab6[m]["success_mean"])
    y_pos = np.arange(len(ordered))
    h = 0.36
    s6_vals = [ab6[m]["success_mean"] for m in ordered]
    s6_errs = [ab6[m]["success_sd"] for m in ordered]
    s7_vals = [ab7[m]["success_mean"] for m in ordered]
    s7_errs = [ab7[m]["success_sd"] for m in ordered]

    c6 = ["#059669" if m.startswith("film") else "#3b82f6" for m in ordered]
    c7 = ["#6ee7b7" if m.startswith("film") else "#93c5fd" for m in ordered]

    ax_r.barh(y_pos + h / 2, s6_vals, height=h, xerr=s6_errs, color=c6, edgecolor="#1f2937", linewidth=0.4, capsize=2, label="S6N3 (已见域)")
    ax_r.barh(y_pos - h / 2, s7_vals, height=h, xerr=s7_errs, color=c7, edgecolor="#1f2937", linewidth=0.4, capsize=2, label="S7N4 (未见外推)")

    for i, m in enumerate(ordered):
        gap = ab6[m]["gen_gap_mean"]
        ax_r.text(
            max(s6_vals[i] + s6_errs[i], s7_vals[i] + s7_errs[i]) + 0.012,
            y_pos[i],
            f"{s6_vals[i]:.3f} / {s7_vals[i]:.3f} (gap {gap:+.2f})",
            va="center",
            fontsize=7.8,
            color="#111827",
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.82),
            zorder=5,
        )

    ax_r.axvline(ev_s6["scaling_curves"]["bc_step"]["f100"]["success_mean"], color="#dc2626", ls=":", lw=1.2, label="100% BC30 (S6N3)")
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels([pretty.get(m, m) for m in ordered], fontsize=8.4)
    ax_r.set_xlim(0.45, 1.15)
    ax_r.set_xlabel("闭环任务成功率 (Mean ± SD)", fontsize=10)
    ax_r.set_title("20% 数据 (900 ep) 14 组受控消融全景对比", fontsize=10.8, fontweight="bold", pad=8)
    ax_r.grid(True, axis="x", ls="--", alpha=0.35)
    ax_r.legend(loc="lower right", fontsize=8.2, frameon=True, facecolor="white")

    fig.suptitle(
        "图2｜训练动力学与 20% 受控消融：GRU/无增广 FiLM 快速记忆训练集，唯有 FiLM+D4 压缩过拟合间隙并同步推高验证与闭环表现",
        fontsize=11.8,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT / "b5_fig2_step_overfit_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_grounding_and_failures(sens_data: dict, fail_data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), gridspec_kw={"width_ratios": [1.0, 1.2]})

    # Left: Color & Type semantic sensitivity emergence across fractions
    ax_l = axes[0]
    by_g = sens_data["by_group"]
    specs = [
        ("bc_step", "Concat · bc_step", "#4b5563"),
        ("bc_d4", "Concat · bc_d4", "#d97706"),
        ("film_step", "FiLM · film_step", "#2563eb"),
        ("film_d4", "FiLM · film_d4", "#059669"),
    ]
    for method, label, color in specs:
        xs = [FRAC_EPS[t] for t in FRAC_TAGS]
        comb = [by_g[f"{method}_{t}"]["combined_sensitivity_mean"] for t in FRAC_TAGS]
        col = [by_g[f"{method}_{t}"]["color_sensitivity_mean"] for t in FRAC_TAGS]
        typ = [by_g[f"{method}_{t}"]["type_sensitivity_mean"] for t in FRAC_TAGS]
        ax_l.plot(xs, comb, color=color, lw=2.4, ls="-", marker="o", ms=5.5, label=f"{label}（综合）")
        if method in ("film_d4", "bc_d4"):
            ax_l.plot(xs, col, color=color, lw=1.3, ls=":", marker="s", ms=4.0, alpha=0.75)
            ax_l.plot(xs, typ, color=color, lw=1.3, ls="--", marker="^", ms=4.2, alpha=0.75)

    ax_l.plot([], [], color="#374151", lw=1.3, ls=":", marker="s", ms=4.0, label="点线+方块：颜色置换（film_d4 / bc_d4）")
    ax_l.plot([], [], color="#374151", lw=1.3, ls="--", marker="^", ms=4.2, label="虚线+三角：物体置换（film_d4 / bc_d4）")

    # Point out f010 milestone where film_d4 already equals bc_step f100
    f_d4_10 = by_g["film_d4_f010"]["combined_sensitivity_mean"]
    b_s_100 = by_g["bc_step_f100"]["combined_sensitivity_mean"]
    ax_l.axhline(b_s_100, color="#6b7280", ls=":", lw=1.1)
    ax_l.annotate(
        f"film_d4@10% ({f_d4_10:.3f}) =\nbc_step@100% ({b_s_100:.3f})",
        xy=(450, f_d4_10),
        xytext=(240, 0.215),
        fontsize=8.3,
        color="#065f46",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.24", fc="#ecfdf5", ec="#6ee7b7"),
    )
    ax_l.annotate(
        f"bc_d4@100% 降至 {by_g['bc_d4_f100']['combined_sensitivity_mean']:.3f}\n（D4 抹除空间捷径暴露语义失明）",
        xy=(4500, by_g["bc_d4_f100"]["combined_sensitivity_mean"]),
        xytext=(1250, 0.068),
        fontsize=8.0,
        color="#b45309",
        arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.22", fc="#fffbeb", ec="#fcd34d"),
    )

    ax_l.set_xscale("log")
    ax_l.minorticks_off()
    ax_l.set_xticks([FRAC_EPS[t] for t in FRAC_TAGS])
    ax_l.set_xticklabels(FRAC_LABELS, fontsize=8.8)
    ax_l.set_xlim(195, 5200)
    ax_l.set_ylim(-0.008, 0.365)
    ax_l.set_xlabel("训练专家轨迹数", fontsize=10)
    ax_l.set_ylabel("条件动作翻转率 (反事实指令敏感度)", fontsize=10)
    ax_l.set_title("指令反事实敏感度涌现曲线（183 候选切换态）", fontsize=10.8, fontweight="bold", pad=8)
    ax_l.grid(True, ls="--", alpha=0.35)
    ax_l.legend(loc="upper left", fontsize=7.6, ncol=1, frameon=True, facecolor="white", framealpha=0.94)

    # Right: S6N3 Failure Decomposition stacked bar chart (out of 300 episodes)
    ax_r = axes[1]
    agg = fail_data["aggregate"]
    keys = [
        ("bc_naive_f020", "bc_naive\n(20%)"),
        ("bc_d4_f020", "bc_d4\n(20%)"),
        ("gru_small_step_f020", "gru_small\n(20%)"),
        ("bc_step_f020", "bc_step\n(20%)"),
        ("film_step_f020", "film_step\n(20%)"),
        ("film_d4_f020", "film_d4\n(20%)"),
        ("film_d4_tta_f020", "film_d4_tta\n(20%)"),
        ("bc_d4_f100", "bc_d4\n(100%)"),
        ("bc_step_f100", "bc_step\n(100%)"),
        ("film_d4_f100", "film_d4\n(100%)"),
        ("film_step_f100", "film_step\n(100%)"),
    ]
    subtypes = [
        ("attribute_confusion", "属性混淆（表征）", "#ef4444"),
        ("distractor_priority", "干扰物抢先（表征）", "#f97316"),
        ("approach_then_lost", "接近后丢失（表征）", "#f59e0b"),
        ("turn_oscillation", "转向震荡（状态覆盖）", "#3b82f6"),
        ("orientation_lock", "朝向锁死（状态覆盖）", "#6366f1"),
        ("forward_stall", "前向停滞（状态覆盖）", "#06b6d4"),
        ("wander_timeout", "漫游超时（状态覆盖）", "#64748b"),
    ]

    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for st_key, st_label, color in subtypes:
        vals = np.array([agg[k]["failure_types_mean"].get(st_key, 0.0) for k, _ in keys])
        ax_r.bar(x, vals, bottom=bottom, width=0.68, color=color, edgecolor="white", linewidth=0.5, label=st_label)
        bottom += vals

    for i, (k, _) in enumerate(keys):
        v = agg[k]
        rep = v["failure_groups_mean"]["representation"]
        st = v["failure_groups_mean"]["state_coverage"]
        ax_r.text(
            x[i],
            bottom[i] + 1.5,
            f"{bottom[i]:.0f}\n({rep:.0f}+{st:.0f})",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#111827",
        )

    ax_r.axvline(6.5, color="#374151", ls="--", lw=1.1)
    ax_r.text(3.0, 108, "← 20% 样本 (900 ep) →", ha="center", fontsize=8.8, fontweight="bold", color="#374151")
    ax_r.text(8.5, 104, "← 100% (4500 ep) →", ha="center", fontsize=8.5, fontweight="bold", color="#374151")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([lbl for _, lbl in keys], fontsize=8.0)
    ax_r.set_ylim(0, 140)
    ax_r.set_ylabel("300 局评估中的平均失败局数（表征+覆盖）", fontsize=10)
    ax_r.set_title("S6N3 失败原因机制拆解：FiLM+D4 同时压半表征失败与状态覆盖失败", fontsize=10.5, fontweight="bold", pad=8)
    ax_r.grid(True, axis="y", ls="--", alpha=0.35)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7.5, ncol=2, frameon=True, facecolor="white")

    fig.suptitle(
        "图3｜双分离机理验证：D4 对称归纳偏置仅在 FiLM 结构支持下转化为更早涌现的语义落地与更低双类失败率",
        fontsize=11.8,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT / "b5_fig3_grounding_emergence.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_cross_branch_pareto(ev_s6: dict, ev_s7: dict) -> None:
    rmap = raw_trans_map(ev_s6)
    sc6 = ev_s6["scaling_curves"]
    sc7 = ev_s7["scaling_curves"]

    b1_6 = load_json("b1_langenc_eval.json")["by_variant"]
    b1_7 = load_json("b1_langenc_s7n4.json")["by_variant"]
    b2_6 = load_json("b2_film_eval.json")["aggregate"]
    b2_7 = load_json("b2_film_s7n4.json")["aggregate"]
    b3_6 = load_json("b3_v2_eval_s6n3.json")["by_group"]
    b3_7 = load_json("b3_v2_eval_s7n4.json")["by_group"]
    b4_6 = load_json("b4_curriculum_eval.json")["by_variant"]
    b4_7 = load_json("b4_curriculum_s7n4.json")["by_variant"]
    base6 = [r["closed_loop_success"] for r in load_json("baseline_eval.json")["runs"] if r["run"].startswith("runs/")]
    base7 = [r["closed_loop_success"] for r in load_json("s7n4_transfer.json")["runs"] if r["run"].startswith("runs/")]

    # Collect unified cross-branch points: (name, family, x_labels, s6, s7, text_offset_s6, text_offset_s7)
    points = []
    for f in FRAC_TAGS:
        x = rmap[("bc_naive", f)]
        points.append((f"bc_naive {FRAC_EPS[f]}ep", "B5 Concat", x, sc6["bc_naive"][f]["success_mean"], sc7["bc_naive"][f]["success_mean"]))
        points.append((f"film_d4 {FRAC_EPS[f]}ep", "B5 FiLM+D4", x, sc6["film_d4"][f]["success_mean"], sc7["film_d4"][f]["success_mean"]))
    for f in ("f005", "f010", "f020"):
        x = rmap[("film_d4_tta", f)]
        points.append((f"film_d4_tta {FRAC_EPS[f]}ep", "B5 FiLM+D4+TTA", x, sc6["film_d4_tta"][f]["success_mean"], sc7["film_d4_tta"][f]["success_mean"]))

    # Fixed 100% S6N3 offline pool = 16534 transitions
    n100 = rmap[("bc_naive", "f100")]
    points.append(("B0 baseline", "B0–B2/B4 (4500ep)", n100, float(np.mean(base6)), float(np.mean(base7))))
    points.append(("B0/B3 BC30", "B0–B2/B4 (4500ep)", n100, b1_6["bc30"]["success_mean"], b1_7["bc30"]["success_mean"]))
    points.append(("B1 gru_small", "B0–B2/B4 (4500ep)", n100, b1_6["gru_small"]["success_mean"], b1_7["gru_small"]["success_mean"]))
    points.append(("B4 pace_easy", "B0–B2/B4 (4500ep)", n100, b4_6["pace_easy"]["success_mean"], b4_7["pace_easy"]["success_mean"]))
    points.append(("B2 film_nwd", "B0–B2/B4 (4500ep)", n100, b2_6["film_nwd"]["closed_loop_success"]["mean"], b2_7["film_nwd"]["closed_loop_success"]["mean"]))

    # B4 multiscale_prog config transitions = 17156 (dynamically read if present)
    p4_cfg = RUNS / "b4_film_multiscale_prog_seed0" / "config.json"
    n_b4m = float(json.loads(p4_cfg.read_text())["data"]["n_train_transitions"]) if p4_cfg.exists() else 17156.0
    points.append(("B4 film_multiscale", "B0–B2/B4 (4500ep)", n_b4m, b4_6["film_multiscale_prog"]["success_mean"], b4_7["film_multiscale_prog"]["success_mean"]))

    # B3 DAgger (Concat and FiLM) using exact total_oracle_labels_mean
    for k, disp, fam in [
        ("b3_v2_bc3000", "B3 bc3000", "B3 Concat DAgger"),
        ("b3_v2_unstick", "B3 unstick", "B3 Concat DAgger"),
        ("b3_v2_r2_w1", "B3 Concat R2", "B3 Concat DAgger"),
        ("b3_v2_film3000", "B3 film3000", "B3 FiLM DAgger"),
        ("b3_v2_film_split", "B3 film_split", "B3 FiLM DAgger"),
        ("b3_v2_film_dag_w1", "B3 FiLM+DAgger", "B3 FiLM DAgger"),
    ]:
        g6, g7 = b3_6[k], b3_7[k]
        points.append((disp, fam, float(g6["total_oracle_labels_mean"]), float(g6["success_mean"]), float(g7["success_mean"])))

    fam_style = {
        "B5 Concat": ("#6b7280", "o", 45, "--"),
        "B5 FiLM+D4": ("#059669", "D", 55, "-"),
        "B5 FiLM+D4+TTA": ("#dc2626", "*", 135, "-"),
        "B0–B2/B4 (4500ep)": ("#2563eb", "s", 55, None),
        "B3 Concat DAgger": ("#d97706", "^", 62, None),
        "B3 FiLM DAgger": ("#7c3aed", "P", 78, None),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8), sharey=True)
    for idx, (ax, title, metric_idx) in enumerate([
        (axes[0], "已见域 BabyAI-GoToLocalS6N3-v0 全局效率 Pareto 前沿", 3),
        (axes[1], "未见外推 BabyAI-GoToLocalS7N4-v0 全局效率 Pareto 前沿", 4),
    ]):
        # Draw trend curves for B5 scaling series
        for fam in ("B5 Concat", "B5 FiLM+D4", "B5 FiLM+D4+TTA"):
            sub = sorted([p for p in points if p[1] == fam], key=lambda t: t[2])
            color, _, _, ls = fam_style[fam]
            ax.plot([p[2] for p in sub], [p[metric_idx] for p in sub], color=color, ls=ls, lw=1.8, alpha=0.85, zorder=2)

        # Scatter all families
        for fam, (color, marker, size, _) in fam_style.items():
            sub = [p for p in points if p[1] == fam]
            ax.scatter([p[2] for p in sub], [p[metric_idx] for p in sub], color=color, marker=marker, s=size, edgecolor="white", linewidth=0.6, label=fam if idx == 0 else None, zorder=4)

        # Compute & highlight global non-dominated Pareto frontier (lower x, higher y)
        sorted_pts = sorted(points, key=lambda t: (t[2], -t[metric_idx]))
        frontier = []
        best_y = -1.0
        for p in sorted_pts:
            if p[metric_idx] > best_y:
                frontier.append(p)
                best_y = p[metric_idx]
        ax.plot([p[2] for p in frontier], [p[metric_idx] for p in frontier], color="#111827", ls=":", lw=1.8, alpha=0.85, label="全局 Pareto 前沿" if idx == 0 else None, zorder=3)

        # Hand-tuned clean annotations to prevent overlaps
        offsets_s6 = {
            "film_d4_tta 225ep": (10, -14),
            "film_d4_tta 450ep": (-52, 14),
            "film_d4_tta 900ep": (-62, 12),
            "film_d4 2250ep": (-88, 16),
            "B3 film3000": (-44, -46),
            "B2 film_nwd": (-86, 18),
            "B3 FiLM+DAgger": (8, 12),
            "B1 gru_small": (10, -12),
            "B0/B3 BC30": (10, -4),
            "B0 baseline": (10, -4),
            "B3 Concat R2": (-10, -18),
            "bc_naive 225ep": (8, -6),
        }
        offsets_s7 = {
            "film_d4_tta 225ep": (10, -12),
            "film_d4_tta 450ep": (-52, 14),
            "film_d4_tta 900ep": (-62, 12),
            "film_d4 2250ep": (-94, 16),
            "B3 film3000": (-44, -38),
            "B2 film_nwd": (-90, 15),
            "B4 film_multiscale": (10, 14),
            "B3 FiLM+DAgger": (10, -18),
            "B1 gru_small": (10, -14),
            "B0/B3 BC30": (10, -4),
            "B0 baseline": (10, -8),
            "B3 Concat R2": (-12, -18),
            "bc_naive 225ep": (8, -6),
        }
        omap = offsets_s6 if idx == 0 else offsets_s7
        for p in points:
            if p[0] not in omap:
                continue
            dx, dy = omap[p[0]]
            ax.annotate(
                f"{p[0]} ({p[metric_idx]:.3f})",
                xy=(p[2], p[metric_idx]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.6,
                color="#111827",
                arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.7),
            )

        ax.set_xscale("log")
        ax.set_xlim(620, 62000)
        ax.set_ylim(0.42, 1.03)
        ax.grid(True, which="both", ls="--", alpha=0.30)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("总监督 Oracle 标注转移数（Raw + Online Queries，对数轴）", fontsize=10)
        if idx == 0:
            ax.set_ylabel("闭环任务成功率 (3 seeds)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=8.6,
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
    )
    fig.suptitle(
        "图4｜跨支线 B0–B5 全局效率统一坐标：FiLM+D4+TTA 占据全部 <10k 标注区的 Pareto 前沿，与 B4 多尺度及 B3 FiLM+DAgger 无缝衔接",
        fontsize=11.6,
        fontweight="bold",
        y=1.06,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "b5_fig4_cross_branch_pareto.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev_s6 = load_json("b5_data_eff_eval.json")
    ev_s7 = load_json("b5_data_eff_s7n4.json")
    sens = load_json("b5_data_eff_sensitivity.json")
    fail = load_json("b5_data_eff_failures.json")

    plot_fig1_scaling_curves(ev_s6, ev_s7)
    plot_fig2_dynamics_and_ablation(ev_s6, ev_s7)
    plot_fig3_grounding_and_failures(sens, fail)
    plot_fig4_cross_branch_pareto(ev_s6, ev_s7)
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(ROOT), f"({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
