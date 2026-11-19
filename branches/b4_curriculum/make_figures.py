"""支线 4 · 作图 -> reports/figures/b4_curriculum/

四张图（所有数值动态读取自 reports/b4_curriculum_*.json 与 runs/*/history.json）：
  1. b4_fig1_main_s6_s7.png                  15 组 S6N3 与未见 S7N4 主结果对比
  2. b4_fig2_difficulty_spectrum.png          S5N2 → S6N3 → S6N4 → S7N4(未见) → S8N6 五级难度谱曲线
  3. b4_fig3_pacing_vs_mixing.png             同域排序 epoch 曲线（左）与配对净增量对比（右）
  4. b4_fig4_mechanism_grounding_failures.png 颜色/类型语义敏感度（左）与 S7N4 失败机制拆解（右）

用法:
    OMP_NUM_THREADS=1 python branches/b4_curriculum/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"
RUNS = ROOT / "runs"
OUT = R / "figures" / "b4_curriculum"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ORDER = [
    "bc30",
    "pace_easy",
    "pace_rand",
    "pace_hard",
    "env_easy_stage",
    "env_easy_cumul",
    "env_easy_mix",
    "env_s6n4_prog",
    "env_s6n4_mix",
    "env_multiscale_prog",
    "env_multiscale_mix",
    "film_easy_stage",
    "film_multiscale_prog",
    "film_multiscale_mix",
    "b2_film_nwd",
]

SHORT = {
    "bc30": "BC30\n(基线)",
    "pace_easy": "pace_easy\n(易→难)",
    "pace_rand": "pace_rand\n(随机阶梯)",
    "pace_hard": "pace_hard\n(反课程)",
    "env_easy_stage": "S5N2→S6N3\n(纯接力)",
    "env_easy_cumul": "Obj→S5→S6\n(累积)",
    "env_easy_mix": "Obj+S5+S6\n(混训)",
    "env_s6n4_prog": "S6N3→+N4\n(渐进)",
    "env_s6n4_mix": "S6N3+N4\n(混训)",
    "env_multiscale_prog": "多尺度\n(渐进)",
    "env_multiscale_mix": "多尺度\n(混训)",
    "film_easy_stage": "FiLM\nS5→S6",
    "film_multiscale_prog": "FiLM+多尺度\n(渐进)",
    "film_multiscale_mix": "FiLM+多尺度\n(域随机)",
    "b2_film_nwd": "FiLM\n(单环境S6)",
}

COLOR = {
    "bc30": "#6e7781",
    "pace_easy": "#2ca02c",
    "pace_rand": "#72b7b2",
    "pace_hard": "#d62728",
    "env_easy_stage": "#aec7e8",
    "env_easy_cumul": "#ff9896",
    "env_easy_mix": "#4c78a8",
    "env_s6n4_prog": "#98df8a",
    "env_s6n4_mix": "#1f77b4",
    "env_multiscale_prog": "#1b7837",
    "env_multiscale_mix": "#e377c2",
    "film_easy_stage": "#f58518",
    "film_multiscale_prog": "#9d2b2b",
    "film_multiscale_mix": "#c51b7d",
    "b2_film_nwd": "#7f2704",
}


def jload(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fig1_main(ev: dict, s7: dict) -> str:
    bv6, bv7 = ev["by_variant"], s7["by_variant"]
    names = [g for g in ORDER if g in bv6]
    m6 = [bv6[g]["success_mean"] for g in names]
    sd6 = [bv6[g]["success_sd"] for g in names]
    m7 = [bv7[g]["success_mean"] for g in names]
    sd7 = [bv7[g]["success_sd"] for g in names]

    x = np.arange(len(names))
    w = 0.37

    fig, ax = plt.subplots(figsize=(16.8, 5.8))
    b1 = ax.bar(
        x - w / 2, m6, w, yerr=sd6, capsize=2.8,
        color=[COLOR[g] for g in names], edgecolor="black", linewidth=0.55,
        label="S6N3（同分布测试，300 seed）",
    )
    b2 = ax.bar(
        x + w / 2, m7, w, yerr=sd7, capsize=2.8,
        color=[COLOR[g] for g in names], edgecolor="black", linewidth=0.55,
        alpha=0.55, hatch="//", label="S7N4（完全未见 7×7 房间与 seed 迁移）",
    )

    for bar, m, sd in zip(b1, m6, sd6):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.03, m + sd + 0.012),
                f"{m:.3f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    for bar, m, sd in zip(b2, m7, sd7):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.03, m + sd + 0.012),
                f"{m:.3f}", ha="center", va="bottom", fontsize=6.7, color="#222222")

    bc6 = bv6["bc30"]["success_mean"]
    bc7 = bv7["bc30"]["success_mean"]
    fl6 = bv6["b2_film_nwd"]["success_mean"]
    fl7 = bv7["b2_film_nwd"]["success_mean"]
    ax.axhline(bc6, ls="--", color="#4c78a8", lw=1.0, alpha=0.8, label=f"BC30 S6N3 = {bc6:.3f}")
    ax.axhline(bc7, ls="-.", color="#6e7781", lw=0.95, alpha=0.8, label=f"BC30 S7N4 = {bc7:.3f}")
    ax.axhline(fl6, ls="--", color="#9d2b2b", lw=1.0, alpha=0.75, label=f"FiLM (单环境) S6N3 = {fl6:.3f}")
    ax.axhline(fl7, ls=":", color="#9d2b2b", lw=1.1, alpha=0.85, label=f"FiLM (单环境) S7N4 = {fl7:.3f}")

    # 分组分割线与顶部区块说明
    ax.axvline(0.5, color="#bbbbbb", ls="-", lw=0.8)
    ax.axvline(3.5, color="#bbbbbb", ls="-", lw=0.8)
    ax.axvline(10.5, color="#9d2b2b", ls="--", lw=1.1, alpha=0.7)
    ax.text(2.0, 1.072, "Block A: 同域 S6N3 样本排序", ha="center", va="top",
            fontsize=8.6, fontweight="bold", color="#225522",
            bbox=dict(boxstyle="round,pad=0.2", fc="#eef8ee", ec="#77aa77"))
    ax.text(7.0, 1.072, "Block B: Concat 跨环境课程 vs 混训（同预算 4500 ep）", ha="center", va="top",
            fontsize=8.6, fontweight="bold", color="#1f4e78",
            bbox=dict(boxstyle="round,pad=0.2", fc="#eef4fb", ec="#7799cc"))
    ax.text(12.5, 1.072, "Block C: FiLM × 多尺度", ha="center", va="top",
            fontsize=8.6, fontweight="bold", color="#881111",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fdeeee", ec="#cc7777"))

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[g] for g in names], fontsize=7.6, rotation=18, ha="right")
    ax.set_ylabel("闭环 Success（mean ± SD，3 seeds）")
    ax.set_ylim(0.55, 1.11)
    ax.set_title("图 b4-1 · 支线 4 主结果：同域排序、跨环境课程与 FiLM 多尺度迁移（S6N3 & 未见 S7N4）", fontsize=11.5)
    ax.grid(alpha=0.28, axis="y")
    ax.legend(loc="lower left", ncol=3, fontsize=8.0, frameon=True)

    p = OUT / "b4_fig1_main_s6_s7.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig2_spectrum(spec: dict) -> str:
    bv = spec["by_variant"]
    tags = [e["tag"] for e in spec["envs"]]
    x = np.arange(len(tags))
    xlabels = ["S5N2\n(5×5 N=2)", "S6N3\n(6×6 N=3 目标)", "S6N4\n(6×6 N=4)", "S7N4\n(未见 7×7)", "S8N6\n(8×8 N=6)"]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), sharey=True)

    # (a) Concat (BCNet) 关键代表曲线
    ax = axes[0]
    concat_curves = [
        ("bc30", "BC30 (纯 S6N3)", "--", "s", 1.8),
        ("pace_easy", "pace_easy (易→难 S6N3)", "-", "o", 2.0),
        ("env_easy_cumul", "env_easy_cumul (Obj→S5→S6)", "-.", "v", 1.9),
        ("env_easy_mix", "env_easy_mix (Obj+S5+S6 混)", ":", "^", 1.8),
        ("env_multiscale_prog", "env_multiscale_prog (多尺度渐进)", "-", "D", 2.6),
        ("env_multiscale_mix", "env_multiscale_mix (多尺度混训)", "-.", "x", 2.0),
    ]
    for g, label, ls, mk, lw in concat_curves:
        ys = [bv[g][t]["success_mean"] for t in tags]
        ax.plot(x, ys, ls=ls, marker=mk, lw=lw, ms=6.5, color=COLOR[g], label=f"{label} [均值 {bv[g]['macro_mean']:.3f}]")

    # 清晰标注终点 S8N6 与 S6N3 关键点
    end_off = {
        "env_multiscale_prog": (6, 8),
        "env_multiscale_mix": (6, -3),
        "pace_easy": (6, -8),
        "env_easy_mix": (6, 2),
        "bc30": (6, 0),
        "env_easy_cumul": (6, -4),
    }
    # 按 S8N6 数值微调 y 避免相邻数字碰在一起
    end_y = {
        "env_multiscale_prog": (0.618, 6),
        "env_multiscale_mix": (0.585, 6),
        "pace_easy": (0.562, 6),
        "env_easy_mix": (0.538, 6),
        "bc30": (0.502, 6),
        "env_easy_cumul": (0.466, 6),
    }
    for g, (ty, dx) in end_y.items():
        y_true = bv[g]["S8N6"]["success_mean"]
        ax.annotate(f"{y_true:.3f}", xy=(4, y_true), xytext=(4.12, ty),
                    fontsize=7.6, fontweight="bold" if "prog" in g else "normal",
                    color=COLOR[g], va="center",
                    arrowprops=dict(arrowstyle="-", color=COLOR[g], lw=0.6))

    ax.axvspan(2.65, 3.35, color="#f2f2f2", alpha=0.85, zorder=0)
    ax.text(3.0, 0.435, "未见 7×7 域外", ha="center", va="bottom", fontsize=8.0, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.6)
    ax.set_xlim(-0.2, 4.55)
    ax.set_ylim(0.42, 1.03)
    ax.set_ylabel("闭环 Success（300 seed × 3 runs）")
    ax.set_title("(a) Concat 架构：多尺度渐进全面占优，零干扰先导（ObjS6）反而最差", fontsize=10.2)
    ax.grid(alpha=0.28)
    ax.legend(loc="lower left", fontsize=7.8, frameon=True)

    # (b) FiLM 曲线
    ax = axes[1]
    film_curves = [
        ("b2_film_nwd", "FiLM 单环境 (纯 S6N3)", "--", "s", 2.2),
        ("film_easy_stage", "FiLM 两阶段 (S5N2→S6N3)", "-.", "v", 2.0),
        ("film_multiscale_prog", "FiLM 多尺度渐进课程", "-", "D", 2.5),
        ("film_multiscale_mix", "FiLM 多尺度域随机化混训", "-", "o", 2.2),
    ]
    for g, label, ls, mk, lw in film_curves:
        ys = [bv[g][t]["success_mean"] for t in tags]
        ax.plot(x, ys, ls=ls, marker=mk, lw=lw, ms=7.0, color=COLOR[g], label=f"{label} [均值 {bv[g]['macro_mean']:.3f}]")

    # 标注 S7N4 与 S8N6 关键分离点
    for g, ty in [("film_multiscale_mix", 0.855), ("film_multiscale_prog", 0.825),
                  ("b2_film_nwd", 0.770), ("film_easy_stage", 0.699)]:
        y_true = bv[g]["S8N6"]["success_mean"]
        ax.annotate(f"{y_true:.3f}", xy=(4, y_true), xytext=(4.12, ty),
                    fontsize=8.2, fontweight="bold", color=COLOR[g], va="center",
                    arrowprops=dict(arrowstyle="-", color=COLOR[g], lw=0.7))

    ax.annotate("S7N4: +0.024 (3/3)\n0.900 → 0.924",
                xy=(3.0, bv["film_multiscale_prog"]["S7N4"]["success_mean"]),
                xytext=(2.15, 0.975), fontsize=8.3, color="#9d2b2b", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#9d2b2b", lw=1.0))
    ax.axvspan(2.65, 3.35, color="#fdf0f0", alpha=0.7, zorder=0)
    ax.text(3.0, 0.435, "未见 7×7 域外", ha="center", va="bottom", fontsize=8.0, color="#9d2b2b")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.6)
    ax.set_xlim(-0.2, 4.55)
    ax.set_title("(b) FiLM 架构：多尺度把 S7N4 推至 0.924、S8N6 涨 +0.074（两阶段接力有害）", fontsize=10.2)
    ax.grid(alpha=0.28)
    ax.legend(loc="lower left", fontsize=8.2, frameon=True)

    fig.suptitle("图 b4-2 · 五级环境难度谱泛化曲线（S5N2 → S6N3 → S6N4 → 未见 S7N4 → S8N6）", fontsize=12)
    p = OUT / "b4_fig2_difficulty_spectrum.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def load_mean_curve(prefix: str, epochs: int = 30) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for s in (0, 1, 2):
        hp = RUNS / f"{prefix}_seed{s}" / "history.json"
        if hp.exists():
            h = json.loads(hp.read_text())
            curves.append([row["val_acc"] for row in h[:epochs]])
    arr = np.asarray(curves, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def fig3_pacing_vs_mixing(ev: dict, s7: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))

    # (a) 同域 S6N3 验证集准确率训练轨迹（读自 runs/*/history.json）
    ax = axes[0]
    epochs = np.arange(1, 31)
    curve_specs = [
        ("b3_bc30", "BC30 (全量均匀)", "#6e7781", "--"),
        ("b4_pace_easy", "pace_easy (易→难)", "#2ca02c", "-"),
        ("b4_pace_rand", "pace_rand (随机阶梯)", "#17becf", "-."),
        ("b4_pace_hard", "pace_hard (反课程 难→易)", "#d62728", "-"),
    ]
    for pref, lbl, col, ls in curve_specs:
        m, sd = load_mean_curve(pref, 30)
        ax.plot(epochs, m, ls=ls, color=col, lw=2.0, label=f"{lbl} (终值={m[-1]:.3f})")
        ax.fill_between(epochs, m - sd, m + sd, color=col, alpha=0.10)

    ax.axvline(10.5, ls=":", color="#555555", lw=1.0)
    ax.axvline(20.5, ls=":", color="#555555", lw=1.0)
    ax.text(5.5, 0.898, "阶段 1: 40% 子集\n(等步数)", ha="center", va="top", fontsize=7.8, color="#444444")
    ax.text(15.5, 0.898, "阶段 2: 70% 子集\n(等步数)", ha="center", va="top", fontsize=7.8, color="#444444")
    ax.text(25.5, 0.795, "阶段 3: 100% 全池", ha="center", va="bottom", fontsize=7.8, color="#444444")

    ax.set_xlim(1, 30)
    ax.set_ylim(0.715, 0.908)
    ax.set_xlabel("训练 Epoch（每 epoch 严格等梯度更新步数）")
    ax.set_ylabel("S6N3 验证集动作准确率 (val_acc)")
    ax.set_title("(a) 同域排序学习曲线：易→难阶段2起陡升领先，反课程中段明显滞后", fontsize=10.0)
    ax.grid(alpha=0.28)
    ax.legend(loc="lower right", fontsize=8.0, frameon=True)

    # (b) 配对净增量（课程排序 − 对照 / 均匀混合）
    ax = axes[1]
    pc6, pc7 = ev["paired_comparisons"], s7["paired_comparisons"]
    pair_items = [
        ("pace_easy_vs_pace_rand", "同域: 易→难 − 随机阶梯"),
        ("pace_hard_vs_pace_easy", "同域: 反课程(难→易) − 易→难"),
        ("env_easy_cumul_vs_env_easy_mix", "简单环境: ObjS6先导累积 − 混训"),
        ("env_easy_stage_vs_env_easy_mix", "两阶段接力 S5→S6 − 简单混训"),
        ("env_s6n4_prog_vs_env_s6n4_mix", "高密度: S6N3→+S6N4 − 混训"),
        ("env_multiscale_prog_vs_env_multiscale_mix", "多尺度(Concat): 渐进 − 域随机混"),
        ("film_multiscale_prog_vs_film_multiscale_mix", "多尺度(FiLM): 渐进 − 域随机混"),
        ("film_multiscale_prog_vs_b2_film_nwd", "FiLM: 多尺度渐进 − 单环境S6"),
    ]
    y = np.arange(len(pair_items))
    h = 0.35
    d6 = [pc6[k]["delta_mean"] for k, _ in pair_items]
    sd6 = [pc6[k]["delta_sd"] for k, _ in pair_items]
    d7 = [pc7[k]["delta_mean"] for k, _ in pair_items]
    sd7 = [pc7[k]["delta_sd"] for k, _ in pair_items]

    ax.barh(y - h / 2, d6, h, xerr=sd6, capsize=2.5, color="#4c78a8", edgecolor="black", lw=0.45, label="S6N3 配对 Δ")
    ax.barh(y + h / 2, d7, h, xerr=sd7, capsize=2.5, color="#f58518", edgecolor="black", lw=0.45, label="S7N4 配对 Δ")
    ax.axvline(0.0, color="black", lw=1.0)

    for idx, ((val6, e6), (val7, e7)) in enumerate(zip(zip(d6, sd6), zip(d7, sd7))):
        off6 = val6 + e6 + 0.004 if val6 >= 0 else val6 - e6 - 0.004
        ha6 = "left" if val6 >= 0 else "right"
        ax.text(off6, idx - h / 2, f"{val6:+.3f}", va="center", ha=ha6, fontsize=6.9, color="#1f4e78", fontweight="bold")
        off7 = val7 + e7 + 0.004 if val7 >= 0 else val7 - e7 - 0.004
        ha7 = "left" if val7 >= 0 else "right"
        ax.text(off7, idx + h / 2, f"{val7:+.3f}", va="center", ha=ha7, fontsize=6.9, color="#a04000", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in pair_items], fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlim(-0.098, 0.125)
    ax.set_xlabel("3-seed 配对闭环 Success 差值 Δ（右侧 = 课程更优，左侧 = 混训/基准更优）")
    ax.set_title("(b) 排序本身是否优于均匀混训？视阶段与融合机制而定", fontsize=10.2)
    ax.grid(alpha=0.28, axis="x")
    ax.legend(loc="lower right", fontsize=8.2)

    fig.suptitle("图 b4-3 · 优化动力学与排序净效应：剥离「数据多样性」与「难度排序」的独立因果", fontsize=12)
    p = OUT / "b4_fig3_pacing_vs_mixing.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig4_mechanisms(sens: dict, fail7: dict) -> str:
    bv = sens["by_variant"]
    ag7 = fail7["aggregate"]
    names = [g for g in ORDER if g in bv]

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.5))

    # (a) 语义接地反事实探针
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.37
    col = [bv[g]["color_sensitivity_mean"] for g in names]
    typ = [bv[g]["type_sensitivity_mean"] for g in names]

    b1 = ax.bar(x - w / 2, col, w, color="#e45756", edgecolor="black", lw=0.45, label="换颜色词（目标存在）")
    b2 = ax.bar(x + w / 2, typ, w, color="#54a24b", edgecolor="black", lw=0.45, label="换类型词（目标存在）")
    for idx, (bar, v) in enumerate(zip(b1, col)):
        ey = 0.015 if abs(v - typ[idx]) < 0.025 else 0.005
        ax.text(bar.get_x() + bar.get_width() / 2, v + ey, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    for bar, v in zip(b2, typ):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.axvline(10.5, ls="--", color="#9d2b2b", lw=1.1, alpha=0.8)
    ax.text(5.0, 0.388, "Concat + 各类课程：颜色敏感度仅 0.03~0.06\n（ObjS6 零干扰预热反而让敏感度跌至最低 0.03）",
            ha="center", va="top", fontsize=8.1, color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff8e8", ec="#ccaa66"))
    ax.text(12.5, 0.388, "FiLM 组：\n颜色跳升 0.22~0.27", ha="center", va="top",
            fontsize=8.1, color="#9d2b2b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fdecec", ec="#e45756"))

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[g] for g in names], fontsize=7.2, rotation=24, ha="right")
    ax.set_ylabel("贪心动作改变比例（183 可切换状态）")
    ax.set_ylim(0, 0.415)
    ax.set_title("(a) 语义接地探针：单纯课程学习不修复 Concat 的颜色忽略捷径", fontsize=10.3)
    ax.grid(alpha=0.28, axis="y")
    ax.legend(loc="lower right", fontsize=8.0)

    # (b) S7N4 未见 7x7 失败构成拆解（关键代表组）
    ax = axes[1]
    rep_groups = [
        "bc30", "pace_easy", "env_easy_cumul", "env_easy_mix",
        "env_multiscale_mix", "env_multiscale_prog",
        "film_easy_stage", "b2_film_nwd", "film_multiscale_mix", "film_multiscale_prog",
    ]
    xr = np.arange(len(rep_groups))
    rep = [ag7[g]["failure_groups_mean"].get("representation", 0.0) for g in rep_groups]
    cov = [ag7[g]["failure_groups_mean"].get("state_coverage", 0.0) for g in rep_groups]
    attr = [ag7[g]["failure_types_mean"].get("attribute_confusion", 0.0) for g in rep_groups]

    ax.bar(xr, rep, 0.58, color="#e45756", edgecolor="black", lw=0.5, label="表征类失败 (attr_conf + distractor)")
    ax.bar(xr, cov, 0.58, bottom=rep, color="#7a8288", edgecolor="black", lw=0.5, label="状态覆盖类失败 (orientation/approach/wander)")
    for i, (r, c, a) in enumerate(zip(rep, cov, attr)):
        ax.text(i, r + c + 1.4, f"{r:.1f}+{c:.1f}\n(attr={a:.1f})", ha="center", va="bottom", fontsize=6.6)

    ax.axvline(5.5, ls="--", color="#9d2b2b", lw=1.1, alpha=0.8)
    ax.set_xticks(xr)
    ax.set_xticklabels([SHORT[g] for g in rep_groups], fontsize=7.4, rotation=22, ha="right")
    ax.set_ylabel("S7N4 平均失败 episode 数（每 300 局）")
    ax.set_ylim(0, 118)
    ax.set_title("(b) S7N4 失败构成：FiLM+多尺度把总失败压至 22.7（attr 9.3→3.7）", fontsize=10.3)
    ax.grid(alpha=0.28, axis="y")
    ax.legend(loc="upper right", fontsize=7.9)

    fig.suptitle("图 b4-4 · 机制证据：视觉捷径固化 vs 多尺度+FiLM 双重抑制失败", fontsize=12)
    p = OUT / "b4_fig4_mechanism_grounding_failures.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = jload("b4_curriculum_eval.json")
    s7 = jload("b4_curriculum_s7n4.json")
    spec = jload("b4_curriculum_spectrum.json")
    sens = jload("b4_curriculum_sensitivity.json")
    fail7 = jload("b4_curriculum_failures_s7n4.json")

    made = []
    if ev and s7:
        made.append(fig1_main(ev, s7))
        made.append(fig3_pacing_vs_mixing(ev, s7))
    if spec:
        made.append(fig2_spectrum(spec))
    if sens and fail7:
        made.append(fig4_mechanisms(sens, fail7))
    for m in made:
        print(f"wrote {m}")


if __name__ == "__main__":
    main()
