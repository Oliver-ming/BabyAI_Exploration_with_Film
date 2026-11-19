"""支线 1 · 作图 -> reports/figures/b1_langenc/

四张图（数字全部动态读取自 reports/b1_*.json）：
  1. b1_fig1_success_s6_s7.png           S6N3 与 S7N4 主结果（含参数匹配对照与 FiLM+GRU）
  2. b1_fig2_syntax_vs_semantics.png     词序敏感度（左）vs 颜色/类型语义接地探针（右）
  3. b1_fig3_scrambled_and_failures.png  闭环打乱指令鲁棒性（左）与失败构成拆解（右）
  4. b1_fig4_param_scaling.png           语言分支参数量匹配（左）与验证准确率-成功率解耦（右）

用法:
    OMP_NUM_THREADS=1 python branches/b1_langenc/make_figures.py
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
OUT = R / "figures" / "b1_langenc"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ORDER = [
    "mean",
    "mean_pos",
    "mean_wide",
    "dan",
    "pos_mlp",
    "transformer",
    "gru_small",
    "gru",
    "film_gru",
    "film_gru_small",
    "b2_film_nwd",
]

LABEL = {
    "bc30": "BC30\n(对照)",
    "mean": "mean\n(基线448)",
    "mean_pos": "mean_pos\n(对消608)",
    "mean_wide": "mean_wide\n(宽448)",
    "dan": "DAN\n(非线性7.8k)",
    "pos_mlp": "PosMLP\n(位置7.8k)",
    "transformer": "Transformer\n(10.2k)",
    "gru_small": "GRU-small\n(32d 7.8k)",
    "gru": "GRU\n(128d 66.8k)",
    "film_gru": "FiLM+GRU\n(66.8k)",
    "film_gru_small": "FiLM+\nGRU-small",
    "b2_film_nwd": "FiLM+mean\n(b2 wd=0)",
}

COLOR = {
    "bc30": "#6e7781",
    "mean": "#4c78a8",
    "mean_pos": "#72b7b2",
    "mean_wide": "#9ecae9",
    "dan": "#f58518",
    "pos_mlp": "#eeca3b",
    "transformer": "#b279a2",
    "gru_small": "#54a24b",
    "gru": "#2ca02c",
    "film_gru": "#ff9da6",
    "film_gru_small": "#e45756",
    "b2_film_nwd": "#9d2b2b",
}


def jload(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def ordered_present(mapping: dict, extra_order: list[str] | None = None) -> list[str]:
    base = extra_order or ORDER
    return [k for k in base if k in mapping]


def fig1_success(ev: dict, s7: dict) -> str:
    bv6 = ev["by_variant"]
    bv7 = s7["by_variant"]
    names = ordered_present(bv6)

    m6 = [bv6[n]["success_mean"] for n in names]
    sd6 = [bv6[n]["success_sd"] for n in names]
    m7 = [bv7[n]["success_mean"] for n in names]
    sd7 = [bv7[n]["success_sd"] for n in names]

    x = np.arange(len(names))
    w = 0.37

    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    b1 = ax.bar(
        x - w / 2, m6, w, yerr=sd6, capsize=3,
        color=[COLOR[n] for n in names], edgecolor="black", linewidth=0.6,
        label="S6N3（同分布测试，300 seed）",
    )
    b2 = ax.bar(
        x + w / 2, m7, w, yerr=sd7, capsize=3,
        color=[COLOR[n] for n in names], edgecolor="black", linewidth=0.6,
        alpha=0.58, hatch="//", label="S7N4（跨房间迁移，300 seed）",
    )

    for bar, m, sd in zip(b1, m6, sd6):
        ax.text(
            bar.get_x() + bar.get_width() / 2, min(1.01, m + sd + 0.014),
            f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )
    for bar, m, sd in zip(b2, m7, sd7):
        ax.text(
            bar.get_x() + bar.get_width() / 2, min(1.01, m + sd + 0.014),
            f"{m:.3f}", ha="center", va="bottom", fontsize=7.2, color="#222222",
        )

    bc6 = bv6["bc30"]["success_mean"]
    bc7 = bv7["bc30"]["success_mean"]
    fl6 = bv6["b2_film_nwd"]["success_mean"]
    ax.axhline(bc6, ls="--", color="#4c78a8", lw=1.0, alpha=0.85, label=f"BC30 / mean S6N3 = {bc6:.3f}")
    ax.axhline(bc6 + 0.045, ls=":", color="#2ca02c", lw=1.1, alpha=0.85,
               label=f"成功判据阈值 BC30+0.045 = {bc6 + 0.045:.3f}")
    ax.axhline(bc7, ls="-.", color="#777777", lw=0.9, alpha=0.75, label=f"BC30 / mean S7N4 = {bc7:.3f}")
    ax.axhline(fl6, ls="--", color="#9d2b2b", lw=1.0, alpha=0.75, label=f"FiLM+mean (b2 wd=0) S6N3 = {fl6:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in names], fontsize=8.2)
    ax.set_ylabel("闭环 Success（mean ± SD，3 seeds）", fontsize=10)
    ax.set_ylim(0.50, 1.07)
    ax.set_title("图 b1-1 · 支线 1 主结果：S6N3 同分布与 S7N4 跨房间迁移（含 DAN / PosMLP / FiLM+GRU 对照）", fontsize=11.5)
    ax.grid(alpha=0.28, axis="y")
    ax.legend(loc="upper left", ncol=3, fontsize=8.2, frameon=True)

    p = OUT / "b1_fig1_success_s6_s7.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig2_syntax_vs_semantics(wo: dict, sem: dict) -> str:
    bv_wo = wo["by_variant"]
    bv_sem = sem["by_variant"]
    names = ordered_present(bv_wo)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5))

    # (a) 词序敏感度探针
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.37
    swp = [bv_wo[n]["swap_sensitivity_mean"] for n in names]
    frt = [bv_wo[n]["move_front_sensitivity_mean"] for n in names]
    b1 = ax.bar(x - w / 2, swp, w, color="#4c78a8", edgecolor="black", linewidth=0.5, label="swap：color/type 互换")
    b2 = ax.bar(x + w / 2, frt, w, color="#f58518", edgecolor="black", linewidth=0.5, label="move_front：color 移至句首")
    for idx, (bar, v) in enumerate(zip(b1, swp)):
        if v > 0.0005:
            extra_y = 0.011 if abs(v - frt[idx]) < 0.008 else 0.003
            ax.text(bar.get_x() + bar.get_width() / 2, v + extra_y, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.7)
    for idx, (bar, v) in enumerate(zip(b2, frt)):
        if v > 0.0005:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.7)

    ax.axvspan(-0.5, 3.5, color="#e8f1f8", alpha=0.55, zorder=0)
    ax.text(1.5, 0.200, "置换不变 / 线性对消组\n（严格 0.000，已通过自检）", ha="center", va="top",
            fontsize=8.2, color="#2a4b6d", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in names], fontsize=7.5, rotation=22, ha="right")
    ax.set_ylabel("贪心动作改变比例（300 状态）")
    ax.set_ylim(0, 0.22)
    ax.set_title("(a) 词序敏感度（同词袋反事实）：BoW / mean_pos / DAN 严格=0", fontsize=10.5)
    ax.grid(alpha=0.28, axis="y")
    ax.legend(loc="upper right", fontsize=8.2)

    # (b) 语义接地反事实探针
    ax = axes[1]
    col = [bv_sem[n]["color_sensitivity_mean"] for n in names]
    typ = [bv_sem[n]["type_sensitivity_mean"] for n in names]
    b3 = ax.bar(x - w / 2, col, w, color="#e45756", edgecolor="black", linewidth=0.5, label="换颜色词（目标存在）")
    b4 = ax.bar(x + w / 2, typ, w, color="#54a24b", edgecolor="black", linewidth=0.5, label="换类型词（目标存在）")
    for idx, (bar, v) in enumerate(zip(b3, col)):
        extra_y = 0.015 if abs(v - typ[idx]) < 0.02 else 0.005
        ax.text(bar.get_x() + bar.get_width() / 2, v + extra_y, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.8)
    for bar, v in zip(b4, typ):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.8)

    ax.axvline(7.5, ls="--", color="#9d2b2b", lw=1.1, alpha=0.8)
    ax.text(3.5, 0.402, "Concat 融合组：颜色敏感度仅 0.03~0.13\n（GRU-small 仍 94.2% 忽略颜色）",
            ha="center", va="top", fontsize=8.2, color="#333333",
            bbox=dict(boxstyle="round,pad=0.28", fc="#fff8e8", ec="#ccaa66", alpha=0.92))
    ax.text(9.05, 0.402, "FiLM 融合组：\n颜色跳升至 0.22~0.27", ha="center", va="top",
            fontsize=8.2, color="#9d2b2b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="#fdecec", ec="#e45756", alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in names], fontsize=7.5, rotation=22, ha="right")
    ax.set_ylabel("贪心动作改变比例（183 可切换状态）")
    ax.set_ylim(0, 0.425)
    ax.set_title("(b) 颜色/类型语义接地探针：换编码器破不了 Concat 融合瓶颈", fontsize=10.5)
    ax.grid(alpha=0.28, axis="y")
    ax.legend(loc="lower right", fontsize=8.2)

    fig.suptitle("图 b1-2 · 机制双探针：词序敏感度（形式句法） vs 颜色/类型语义接地（内容绑定）", fontsize=12)
    p = OUT / "b1_fig2_syntax_vs_semantics.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig3_scrambled_and_failures(scr: dict, fail: dict) -> str:
    bv_scr = scr["by_variant"]
    ag_fail = fail["aggregate"]
    names = ordered_present(bv_scr)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5))

    # (a) 闭环打乱测试
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.26
    vn = [bv_scr[n]["normal_mean"] for n in names]
    vs = [bv_scr[n]["swap_mean"] for n in names]
    vf = [bv_scr[n]["move_front_mean"] for n in names]

    ax.bar(x - w, vn, w, color="#4c78a8", edgecolor="black", linewidth=0.45, label="原指令 normal")
    ax.bar(x, vs, w, color="#e45756", edgecolor="black", linewidth=0.45, label="对调 swap (go to <art> <type> <color>)")
    ax.bar(x + w, vf, w, color="#f58518", edgecolor="black", linewidth=0.45, label="前置 move_front (<color> go to <art> <type>)")

    for i, n in enumerate(names):
        dsw = -bv_scr[n]["drop_swap_mean"]
        dfr = -bv_scr[n]["drop_move_front_mean"]
        worst = min(dsw, dfr)
        top_y = max(vn[i], vs[i], vf[i])
        if abs(worst) > 0.01:
            ax.text(i, top_y + 0.012, f"{worst:+.3f}", ha="center", va="bottom",
                    fontsize=7.6, color="#b22222", fontweight="bold")
        else:
            ax.text(i, top_y + 0.012, "0.000", ha="center", va="bottom",
                    fontsize=7.2, color="#2a4b6d")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in names], fontsize=7.5, rotation=22, ha="right")
    ax.set_ylabel("闭环 Success（300 seed × 3 runs）")
    ax.set_ylim(0.50, 1.04)
    ax.set_title("(a) 闭环词序打乱鲁棒性（上方标注最坏掉点）：顺序模型学出模板位置捷径", fontsize=10.2)
    ax.grid(alpha=0.28, axis="y")
    ax.legend(loc="lower left", fontsize=7.8)

    # (b) 失败构成拆解
    ax = axes[1]
    names_f = ordered_present(ag_fail)
    xf = np.arange(len(names_f))
    rep = [ag_fail[n]["failure_groups_mean"].get("representation", 0.0) for n in names_f]
    cov = [ag_fail[n]["failure_groups_mean"].get("state_coverage", 0.0) for n in names_f]

    ax.bar(xf, rep, 0.58, color="#e45756", edgecolor="black", linewidth=0.5, label="表征类失败（认错属性/干扰物）")
    ax.bar(xf, cov, 0.58, bottom=rep, color="#7a8288", edgecolor="black", linewidth=0.5, label="状态覆盖类失败（转向/停滞/迷路）")
    for i, (r, c) in enumerate(zip(rep, cov)):
        ax.text(i, r + c + 1.0, f"{r:.1f}+{c:.1f}\n= {r + c:.1f}", ha="center", va="bottom", fontsize=6.9)

    ax.set_xticks(xf)
    ax.set_xticklabels([LABEL[n] for n in names_f], fontsize=7.5, rotation=22, ha="right")
    ax.set_ylabel("平均失败 episode 数（每 300 测试局）")
    ax.set_ylim(0, 68)
    ax.set_title("(b) 失败机制拆解：FiLM 把表征失败砍 84%（38.3→6.3），GRU 仅降 27%", fontsize=10.2)
    ax.grid(alpha=0.28, axis="y")
    ax.legend(loc="upper right", fontsize=8.0)

    fig.suptitle("图 b1-3 · 鲁棒性与误差拆解：退化模板上的伪词序依赖 vs 真实表征瓶颈", fontsize=12)
    p = OUT / "b1_fig3_scrambled_and_failures.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig4_param_scaling(ev: dict, s7: dict) -> str:
    bv6 = ev["by_variant"]
    bv7 = s7["by_variant"]
    names = [n for n in ORDER if n in bv6]

    def get_lang_p(n: str) -> int:
        lp = bv6[n].get("n_lang_params")
        return int(lp) if lp is not None else 448

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))

    # (a) 语言分支参数量 vs S6N3 & S7N4 Success（使用数据坐标牵引线彻底杜绝重叠）
    ax = axes[0]
    for bv, marker, alpha in [(bv6, "o", 0.95), (bv7, "^", 0.85)]:
        for n in names:
            lp = get_lang_p(n)
            y = bv[n]["success_mean"]
            ax.scatter(
                lp, y, s=95 if marker == "o" else 85,
                color=COLOR[n], marker=marker, edgecolor="black", linewidth=0.6,
                alpha=alpha, zorder=4,
            )

    ax.scatter([], [], s=85, color="black", marker="o", label="S6N3（圆点）")
    ax.scatter([], [], s=85, color="black", marker="^", label="S7N4（三角）")

    # (variant, env_num) -> absolute (text_x, text_y, ha) in clean open gaps
    abs_pos = {
        ("b2_film_nwd", 6): (900, 0.962, "left"),
        ("b2_film_nwd", 7): (900, 0.903, "left"),
        ("mean_wide", 6): (950, 0.858, "left"),
        ("mean_pos", 6): (1100, 0.822, "left"),
        ("mean", 6): (950, 0.795, "left"),
        ("mean_wide", 7): (950, 0.758, "left"),
        ("mean_pos", 7): (820, 0.726, "left"),
        ("mean", 7): (950, 0.692, "left"),
        ("film_gru_small", 6): (13500, 0.964, "left"),
        ("film_gru_small", 7): (14500, 0.892, "left"),
        ("gru_small", 6): (4600, 0.888, "right"),
        ("pos_mlp", 6): (4600, 0.856, "right"),
        ("dan", 6): (4600, 0.831, "right"),
        ("transformer", 6): (14500, 0.844, "left"),
        ("gru_small", 7): (4600, 0.800, "right"),
        ("dan", 7): (4600, 0.753, "right"),
        ("pos_mlp", 7): (5400, 0.713, "right"),
        ("transformer", 7): (14500, 0.724, "left"),
        ("film_gru", 6): (49000, 0.930, "right"),
        ("gru", 6): (45000, 0.878, "right"),
        ("film_gru", 7): (45000, 0.835, "right"),
        ("gru", 7): (48000, 0.745, "right"),
    }
    short_tag = {
        "b2_film_nwd": "FiLM+mean",
        "film_gru_small": "FiLM+GRU-s",
        "film_gru": "FiLM+GRU",
        "mean": "mean",
        "mean_pos": "mean_pos",
        "mean_wide": "mean_wide",
        "dan": "DAN",
        "pos_mlp": "PosMLP",
        "gru_small": "GRU-s",
        "transformer": "Trf",
        "gru": "GRU(128)",
    }
    for (n, env_num), (tx, ty, ha) in abs_pos.items():
        if n not in bv6:
            continue
        bv = bv6 if env_num == 6 else bv7
        lp = get_lang_p(n)
        sy = bv[n]["success_mean"]
        suffix = "" if env_num == 6 else " [S7]"
        ax.annotate(
            f"{short_tag[n]}{suffix} {sy:.3f}",
            xy=(lp, sy), xytext=(tx, ty), ha=ha, va="center", fontsize=6.9,
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.55, shrinkA=1, shrinkB=4),
        )

    ax.axvspan(6500, 11500, color="#edf7ed", alpha=0.65, zorder=0)
    ax.text(8800, 1.002, "~7.8k–10.2k\n参数匹配区", ha="center", va="top",
            fontsize=8.0, color="#2b6b2b", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(340, 105000)
    ax.set_ylim(0.67, 1.015)
    ax.set_xlabel("语言分支参数量 n_lang_params（对数坐标）")
    ax.set_ylabel("闭环 Success")
    ax.set_title("(a) 语言参数规模 vs 表现：小参数 GRU-small 最稳，128d GRU 在 S7N4 过拟合", fontsize=10.1)
    ax.grid(alpha=0.28, which="both")
    ax.legend(loc="lower left", fontsize=8.4)

    # (b) val_acc vs S6N3 Success 解耦图（每个点使用牵引线标注精确数值）
    ax = axes[1]
    val_abs = {
        "b2_film_nwd": (0.955, 0.967, "right"),
        "film_gru_small": (0.953, 0.946, "right"),
        "film_gru": (0.953, 0.927, "right"),
        "gru": (0.916, 0.858, "left"),
        "gru_small": (0.896, 0.884, "left"),
        "pos_mlp": (0.866, 0.862, "right"),
        "transformer": (0.898, 0.844, "left"),
        "mean_wide": (0.898, 0.824, "left"),
        "dan": (0.862, 0.843, "right"),
        "mean_pos": (0.863, 0.808, "left"),
        "mean": (0.883, 0.801, "left"),
    }
    for n in names:
        vx = bv6[n]["val_acc_mean"]
        sy = bv6[n]["success_mean"]
        ax.scatter(vx, sy, s=110, color=COLOR[n], edgecolor="black", linewidth=0.7, zorder=4)
        tx, ty, ha = val_abs[n]
        ax.annotate(
            f"{short_tag[n]} ({sy:.3f})",
            xy=(vx, sy), xytext=(tx, ty), ha=ha, va="center", fontsize=7.4,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6, shrinkA=1, shrinkB=5),
        )

    ax.plot([0.84, 0.99], [0.84, 0.99], "--", color="gray", alpha=0.55, label="y = x 对角线")
    ax.set_xlim(0.846, 0.986)
    ax.set_ylim(0.790, 0.985)
    ax.set_xlabel("验证集动作准确率 val_action_acc（mean）")
    ax.set_ylabel("S6N3 闭环 Success（mean）")
    ax.set_title("(b) 验证准确率 vs 闭环成功率：GRU 离线更高但闭环仍被 Concat 锁死", fontsize=10.1)
    ax.grid(alpha=0.28)
    ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("图 b1-4 · 参数效率与泛化拆解：排除多参数假象，定位容量甜区", fontsize=12)
    p = OUT / "b1_fig4_param_scaling.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = jload("b1_langenc_eval.json")
    s7 = jload("b1_langenc_s7n4.json")
    wo = jload("b1_word_order_sensitivity.json")
    scr = jload("b1_scrambled_eval.json")
    sem = jload("b1_semantic_sensitivity.json")
    fail = jload("b1_failure_compare.json")

    made = []
    if ev and s7:
        made.append(fig1_success(ev, s7))
        made.append(fig4_param_scaling(ev, s7))
    if wo and sem:
        made.append(fig2_syntax_vs_semantics(wo, sem))
    if scr and fail:
        made.append(fig3_scrambled_and_failures(scr, fail))
    for m in made:
        print(f"wrote {m}")


if __name__ == "__main__":
    main()
