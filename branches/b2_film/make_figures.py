"""支线 2 · 作图 -> reports/figures/b2_film/

三张图：
  1. 主结果条形图（各组 Success，含参数量匹配对照）
  2. 机制图：语言敏感度 + 失败构成 + 准确率vs成功率
  3. weight-decay 对照图（§5.1 那个反直觉结论）

用法:
    python branches/b2_film/make_figures.py
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
OUT = R / "figures" / "b2_film"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SHORT = {
    "bc30": "BC30\n(对照)",
    "concat_nwd": "concat\n(wd=0)",
    "concat_wd": "concat\n(wd=1e-4)",
    "concat_wide_wd": "concat_wide\n(参数匹配)",
    "film_nwd": "FiLM\n(wd=0)",
    "film_wd": "FiLM\n(wd=1e-4)",
    "film_only_wd": "FiLM-only\n(不拼接)",
    "film_res_wd": "FiLM+res",
}
COLOR = {
    "bc30": "tab:gray",
    "concat_nwd": "tab:blue",
    "concat_wd": "tab:cyan",
    "concat_wide_wd": "tab:olive",
    "film_nwd": "tab:orange",
    "film_wd": "tab:red",
    "film_only_wd": "tab:purple",
    "film_res_wd": "tab:brown",
}


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fig_success(ev: dict) -> str:
    agg = ev["aggregate"]
    names = [n for n in SHORT if n in agg]
    means = [agg[n]["closed_loop_success"]["mean"] for n in names]
    sds = [agg[n]["closed_loop_success"]["sd"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(range(len(names)), means, yerr=sds, capsize=4,
                  color=[COLOR[n] for n in names], edgecolor="black", linewidth=0.6)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}",
                ha="center", fontsize=8, fontweight="bold")
    base = agg["bc30"]["closed_loop_success"]["mean"]
    ax.axhline(base, ls="--", color="gray", lw=1, label=f"BC30 baseline {base:.3f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SHORT[n] for n in names], fontsize=8)
    ax.set_ylabel("闭环 Success（300 个未见 seed）")
    ax.set_ylim(0, 1.05)
    ax.set_title("图 b2-1 · FiLM 跨模态融合：主结果（3 seed，mean ± SD）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8)
    p = OUT / "b2_fig1_success.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig_mechanism(sens: dict, fail: dict, ev: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) 语言敏感度
    ax = axes[0]
    bv = sens["by_variant"]
    x = np.arange(2)
    w = 0.36
    cvals = [bv.get("concat", {}).get("color_sensitivity_mean", 0),
             bv.get("film", {}).get("color_sensitivity_mean", 0)]
    tvals = [bv.get("concat", {}).get("type_sensitivity_mean", 0),
             bv.get("film", {}).get("type_sensitivity_mean", 0)]
    ax.bar(x - w / 2, cvals, w, label="换颜色", color="tab:blue")
    ax.bar(x + w / 2, tvals, w, label="换类型", color="tab:orange")
    for i, (c, t) in enumerate(zip(cvals, tvals)):
        ax.text(i - w / 2, c + 0.01, f"{c:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, t + 0.01, f"{t:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["concat (基线)", "FiLM"])
    ax.set_ylabel("动作改变比率")
    ax.set_title("(a) 语言敏感度：改指令后动作是否变\n越高 = 越真的在读语言")
    ax.set_ylim(0, 0.45)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.annotate(
        "基线 93.5% 情况下\n改颜色不改动作",
        xy=(0 - w / 2, cvals[0]), xytext=(0.15, 0.32),
        fontsize=7.5, color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
    )

    # (b) 失败构成
    ax = axes[1]
    ag = fail["aggregate"]
    label_map = {"bc30": "BC30", "film": "FiLM", "film_only": "FiLM-only", "film_res": "FiLM+res"}
    names = [n for n in label_map if n in ag]
    rep = [ag[n]["failure_groups_mean"].get("representation", 0) for n in names]
    cov = [ag[n]["failure_groups_mean"].get("state_coverage", 0) for n in names]
    x = np.arange(len(names))
    ax.bar(x, rep, 0.55, label="表征类（认错属性/干扰物）", color="tab:red")
    ax.bar(x, cov, 0.55, bottom=rep, label="状态覆盖类（转圈/顶墙）", color="tab:gray")
    for i, (r, c) in enumerate(zip(rep, cov)):
        ax.text(i, r + c + 1, f"{r:.1f}+{c:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[n] for n in names])
    ax.set_ylabel("平均失败 episode 数（每 300 测试 seed）")
    ax.set_title("(b) 失败构成：表征类失败砍掉 78%")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7.5)

    # (c) 准确率 vs 成功率
    ax = axes[2]
    for n in SHORT:
        if n not in ev["aggregate"]:
            continue
        a = ev["aggregate"][n]
        ax.scatter(a["val_action_acc"]["mean"], a["closed_loop_success"]["mean"],
                   s=110, color=COLOR[n], edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(n.replace("_wd", "").replace("_nwd", "\n(wd=0)"),
                    (a["val_action_acc"]["mean"], a["closed_loop_success"]["mean"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=6.5)
    ax.plot([0.7, 1.0], [0.7, 1.0], "--", color="gray", alpha=0.6)
    ax.set_xlabel("验证集动作准确率")
    ax.set_ylabel("闭环 Success")
    ax.set_title("(c) 准确率 vs 成功率\nFiLM 把两者一起抬起来")
    ax.set_xlim(0.78, 1.0)
    ax.set_ylim(0.65, 1.0)
    ax.grid(alpha=0.3)

    fig.suptitle("图 b2-2 · 机制证据：FiLM 的收益来自真实语言接地，不是容量", fontsize=11)
    p = OUT / "b2_fig2_mechanism.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig_weight_decay(ev: dict) -> str:
    agg = ev["aggregate"]
    pairs = [("concat_nwd", "concat_wd", "concat（基线架构）"),
             ("film_nwd", "film_wd", "FiLM")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(pairs))
    w = 0.35
    v0 = [agg[a]["closed_loop_success"]["mean"] for a, _b, _l in pairs]
    v1 = [agg[b]["closed_loop_success"]["mean"] for _a, b, _l in pairs]
    s0 = [agg[a]["closed_loop_success"]["sd"] for a, _b, _l in pairs]
    s1 = [agg[b]["closed_loop_success"]["sd"] for _a, b, _l in pairs]
    ax.bar(x - w / 2, v0, w, yerr=s0, capsize=4, label="wd = 0", color="tab:green")
    ax.bar(x + w / 2, v1, w, yerr=s1, capsize=4, label="wd = 1e-4", color="tab:red")
    for i, (a0, a1) in enumerate(zip(v0, v1)):
        ax.text(i - w / 2, a0 + 0.02, f"{a0:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, a1 + 0.02, f"{a1:.3f}", ha="center", fontsize=8)
        ax.text(i, min(a0, a1) - 0.07, f"Δ={a1 - a0:+.3f}", ha="center",
                fontsize=9, color="crimson", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _a, _b, l in pairs])
    ax.set_ylabel("闭环 Success")
    ax.set_ylim(0, 1.05)
    ax.set_title("图 b2-3 · weight decay 在本任务上是有害的（与文献 2 相反）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    p = OUT / "b2_fig3_weight_decay.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = jload(R / "b2_film_eval.json")
    sens = jload(R / "b2_language_sensitivity.json")
    fail = jload(R / "b2_failure_compare.json")

    made = {}
    if ev:
        made["fig1"] = fig_success(ev)
    if ev and sens and fail:
        made["fig2"] = fig_mechanism(sens, fail, ev)
    if ev:
        made["fig3"] = fig_weight_decay(ev)
    for k, v in made.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
