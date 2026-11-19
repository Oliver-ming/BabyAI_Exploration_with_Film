"""支线 3 (v2) · 生成发表级对比与机制图表 -> reports/figures/b3_dagger/

四张图：
  1. b3_fig1_ablation_progression.png : 主结果消融阶梯图（S6N3，mean ± SD）
  2. b3_fig2_query_efficiency_pareto.png : 主动查询效率 Pareto 曲线（Success vs. 总专家标注步数）
  3. b3_fig3_coverage_and_failures.png : 新颖状态/纠偏覆盖效率 + 失败机理分解（状态覆盖 vs 表征）
  4. b3_fig4_s7n4_transfer.png : S6N3 同分布 vs S7N4 零样本跨房间迁移对比

用法:
    python branches/b3_dagger/make_figures.py
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
OUT = R / "figures" / "b3_dagger"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fig1_progression(s6: dict) -> str:
    bg = s6["by_group"]
    items = [
        ("baseline", "基线\n(早停p=5)", "#9e9e9e"),
        ("b3_bc30", "BC30\n(4500专家)", "#607d8b"),
        ("b3_v2_bc3000", "BC-3000\n(等预算起点)", "#90a4ae"),
        ("b3_v2_split_dag", "Split-DAgger\n(3k+1.5k=4500)", "#26a69a"),
        ("b3_v2_split_ens", "Split-Ens\n(等预算+集成)", "#00897b"),
        ("b3_v2_bcfresh", "BC-Fresh\n(+1500新专家)", "#78909c"),
        ("b3_dagger", "v1 DAgger\n(记忆种子)", "#64b5f6"),
        ("b3_v2_greedy", "v2 Greedy\n(新种子)", "#42a5f5"),
        ("b3_v2_unstick", "v2 Unstick\n(死锁解锁)", "#1e88e5"),
        ("b3_v2_iwr2", "v2 IWR×2\n(解锁+加权)", "#1565c0"),
        ("b3_v2_r2_w1", "v2 R2 (w=1)\n(两轮非重叠)", "#0d47a1"),
        ("b2_film_nwd", "B2 FiLM\n(4500专家)", "#fb8c00"),
        ("b3_v2_film_dag_w1", "FiLM+DAgger\n(w=1)", "#ef6c00"),
        ("b3_v2_film_dag", "FiLM+DAgger\n(w=2)", "#e65100"),
    ]
    items = [(k, lbl, c) for k, lbl, c in items if k in bg]

    means = [bg[k]["success_mean"] for k, _, _ in items]
    sds = [bg[k]["success_sd"] for k, _, _ in items]
    colors = [c for _, _, c in items]
    labels = [lbl for _, lbl, _ in items]

    fig, ax = plt.subplots(figsize=(15.2, 5.4))
    x = np.arange(len(items))
    bars = ax.bar(x, means, yerr=sds, capsize=3.5, color=colors, edgecolor="black", linewidth=0.6)
    for b, m, sd in zip(bars, means, sds):
        ax.text(b.get_x() + b.get_width() / 2, m + sd + 0.008, f"{m:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    if "b3_bc30" in bg:
        bc30_m = bg["b3_bc30"]["success_mean"]
        ax.axhline(bc30_m, ls="--", color="#455a64", lw=1.1, label=f"BC30 真基线 ({bc30_m:.3f})")
    if "b2_film_nwd" in bg:
        film_m = bg["b2_film_nwd"]["success_mean"]
        ax.axhline(film_m, ls=":", color="#e65100", lw=1.1, label=f"B2 FiLM 基线 ({film_m:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.3)
    ax.set_ylabel("闭环 Success（S6N3，300 未见测试 seed）")
    ax.set_ylim(0.55, 1.04)
    ax.set_title("图 b3-1 · 支线 3 (v2) 消融演进：从等预算 Split-DAgger 到死锁解锁 + IWR 与 FiLM 正交叠加（3 seed，mean ± SD）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8.5)

    p = OUT / "b3_fig1_ablation_progression.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig2_query_pareto(s6: dict) -> str:
    bg = s6["by_group"]
    bc30_runs = [r for r in s6.get("runs", []) if r.get("group") == "b3_bc30"]
    bc30_labels = float(bc30_runs[0]["n_train_transitions"]) if bc30_runs and bc30_runs[0].get("n_train_transitions") else None
    if bc30_labels is None:
        cfg_bc30 = jload(ROOT / "runs" / "b3_bc30_seed0" / "config.json")
        bc30_labels = float(cfg_bc30["data"]["n_train_transitions"])

    # (group, label, color, marker, (tx, ty), ha)
    points = [
        ("b3_v2_bc3000", "BC-3000", "#78909c", "s", (11800, 0.742), "left"),
        ("b3_bc30", "BC30 (4500专家)", "#455a64", "s", (17100, 0.804), "left"),
        ("b3_v2_bcfresh", "BC-Fresh (+1500新专家)", "#546e7a", "s", (22100, 0.880), "center"),
        ("b3_v2_split_ens", "Split-EnsembleDAgger (4500预算)", "#00897b", "*", (10600, 0.903), "left"),
        ("b3_v2_split_dag", "Split-DAgger (4500预算)", "#26a69a", "^", (20600, 0.785), "center"),
        ("b3_v2_qens", "v2 Ensemble门控", "#2e7d32", "*", (16500, 0.822), "left"),
        ("b3_v2_qmar", "v2 Margin门控", "#43a047", "D", (16000, 0.892), "left"),
        ("b3_v2_qdis", "v2 Disagreement过滤", "#66bb6a", "o", (25700, 0.825), "left"),
        ("b3_v2_greedy", "v2 Greedy全查", "#42a5f5", "o", (31300, 0.840), "center"),
        ("b3_v2_unstick", "v2 Unstick全查 (w=1)", "#1e88e5", "o", (25700, 0.916), "left"),
        ("b3_v2_iwr2", "v2 Unstick+IWR×2", "#1565c0", "o", (25700, 0.862), "left"),
        ("b3_v2_r2_w1", "v2 R2两轮 (w=1)", "#0d47a1", "o", (32000, 0.927), "right"),
    ]

    fig, ax = plt.subplots(figsize=(11.4, 5.9))
    xs_bc, ys_bc = [], []
    pareto_candidates = []

    for k, lbl, color, marker, xy_text, ha in points:
        if k not in bg:
            continue
        g = bg[k]
        q = g.get("total_oracle_labels_mean")
        if q is None and k == "b3_bc30":
            q = bc30_labels
        if q is None:
            continue
        y = g["success_mean"]
        ysd = g["success_sd"]
        ms = 160 if marker == "*" else 85
        ax.errorbar(q, y, yerr=ysd, fmt="none", ecolor=color, alpha=0.4, capsize=3, zorder=2)
        ax.scatter(q, y, s=ms, color=color, marker=marker, edgecolor="black", linewidth=0.7, zorder=4)
        ax.annotate(
            f"{lbl} ({q/1000:.1f}k, {y:.3f})",
            xy=(q, y),
            xytext=xy_text,
            textcoords="data",
            ha=ha,
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.7, shrinkA=2, shrinkB=5),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.9, edgecolor="#dddddd", linewidth=0.5),
            zorder=5,
        )
        if k in ("b3_v2_bc3000", "b3_bc30", "b3_v2_bcfresh"):
            xs_bc.append(q); ys_bc.append(y)
        if k in ("b3_v2_split_ens", "b3_v2_qmar", "b3_v2_unstick", "b3_v2_greedy", "b3_v2_r2_w1"):
            pareto_candidates.append((q, y))

    if len(xs_bc) >= 2:
        order = np.argsort(xs_bc)
        ax.plot(np.array(xs_bc)[order], np.array(ys_bc)[order], "--", color="#607d8b", lw=1.3, label="纯被动 BC 缩放曲线")
    if pareto_candidates:
        pareto_candidates.sort(key=lambda t: t[0])
        frontier = []
        best_y = -1.0
        for q, y in pareto_candidates:
            if y > best_y:
                frontier.append((q, y))
                best_y = y
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], "-", color="#1b5e20", lw=1.8, alpha=0.8, label="主动交互 Pareto 前沿")

    ax.set_xlim(10000, 34500)
    ax.set_ylim(0.725, 0.945)
    ax.set_xlabel("总专家动作标注数（总 Oracle 查询步数：基础专家 + 交互查询）")
    ax.set_ylabel("闭环 Success（S6N3，3 seed 均值 ± SD）")
    ax.set_title("图 b3-2 · 查询效率 Pareto 前沿：等标注预算下主动门控交互显著胜出纯 BC")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

    p = OUT / "b3_fig2_query_efficiency_pareto.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig3_coverage_and_failures(cov: dict | None, s6: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

    ax = axes[0]
    if cov and cov.get("datasets"):
        ds_map = {d["key"]: d for d in cov["datasets"]}
        order = [
            ("v1_expert_extra_dup", "v1 额外专家\n(重复seed)"),
            ("v2_fresh_expert", "v2 新专家\n(5300:6800)"),
            ("v1_dagger_memorized", "v1 DAgger\n(记忆seed)"),
            ("v2_greedy_fresh", "v2 Greedy\n(新seed)"),
            ("v2_beta025_fresh", "v2 β=0.25\n(新seed)"),
            ("v2_unstick_fresh", "v2 解锁\n(新seed)"),
            ("v2_qens_fresh", "v2 Ensemble\n门控(新seed)"),
        ]
        order = [(k, lbl) for k, lbl in order if k in ds_map]
        x = np.arange(len(order))
        w = 0.36
        novels = [ds_map[k]["novel_idm_vs_expert_train"] for k, _ in order]
        nov_dis = [ds_map[k]["novel_disagree_idm_states"] for k, _ in order]
        ax.bar(x - w / 2, novels, w, label="新颖 (img,dir,mission) 状态数", color="#42a5f5")
        ax.bar(x + w / 2, nov_dis, w, label="新颖且 π(s)≠a* 纠偏状态数", color="#ef5350")
        for i, (n1, n2) in enumerate(zip(novels, nov_dis)):
            ax.text(i - w / 2, n1 + 90, f"{n1}", ha="center", fontsize=7)
            ax.text(i + w / 2, n2 + 90, f"{n2}", ha="center", fontsize=7, color="#b71c1c")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in order], fontsize=7.5)
        ax.set_ylim(0, 7700)
        ax.set_ylabel("唯一状态数（1500 条采集轨迹）")
        ax.set_title("(a) 采样分布对比：修正重复种子与记忆盲区后新颖纠偏态翻倍")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    ax = axes[1]
    bg = s6["by_group"]
    f_items = [
        ("b3_bc30", "BC30"),
        ("b3_dagger", "v1 DAgger"),
        ("b3_v2_unstick", "v2 Unstick"),
        ("b3_v2_iwr2", "v2 IWR×2"),
        ("b3_v2_r2_w1", "v2 R2(w=1)"),
        ("b2_film_nwd", "B2 FiLM"),
        ("b3_v2_film_dag", "FiLM+DAgger"),
    ]
    f_items = [(k, lbl) for k, lbl in f_items if k in bg and "failure_groups_mean" in bg[k]]
    x = np.arange(len(f_items))
    rep = [bg[k]["failure_groups_mean"].get("representation", 0.0) for k, _ in f_items]
    sc = [bg[k]["failure_groups_mean"].get("state_coverage", 0.0) for k, _ in f_items]
    ax.bar(x, rep, 0.56, label="表征类失败（属性混淆 / 干扰物）", color="#e53935")
    ax.bar(x, sc, 0.56, bottom=rep, label="状态覆盖类失败（转圈 / 顶墙 / 卡死）", color="#546e7a")
    for i, (r, s) in enumerate(zip(rep, sc)):
        ax.text(i, r + s + 1.1, f"{r:.1f}+{s:.1f}\n={r+s:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in f_items], fontsize=7.5)
    ax.set_ylim(0, 66)
    ax.set_ylabel("平均失败 episode 数（每 300 测试 seed）")
    ax.set_title("(b) 失败机理分解：DAgger 与 FiLM 在机制上正交互补")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("图 b3-3 · 支线 3 (v2) 机制验证：状态覆盖密度提升与失败构成正交削减", fontsize=11)
    p = OUT / "b3_fig3_coverage_and_failures.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig4_transfer(s6: dict, s7: dict) -> str:
    g6, g7 = s6["by_group"], s7["by_group"]
    items = [
        ("b3_bc30", "BC30"),
        ("b3_v2_bc3000", "BC-3000"),
        ("b3_v2_split_dag", "Split-DAgger"),
        ("b3_v2_bcfresh", "BC-Fresh"),
        ("b3_dagger", "v1 DAgger"),
        ("b3_v2_unstick", "v2 Unstick"),
        ("b3_v2_r2_w1", "v2 R2(w=1)"),
        ("b2_film_nwd", "B2 FiLM"),
        ("b3_v2_film_dag_w1", "FiLM+DAgger\n(w=1)"),
    ]
    items = [(k, lbl) for k, lbl in items if k in g6 and k in g7]
    x = np.arange(len(items))
    w = 0.36
    v6 = [g6[k]["success_mean"] for k, _ in items]
    v7 = [g7[k]["success_mean"] for k, _ in items]
    e6 = [g6[k]["success_sd"] for k, _ in items]
    e7 = [g7[k]["success_sd"] for k, _ in items]

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.bar(x - w / 2, v6, w, yerr=e6, capsize=3, label="S6N3（同分布测试 4500:4800）", color="#1e88e5")
    ax.bar(x + w / 2, v7, w, yerr=e7, capsize=3, label="S7N4（零样本迁移 10000:10300）", color="#fb8c00")
    for i, (a, ea, b, eb) in enumerate(zip(v6, e6, v7, e7)):
        ax.text(i - w / 2, a + ea + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, b + eb + 0.008, f"{b:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in items], fontsize=8)
    ax.set_ylabel("闭环 Success（3 seed，mean ± SD）")
    ax.set_ylim(0.50, 1.04)
    ax.set_title("图 b3-4 · S6N3 同分布 vs S7N4 跨房间零样本迁移对比")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8.5)

    p = OUT / "b3_fig4_s7n4_transfer.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s6 = jload(R / "b3_v2_eval_s6n3.json")
    s7 = jload(R / "b3_v2_eval_s7n4.json")
    cov = jload(R / "b3_v2_rollout_coverage.json")

    made = {}
    if s6:
        made["fig1"] = fig1_progression(s6)
        made["fig2"] = fig2_query_pareto(s6)
        made["fig3"] = fig3_coverage_and_failures(cov, s6)
    if s6 and s7:
        made["fig4"] = fig4_transfer(s6, s7)
    for k, v in made.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
