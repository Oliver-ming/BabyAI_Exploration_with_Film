"""P4-1 · 汇总表 + 作图（§11.1，本阶段只做基线相关的图 1/2/4/6/7）。

用法:
    python src/make_figures.py --reports reports --out reports/figures

产出：
    fig1_loss_curves.png        训练/验证损失曲线（3 seed，mean ± SD）
    fig2_acc_vs_success.png     动作准确率 vs 闭环 Success —— 本赛道的「灵魂图」
    fig4_mission_heatmap.png    mission 类型分解热图（颜色 × 类型 × 冠词）
    fig6_s6n3_vs_s7n4.png       S6N3 vs S7N4 迁移对比
    fig7_trajectories.png       成功/失败轨迹拼图
    summary_table.md            所有关键数字的汇总表

图 3 / 图 5 依赖 DAgger 与数据效率消融（P3 阶段），本阶段不产出。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COLORS, TYPES  # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fig1_loss_curves(reports: Path, out: Path) -> str | None:
    runs = sorted(Path("runs").glob("baseline_seed*/history.json")) or sorted(
        reports.glob("baseline_seed*/history.json")
    )
    if not runs:
        return None
    curves = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for r in runs:
        h = json.loads(r.read_text())
        for k in curves:
            curves[k].append([e[k] for e in h])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (a, b), title in (
        (axes[0], ("train_loss", "val_loss"), "Loss"),
        (axes[1], ("train_acc", "val_acc"), "Action accuracy"),
    ):
        for key, color, label in ((a, "tab:blue", "train"), (b, "tab:orange", "val")):
            arrs = curves[key]
            n = max(len(x) for x in arrs)
            padded = np.full((len(arrs), n), np.nan)
            for i, x in enumerate(arrs):
                padded[i, : len(x)] = x
            mean = np.nanmean(padded, axis=0)
            sd = np.nanstd(padded, axis=0)
            ep = np.arange(1, n + 1)
            ax.plot(ep, mean, color=color, label=f"{label} (mean of {len(arrs)} seeds)")
            ax.fill_between(ep, mean - sd, mean + sd, color=color, alpha=0.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title}（3 seed，mean ± SD）")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("图 1 · BC 基线的训练/验证曲线（BabyAI-GoToLocalS6N3）")
    p = out / "fig1_loss_curves.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def _train_action_dist() -> dict[int, float]:
    """从专家数据算训练集动作分布，用于标注平凡策略的 teacher-forcing 准确率。"""
    import pickle

    p = Path("data/expert_5000.pkl")
    if not p.exists():
        return {2: 0.563, 0: 0.221, 1: 0.216}
    eps = pickle.load(open(p, "rb"))
    c = np.zeros(7)
    for e in eps:
        if e.get("split", "train") == "train":
            for a in e["actions"]:
                c[a] += 1
    return {i: float(v / c.sum()) for i, v in enumerate(c)}


def fig2_acc_vs_success(reports: Path, out: Path) -> str | None:
    ev = load_json(reports / "baseline_eval.json")
    if not ev:
        return None
    bcs = [r for r in ev["runs"] if r.get("kind", "bc") == "bc"]
    triv = [r for r in ev["runs"] if r.get("kind") == "trivial"]
    dist = _train_action_dist()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        [r["val_action_acc"] for r in bcs],
        [r["closed_loop_success"] for r in bcs],
        s=120, c="tab:blue", label="BC baseline (3 seeds)", zorder=3,
    )
    for r in bcs:
        ax.annotate(
            Path(r["run"]).name.replace("baseline_", ""),
            (r["val_action_acc"], r["closed_loop_success"]),
            textcoords="offset points", xytext=(8, -4), fontsize=8,
        )
    ax.scatter(
        [dist[2]], [triv_forward(triv)], s=100, marker="s", c="tab:red",
        label=f"always `forward` (acc {dist[2]:.1%})", zorder=3,
    )
    ax.scatter(
        [dist[0]], [triv_left(triv)], s=100, marker="^", c="tab:green",
        label=f"always `left` (acc {dist[0]:.1%})", zorder=3,
    )
    if any("random" in r["run"] for r in triv):
        ax.scatter(
            [1 / 3], [triv_value(triv, "random")], s=100, marker="D", c="tab:purple",
            label="uniform random (acc ≈ 33%)", zorder=3,
        )
    # 对角线参考：若准确率能线性映射到成功率
    xs = np.linspace(0, 1, 10)
    ax.plot(xs, xs, "--", color="grey", alpha=0.6, label="acc == success (reference)")

    ax.set_xlabel("验证集动作准确率 (teacher forcing)")
    ax.set_ylabel("闭环 episode Success 率（300 个未见 seed）")
    ax.set_title("图 2 · 动作准确率 vs 闭环 Success\n「准确率高 ≠ 任务成功」")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    p = out / "fig2_acc_vs_success.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def triv_forward(triv):
    for r in triv:
        if "always_forward" in r["run"]:
            return r["closed_loop_success"]
    return 0.0


def triv_left(triv):
    for r in triv:
        if "always_left" in r["run"]:
            return r["closed_loop_success"]
    return 0.0


def triv_value(triv, name: str) -> float:
    for r in triv:
        if r["run"].endswith(":" + name):
            return r["closed_loop_success"]
    return 0.0


def fig4_mission_heatmap(reports: Path, out: Path) -> str | None:
    p = reports / "mission_breakdown.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    grid = np.full((len(COLORS), len(TYPES) * 2), np.nan)
    labels = []
    for j, t in enumerate(TYPES):
        labels += [f"{t}\n(the)", f"{t}\n(a)"]
    for r in rows:
        if r["success_rate"] == "":
            continue
        i = COLORS.index(r["color"])
        j = TYPES.index(r["type"]) * 2 + (0 if r["article"] == "the" else 1)
        grid[i, j] = float(r["success_rate"])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(COLORS)))
    ax.set_yticklabels(COLORS)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(
                j, i, "n/a" if np.isnan(v) else f"{v:.2f}",
                ha="center", va="center", fontsize=8,
                color="black" if np.isnan(v) or v > 0.45 else "white",
            )
    fig.colorbar(im, ax=ax, label="Success rate")
    ax.set_title("图 4 · mission 类型分解（颜色 × 类型 × 冠词）\n空白 = 测试集中未出现")
    p = out / "fig4_mission_heatmap.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def fig6_transfer(reports: Path, out: Path) -> str | None:
    s6 = load_json(reports / "baseline_eval.json")
    s7 = load_json(reports / "s7n4_transfer.json")
    if not s7:
        return None
    b6 = [r for r in s6["runs"] if r.get("kind", "bc") == "bc"] if s6 else []
    b7 = [r for r in s7["runs"] if r.get("kind", "bc") == "bc"]

    labels = ["seed0", "seed1", "seed2"]
    x = np.arange(len(labels))
    w = 0.36
    v6 = [r["closed_loop_success"] for r in b6]
    v7 = [r["closed_loop_success"] for r in b7]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, v6, w, label="S6N3 (in-distribution)", color="tab:blue")
    ax.bar(x + w / 2, v7, w, label="S7N4 (transfer)", color="tab:orange")
    for xi, (a, b) in enumerate(zip(v6, v7)):
        ax.text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(xi, max(a, b) + 0.08, f"Δ={b - a:+.3f}", ha="center", fontsize=8, color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("closed-loop Success")
    ax.set_ylim(0, 1)
    ax.set_title("图 6 · S6N3 → S7N4 零样本迁移（各 300 个未见 seed）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    p = out / "fig6_s6n3_vs_s7n4.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def fig7_trajectories(reports: Path, out: Path) -> str | None:
    import imageio.v2 as imageio

    fail_pngs = sorted((reports / "failures").glob("seed*.png"))
    succ_gifs = sorted(Path("videos").glob("success_seed*.gif"))
    if not fail_pngs and not succ_gifs:
        return None

    rows: list[tuple[np.ndarray, str]] = []
    for g in succ_gifs[:2]:
        frames = imageio.imread(g)
        idx = [0, len(frames) // 2, len(frames) - 1]
        rows.append(
            (
                np.concatenate([frames[i] for i in idx], axis=1),
                f"成功 {g.stem}（首 / 中 / 末帧）",
            )
        )
    for png in fail_pngs[:2]:
        rows.append((imageio.imread(png), f"失败 {png.stem}（首 / 中 / 末帧）"))
    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.6 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, (im, cap) in zip(axes, rows):
        ax.imshow(im)
        ax.set_title(cap, fontsize=9, loc="left")
        ax.axis("off")
    fig.suptitle("图 7 · 成功 vs 失败轨迹（同一模型 baseline_seed0）", fontsize=11)
    p = out / "fig7_trajectories.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def summary_table(reports: Path, out: Path) -> str:
    ev = load_json(reports / "baseline_eval.json")
    s7 = load_json(reports / "s7n4_transfer.json")
    audit_done = (reports / "data_audit.md").exists()
    fa = load_json(reports / "failure_analysis.json")

    L = ["# 基线结果汇总表\n"]
    if ev:
        bcs = [r for r in ev["runs"] if r.get("kind", "bc") == "bc"]
        L.append(f"## S6N3 闭环评估（{ev['seeds']}，每模型 {ev['n_episodes_per_run']} episode）\n")
        L.append("| run | 验证动作准确率 | 闭环 Success | 平均回报 | 成功长度中位数 | 超时数 |")
        L.append("|---|---|---|---|---|---|")
        for r in bcs:
            L.append(
                f"| `{Path(r['run']).name}` | {r['val_action_acc']:.3f} | "
                f"**{r['closed_loop_success']:.3f}** | {r['mean_return']:.3f} | "
                f"{r['median_len_success']} | {r['n_timeout']}/{r['n_episodes']} |"
            )
        accs = [r["val_action_acc"] for r in bcs]
        succs = [r["closed_loop_success"] for r in bcs]
        L.append(
            f"| **mean ± SD** | {np.mean(accs):.3f} ± {np.std(accs):.3f} | "
            f"**{np.mean(succs):.3f} ± {np.std(succs):.3f}** | "
            f"{np.mean([r['mean_return'] for r in bcs]):.3f} | "
            f"{np.median([r['median_len_success'] for r in bcs]):.1f} | — |"
        )
        L.append("")
        triv = [r for r in ev["runs"] if r.get("kind") == "trivial"]
        if triv:
            L.append("## 平凡策略（证明「准确率 ≠ 成功率」）\n")
            L.append("| 策略 | teacher-forcing 准确率 | 闭环 Success |")
            L.append("|---|---|---|")
            acc_map = {
                "always_forward": _train_action_dist()[2],
                "always_left": _train_action_dist()[0],
                "always_right": _train_action_dist()[1],
                "random": "≈ 1/3（均匀随机）",
            }
            for r in triv:
                name = r["run"].split(":", 1)[1]
                a = acc_map.get(name, "—")
                a = f"{a:.3f}" if isinstance(a, float) else a
                L.append(f"| `{name}` | {a} | {r['closed_loop_success']:.3f} |")
            L.append("")
    if s7:
        L.append(f"## S7N4 迁移（{s7['seeds']}）\n")
        L.append("| run | S6N3 Success | S7N4 Success | Δ |")
        L.append("|---|---|---|---|")
        b6 = {r["train_seed"]: r["closed_loop_success"] for r in ev["runs"] if r.get("kind", "bc") == "bc"} if ev else {}
        for r in [x for x in s7["runs"] if x.get("kind", "bc") == "bc"]:
            a = b6.get(r["train_seed"])
            L.append(
                f"| `{Path(r['run']).name}` | {a:.3f} | **{r['closed_loop_success']:.3f}** | "
                f"{r['closed_loop_success'] - a:+.3f} |"
            )
        L.append("")
    if fa:
        L.append("## 失败归因（`runs/baseline_seed0`，300 episode）\n")
        L.append(f"成功 {fa['n_success']}/{fa['n_episodes']}，失败构成：\n")
        L.append("| 类型 | 数量 |")
        L.append("|---|---|")
        for k, v in sorted(fa["failure_types"].items(), key=lambda kv: -kv[1]):
            L.append(f"| `{k}` | {v} |")
        L.append("")
    L.append(f"- 数据审计：{'已完成 `reports/data_audit.md`' if audit_done else '缺失'}")
    p = out / "summary_table.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()
    reports, out = Path(args.reports), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = {}
    for fn, name in (
        (fig1_loss_curves, "fig1"),
        (fig2_acc_vs_success, "fig2"),
        (fig4_mission_heatmap, "fig4"),
        (fig6_transfer, "fig6"),
        (fig7_trajectories, "fig7"),
    ):
        try:
            r = fn(reports, out)
            made[name] = r
            print(f"{name}: {r or 'skipped'}")
        except Exception as e:  # 单张图失败不影响其它图
            print(f"{name}: FAILED ({e})")
    print("summary:", summary_table(reports, out))
    print("\n注意：图 3（基线 vs 修改）与图 5（数据效率）依赖 P3 阶段，本阶段不产出。")


if __name__ == "__main__":
    main()
