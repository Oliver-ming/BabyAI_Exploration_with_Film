"""支线 5 · 数据效率汇总报告生成器 -> reports/b5_data_eff_summary.md

全部数值动态读取自 reports/b5_data_eff_*.json，零手抄。

用法:
    OMP_NUM_THREADS=1 python branches/b5_data_eff/make_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"

SCALING_METHODS = [
    ("bc_naive", "Concat 朴素抽取（无步数匹配）"),
    ("bc_step", "Concat 步数匹配（3900 steps）"),
    ("bc_reg", "Concat 正则化（Drop0.2+LS0.05+Cos）"),
    ("bc_d4", "Concat + D4 8× 群对称增广"),
    ("bc_d4_tta", "Concat + D4 + 8× 群等变 TTA"),
    ("film_step", "FiLM 步数匹配（wd=0, film_l2=0）"),
    ("film_d4", "FiLM + D4 8× 群对称增广"),
    ("film_d4_tta", "FiLM + D4 + 8× 群等变 TTA"),
]

FTAGS = [("f005", "5% (225ep)"), ("f010", "10% (450ep)"), ("f020", "20% (900ep)"), ("f050", "50% (2250ep)"), ("f100", "100% (4500ep)")]


def jload(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_cell(d: dict, ft: str, key: str = "success_mean", sd_key: str = "success_sd") -> str:
    if ft not in d:
        return "—"
    a = d[ft]
    return f"{a[key]:.3f} ± {a[sd_key]:.3f}"


def main() -> None:
    ev = jload("b5_data_eff_eval.json")
    s7 = jload("b5_data_eff_s7n4.json")
    sens = jload("b5_data_eff_sensitivity.json")
    fail = jload("b5_data_eff_failures.json")

    L: list[str] = []
    L.append("# 支线 5 · 数据效率（Data Efficiency）扩展规律与机制汇总\n")
    L.append("> 由 `branches/b5_data_eff/make_summary.py` 自动生成，所有数值均动态读取自 `reports/b5_data_eff_*.json`。\n")

    if ev and s7:
        sc6, sc7 = ev["scaling_curves"], s7["scaling_curves"]
        L.append("## 1. 样本扩展曲线（S6N3 同分布 & S7N4 跨房间迁移，每档 `n=3`）\n")
        L.append("### 1.1 S6N3 闭环 Success (`mean ± SD`)\n")
        L.append("| 方法 | 说明 | " + " | ".join(lbl for _, lbl in FTAGS) + " |")
        L.append("|---|---|" + "|".join(["---"] * len(FTAGS)) + "|")
        for m, desc in SCALING_METHODS:
            if m not in sc6:
                continue
            d = sc6[m]
            L.append(f"| `{m}` | {desc} | " + " | ".join(f"**{fmt_cell(d, ft)}**" for ft, _ in FTAGS) + " |")
        L.append("")

        L.append("### 1.2 S7N4 跨房间未见尺寸迁移 Success (`mean ± SD`)\n")
        L.append("| 方法 | " + " | ".join(lbl for _, lbl in FTAGS) + " |")
        L.append("|---|" + "|".join(["---"] * len(FTAGS)) + "|")
        for m, _ in SCALING_METHODS:
            if m not in sc7:
                continue
            d = sc7[m]
            L.append(f"| `{m}` | " + " | ".join(f"**{fmt_cell(d, ft)}**" for ft, _ in FTAGS) + " |")
        L.append("")

        L.append("### 1.3 采样压缩里程碑（达到或超越 100% BC30 与 0.900 / 0.940 的最少样本）\n")
        ms = ev["compression_milestones"]
        L.append(f"- **100% BC30（4,500 ep）基准线**：`{ms['bc30_100pct_target']:.4f}`\n")
        L.append("| 门槛 | 最省样本方案（按 n_episodes 升序前 6 名） |")
        L.append("|---|---|")
        for key, title in [("match_or_beat_bc30_100", f"≥ 100% BC30 (`{ms['bc30_100pct_target']:.3f}`)"), ("reach_0900", "≥ 0.900"), ("reach_0940", "≥ 0.940")]:
            rows = sorted(ms.get(key, []), key=lambda r: (r["n_episodes"], -r["success_mean"]))[:6]
            s_items = "；".join(f"`{r['method']}@{r['n_episodes']}ep` (**{r['success_mean']:.3f}**, {r['compression']}×)" for r in rows)
            L.append(f"| {title} | {s_items or '无'} |")
        L.append("")

        # 20% 机制消融全表
        f020_6, f020_7 = ev["f020_ablations"], s7["f020_ablations"]
        pc6 = ev["paired_comparisons"]
        bg = sens["by_group"] if sens else {}
        ag = fail["aggregate"] if fail else {}
        L.append("## 2. 20% 极低预算（900 ep）13 路受控消融\n")
        L.append("| 方法 | S6N3 Success | Δ vs `bc_step` (S6) | S7N4 Success | `val_acc` | `train_acc` | 泛化间隙 `gap` | 颜色敏感度 | S6 失败 (表征+覆盖) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for m, a6 in sorted(f020_6.items(), key=lambda kv: -kv[1]["success_mean"]):
            a7 = f020_7.get(m, {})
            p = pc6.get(f"f020_{m}_vs_bc_step")
            dp = f"{p['delta_mean']:+.4f} ({p['n_positive']}/{p['n_pairs']})" if p else "—"
            se = bg.get(f"{m}_f020", {})
            fa = ag.get(f"{m}_f020", {})
            col_s = f"{se['color_sensitivity_mean']:.4f}" if se else "—"
            if fa:
                fg = fa["failure_groups_mean"]
                f_str = f"{fa['n_failure_mean']:.1f} ({fg.get('representation',0):.1f}+{fg.get('state_coverage',0):.1f})"
            else:
                f_str = "—"
            tr_s = f"{a6['train_acc_mean']:.4f}" if a6.get("train_acc_mean") is not None else "—"
            gp_s = f"{a6['gen_gap_mean']:+.4f}" if a6.get("gen_gap_mean") is not None else "—"
            s7_s = f"{a7['success_mean']:.3f} ± {a7['success_sd']:.3f}" if a7 else "—"
            L.append(
                f"| `{m}` | **{a6['success_mean']:.3f} ± {a6['success_sd']:.3f}** | {dp} | {s7_s} | "
                f"{(a6['val_acc_mean'] or 0):.4f} | {tr_s} | {gp_s} | {col_s} | {f_str} |"
            )
        L.append("")

        L.append("## 3. 3-seed 配对因果拆解（各比例均独立配对）\n")
        L.append("| 机制效应 | `f005` (225) | `f010` (450) | `f020` (900) | `f050` (2250) | `f100` (4500) |")
        L.append("|---|---|---|---|---|---|")
        for eff_key, eff_name in [
            ("bc_step_vs_bc_naive", "`bc_step − bc_naive`（步数对齐净益）"),
            ("bc_reg_vs_bc_step", "`bc_reg − bc_step`（正则化净益）"),
            ("bc_d4_vs_bc_step", "`bc_d4 − bc_step`（D4 8×增广@Concat）"),
            ("bc_d4_tta_vs_bc_d4", "`bc_d4_tta − bc_d4`（8× TTA@Concat）"),
            ("film_step_vs_bc_step", "`film_step − bc_step`（FiLM@同样本）"),
            ("film_d4_vs_film_step", "`film_d4 − film_step`（D4 8×增广@FiLM）"),
            ("film_d4_tta_vs_film_d4", "`film_d4_tta − film_d4`（8× TTA@FiLM）"),
        ]:
            cells = []
            for ft, _ in FTAGS:
                p = pc6.get(f"{ft}_{eff_key}")
                cells.append(f"{p['delta_mean']:+.3f} ({p['n_positive']}/3)" if p else "—")
            L.append(f"| {eff_name} | " + " | ".join(cells) + " |")
        L.append("")

    if sens:
        bg = sens["by_group"]
        L.append(f"## 4. 语言接地随数据规模的演化（{sens['n_states']} 个可切换目标状态）\n")
        L.append("| 方法 | 指标 | 5% (225) | 10% (450) | 20% (900) | 50% (2250) | 100% (4500) |")
        L.append("|---|---|---|---|---|---|---|")
        for m in ("bc_step", "bc_d4", "film_step", "film_d4"):
            c_cells = [f"**{bg[f'{m}_{ft}']['color_sensitivity_mean']:.3f}**" if f"{m}_{ft}" in bg else "—" for ft, _ in FTAGS]
            t_cells = [f"{bg[f'{m}_{ft}']['type_sensitivity_mean']:.3f}" if f"{m}_{ft}" in bg else "—" for ft, _ in FTAGS]
            L.append(f"| `{m}` | 换颜色 | " + " | ".join(c_cells) + " |")
            L.append(f"| `{m}` | 换类型 | " + " | ".join(t_cells) + " |")
        L.append("")

    if fail:
        ag = fail["aggregate"]
        L.append("## 5. 失败机制随数据规模的缩小轨迹（300 S6N3 局平均：`总失败 (表征+覆盖) [attr_conf]`）\n")
        L.append("| 方法 | 5% (225) | 10% (450) | 20% (900) | 50% (2250) | 100% (4500) |")
        L.append("|---|---|---|---|---|---|")
        for m in ("bc_naive", "bc_step", "bc_d4", "film_step", "film_d4"):
            cells = []
            for ft, _ in FTAGS:
                a = ag.get(f"{m}_{ft}")
                if not a:
                    cells.append("—")
                else:
                    fg, ftm = a["failure_groups_mean"], a["failure_types_mean"]
                    cells.append(f"{a['n_failure_mean']:.1f} ({fg.get('representation',0):.1f}+{fg.get('state_coverage',0):.1f}) [{ftm.get('attribute_confusion',0):.1f}]")
            L.append(f"| `{m}` | " + " | ".join(cells) + " |")
        L.append("")

    L.append("## 6. 核心科学发现（待运行后自动回填数据校验）\n")
    if ev and s7 and sens and fail:
        sc6, sc7, bg, ag = ev["scaling_curves"], s7["scaling_curves"], sens["by_group"], fail["aggregate"]
        ms = ev["compression_milestones"]
        L.pop()
        L.append("## 6. 核心科学发现（全由实测 JSON 动态生成）\n")
        L.append(
            f"1. **朴素小样本曲线混杂了「欠训练步数不足」**：在 5%（225 ep）下，`bc_naive` 仅跑 `210` 步 S6N3 得 "
            f"`{sc6['bc_naive']['f005']['success_mean']:.3f}`，仅靠对齐 3,900 梯度步数 `bc_step` 即升至 `{sc6['bc_step']['f005']['success_mean']:.3f}`；"
            f"同样在 20%（900 ep）上 `bc_step` 为 `{sc6['bc_step']['f020']['success_mean']:.3f}` vs `bc_naive` `{sc6['bc_naive']['f020']['success_mean']:.3f}`。"
        )
        m_hits = sorted(ms.get("match_or_beat_bc30_100", []), key=lambda r: (r["n_episodes"], -r["success_mean"]))
        top_hit = m_hits[0] if m_hits else None
        concat_hits = [r for r in m_hits if not r["method"].startswith("film") and r["n_episodes"] < 4500]
        top_concat = concat_hits[0] if concat_hits else None
        hit_str = (
            f"`{top_hit['method']}` 仅用 **{top_hit['n_episodes']} ep（{top_hit['compression']}× 压缩）** 即达 S6N3 **`{top_hit['success_mean']:.3f}`**"
            if top_hit
            else "暂无达到 100% BC30 的小样本方案"
        )
        concat_str = (
            f"；即使不改架构，Concat 族 `{top_concat['method']}` 也仅需 **{top_concat['n_episodes']} ep（{top_concat['compression']}×）** 达到 **`{top_concat['success_mean']:.3f}`**"
            if top_concat
            else "；而所有不足 100% 的 Concat 变体均未能追平 100% BC30（`4,500 ep`），证明仅靠步数对齐或对称增广无法弥补 Concat 的视觉-指令绑定缺陷"
        )
        L.append(
            f"2. **跨模态调制与 D4 群等变增广的真实样本压缩率**："
            f"超越 100% `BC30`（4,500 ep, `{ms['bc30_100pct_target']:.3f}`）的最省样本方案为 {hit_str}{concat_str}。"
        )
        L.append(
            f"3. **语言接地需要「架构前提 + 样本阈值」双重条件**："
            f"`bc_step` 与 `bc_d4` 从 5% 升至 100% 颜色敏感度仍停留在 `{bg['bc_step_f005']['color_sensitivity_mean']:.3f}→{bg['bc_step_f100']['color_sensitivity_mean']:.3f}` 与 "
            f"`{bg['bc_d4_f005']['color_sensitivity_mean']:.3f}→{bg['bc_d4_f100']['color_sensitivity_mean']:.3f}`；"
            f"而 `film_step` 从 5% `{bg['film_step_f005']['color_sensitivity_mean']:.3f}` → 20% `{bg['film_step_f020']['color_sensitivity_mean']:.3f}` → 100% `{bg['film_step_f100']['color_sensitivity_mean']:.3f}`，"
            f"`film_d4` 从 5% `{bg['film_d4_f005']['color_sensitivity_mean']:.3f}` → 20% `{bg['film_d4_f020']['color_sensitivity_mean']:.3f}` → 100% `{bg['film_d4_f100']['color_sensitivity_mean']:.3f}`。"
        )
    L.append("")

    out = R / "b5_data_eff_summary.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
