"""支线 4 · 课程学习汇总报告生成器 -> reports/b4_curriculum_summary.md

全部数值动态读取自 reports/b4_curriculum_*.json，零手抄。

用法:
    OMP_NUM_THREADS=1 python branches/b4_curriculum/make_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"

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

DESC = {
    "bc30": "基线对照（BCNet, S6N3 4500 ep, 30ep）",
    "pace_easy": "同域排序：易→难阶梯（40%→70%→100%）",
    "pace_rand": "同域排序对照：嵌套随机阶梯（40%→70%→100%）",
    "pace_hard": "同域反课程对照：难→易阶梯（40%→70%→100%）",
    "env_easy_stage": "两阶段纯接力：ep1-10 S5N2 → ep11-30 S6N3",
    "env_easy_cumul": "三阶段累积：ObjS6 → +S5N2 → +S6N3",
    "env_easy_mix": "等数据均匀混训对照（ObjS6+S5N2+S6N3）",
    "env_s6n4_prog": "渐进增难累积：S6N3 → +S6N4（高干扰密度）",
    "env_s6n4_mix": "等数据均匀混训对照（S6N3+S6N4）",
    "env_multiscale_prog": "多尺度渐进：S5N2+S6N3 → +S6N4+S8N6（无7×7）",
    "env_multiscale_mix": "多尺度等数据均匀混训（域随机化对照）",
    "film_easy_stage": "FiLM + 两阶段 S5N2 → S6N3 接力",
    "film_multiscale_prog": "FiLM + 多尺度渐进课程（无7×7）",
    "film_multiscale_mix": "FiLM + 多尺度均匀域随机化（无7×7）",
    "b2_film_nwd": "FiLM 基线对照（S6N3 4500 ep, wd=0, film_l2=0）",
}


def jload(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_ms(a: dict, k: str, sk: str, nd: int = 4) -> str:
    m, s = a.get(k), a.get(sk)
    if m is None:
        return "—"
    if s is None:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main() -> None:
    ev = jload("b4_curriculum_eval.json")
    s7 = jload("b4_curriculum_s7n4.json")
    spec = jload("b4_curriculum_spectrum.json")
    sens = jload("b4_curriculum_sensitivity.json")
    f6 = jload("b4_curriculum_failures.json")
    f7 = jload("b4_curriculum_failures_s7n4.json")

    L: list[str] = []
    L.append("# 支线 4 · 课程学习（Curriculum Learning）消融与机制汇总\n")
    L.append("> 由 `branches/b4_curriculum/make_summary.py` 自动生成，所有数值均动态读取自 `reports/b4_curriculum_*.json`。\n")

    if ev and s7:
        bv6, bv7 = ev["by_variant"], s7["by_variant"]
        L.append(f"## 1. 主结果（S6N3 `{ev['seeds']}` 与跨房间未见尺寸迁移 S7N4 `{s7['seeds']}`，每组 `n=3`）\n")
        L.append("| 组 | 架构 | val 准确率 (S6) | **S6N3 Success** | Δ vs BC30 (S6) | **S7N4 Success** | Δ vs BC30 (S7) | 实验设计 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for g in ORDER:
            if g not in bv6:
                continue
            a6, a7 = bv6[g], bv7.get(g, {})
            arch = "FiLM" if a6.get("use_film") else "Concat"
            p6 = a6.get("paired_vs_bc30")
            p7 = a7.get("paired_vs_bc30")
            d6 = f"{p6['delta_mean']:+.4f} ({p6['n_positive']}/3)" if p6 else "—"
            d7 = f"{p7['delta_mean']:+.4f} ({p7['n_positive']}/3)" if p7 else "—"
            s7s = fmt_ms(a7, "success_mean", "success_sd") if a7 else "—"
            L.append(
                f"| `{g}` | {arch} | {fmt_ms(a6, 'val_acc_mean', 'val_acc_sd')} | "
                f"**{fmt_ms(a6, 'success_mean', 'success_sd')}** | {d6} | "
                f"**{s7s}** | {d7} | {DESC.get(g, '')} |"
            )
        L.append("")

        L.append("### 1.1 课程排序 vs 均匀混合 / 反课程：3-seed 受控配对检验（LHS − RHS）\n")
        L.append("| 配对假设（LHS − RHS） | S6N3 Δ ± SD (正/3) | S7N4 Δ ± SD (正/3) | 机制结论 |")
        L.append("|---|---|---|---|")
        pair_notes = [
            ("pace_easy_vs_pace_rand", "同域 S6N3：易→难 vs 随机阶梯（验证排序本身）"),
            ("pace_hard_vs_pace_easy", "同域 S6N3：反课程难→易 vs 易→难"),
            ("env_easy_cumul_vs_env_easy_mix", "简单环境累积阶段 vs 等数据均匀混合（ObjS6+S5N2+S6N3）"),
            ("env_easy_stage_vs_env_easy_mix", "两阶段非累积接力 S5N2→S6N3 vs 均匀混合"),
            ("env_s6n4_prog_vs_env_s6n4_mix", "高干扰渐进 S6N3→+S6N4 vs 均匀混合"),
            ("env_multiscale_prog_vs_env_multiscale_mix", "多尺度渐进课程 vs 多尺度域随机化（Concat）"),
            ("film_multiscale_prog_vs_film_multiscale_mix", "多尺度渐进课程 vs 多尺度域随机化（FiLM）"),
            ("film_easy_stage_vs_b2_film_nwd", "FiLM 两阶段 S5N2→S6N3 vs 纯 S6N3 FiLM"),
            ("film_multiscale_prog_vs_b2_film_nwd", "FiLM 多尺度渐进 vs 纯 S6N3 FiLM"),
            ("film_multiscale_mix_vs_b2_film_nwd", "FiLM 多尺度域随机化 vs 纯 S6N3 FiLM"),
        ]
        pc6, pc7 = ev.get("paired_comparisons", {}), s7.get("paired_comparisons", {})
        for k, note in pair_notes:
            p6, p7 = pc6.get(k), pc7.get(k)
            s6s = f"{p6['delta_mean']:+.4f} ± {p6['delta_sd']:.4f} ({p6['n_positive']}/{p6['n_pairs']})" if p6 else "—"
            s7s = f"{p7['delta_mean']:+.4f} ± {p7['delta_sd']:.4f} ({p7['n_positive']}/{p7['n_pairs']})" if p7 else "—"
            L.append(f"| `{k}` | {s6s} | {s7s} | {note} |")
        L.append("")

    if spec:
        bv_sp = spec["by_variant"]
        tags = [e["tag"] for e in spec["envs"]]
        L.append("## 2. 五级环境难度谱泛化曲线（S5N2 → S6N3 → S6N4 → S7N4 [未见7×7] → S8N6）\n")
        L.append("| 组 | " + " | ".join(tags) + " | 5环境宏平均 |")
        L.append("|---|" + "|".join(["---"] * (len(tags) + 1)) + "|")
        for g in ORDER:
            if g not in bv_sp:
                continue
            a = bv_sp[g]
            vals = [f"{a[t]['success_mean']:.3f}" for t in tags]
            L.append(f"| `{g}` | " + " | ".join(vals) + f" | **{a['macro_mean']:.3f}** |")
        L.append("")

    if sens:
        bv_se = sens["by_variant"]
        L.append(f"## 3. 语言语义接地反事实探针（{sens['n_states']} 个可切换目标 S6N3 状态）\n")
        L.append("| 组 | 换颜色敏感度 | 换类型敏感度 | 合计敏感度 |")
        L.append("|---|---|---|---|")
        for g in ORDER:
            if g not in bv_se:
                continue
            a = bv_se[g]
            L.append(
                f"| `{g}` | **{a['color_sensitivity_mean']:.4f}** | "
                f"{a['type_sensitivity_mean']:.4f} | {a['combined_sensitivity_mean']:.4f} |"
            )
        L.append("")

    if f6 and f7:
        ag6, ag7 = f6["aggregate"], f7["aggregate"]
        L.append("## 4. 失败机制拆解对比（S6N3 vs S7N4，每 300 局平均）\n")
        L.append("| 组 | S6 失败 (表征+覆盖) | S6 `attr_conf` | S7 失败 (表征+覆盖) | S7 `attr_conf` | S7 `approach` | S7 `wander` |")
        L.append("|---|---|---|---|---|---|---|")
        for g in ORDER:
            if g not in ag6:
                continue
            a6, a7 = ag6[g], ag7.get(g, {})
            fg6, ft6 = a6["failure_groups_mean"], a6["failure_types_mean"]
            fg7, ft7 = a7.get("failure_groups_mean", {}), a7.get("failure_types_mean", {})
            L.append(
                f"| `{g}` | {a6['n_failure_mean']:.1f} ({fg6.get('representation',0):.1f}+{fg6.get('state_coverage',0):.1f}) "
                f"| {ft6.get('attribute_confusion',0):.1f} "
                f"| {a7.get('n_failure_mean',0):.1f} ({fg7.get('representation',0):.1f}+{fg7.get('state_coverage',0):.1f}) "
                f"| {ft7.get('attribute_confusion',0):.1f} "
                f"| {ft7.get('approach_then_lost',0):.1f} | {ft7.get('wander_timeout',0):.1f} |"
            )
        L.append("")

    if ev and s7 and spec and sens and f6 and f7:
        bv6, bv7, pc6, pc7 = ev["by_variant"], s7["by_variant"], ev["paired_comparisons"], s7["paired_comparisons"]
        bv_sp, bv_se = spec["by_variant"], sens["by_variant"]
        L.append("## 5. 核心科学发现（全由实测 JSON 推导）\n")
        L.append(
            f"1. **同域样本排序存在温和但一致的正效应**：在完全相同的 4,500 条 S6N3 训练数据上，"
            f"`pace_easy`（{bv6['pace_easy']['success_mean']:.3f} S6 / {bv7['pace_easy']['success_mean']:.3f} S7）高于 "
            f"`pace_rand`（配对 S6 `{pc6['pace_easy_vs_pace_rand']['delta_mean']:+.4f}`, {pc6['pace_easy_vs_pace_rand']['n_positive']}/3；"
            f"S7 `{pc7['pace_easy_vs_pace_rand']['delta_mean']:+.4f}`, {pc7['pace_easy_vs_pace_rand']['n_positive']}/3），"
            f"且反向课程 `pace_hard` 显著退步至 {bv6['pace_hard']['success_mean']:.3f} S6 / {bv7['pace_hard']['success_mean']:.3f} S7（`pace_hard - pace_easy` S6 `{pc6['pace_hard_vs_pace_easy']['delta_mean']:+.4f}`）。"
        )
        L.append(
            f"2. **但课程是否优于均匀混训，高度取决于课程阶段顺序与网络架构**："
            f"在 Concat 架构下，`env_multiscale_prog`（先学 5×5/6×6 再引入高难 N4/8×8）在 S6/S7 分别达到 "
            f"{bv6['env_multiscale_prog']['success_mean']:.3f} / {bv7['env_multiscale_prog']['success_mean']:.3f}，"
            f"大幅领先等数据均匀混合 `env_multiscale_mix`（{bv6['env_multiscale_mix']['success_mean']:.3f} / {bv7['env_multiscale_mix']['success_mean']:.3f}，"
            f"配对 `{pc6['env_multiscale_prog_vs_env_multiscale_mix']['delta_mean']:+.4f}` S6 / `{pc7['env_multiscale_prog_vs_env_multiscale_mix']['delta_mean']:+.4f}` S7）；"
            f"然而当先用零干扰 `ObjS6`/小房间 `S5N2` 单独预热再混合（`env_easy_cumul` {bv6['env_easy_cumul']['success_mean']:.3f}），反而出现遗忘/捷径锁定。"
        )
        L.append(
            f"3. **跨尺度多环境 + FiLM 将未见 S7N4 与大房间 S8N6 推至新高**："
            f"`film_multiscale_prog` 与 `film_multiscale_mix` 均突破单环境 `b2_film_nwd` 的跨尺度瓶颈："
            f"S7N4 由 {bv7['b2_film_nwd']['success_mean']:.3f} 跃升至 **{bv7['film_multiscale_prog']['success_mean']:.3f}** / {bv7['film_multiscale_mix']['success_mean']:.3f}"
            f"（`film_multiscale_prog - b2_film_nwd` 配对 S7 **`{pc7['film_multiscale_prog_vs_b2_film_nwd']['delta_mean']:+.4f}`**, {pc7['film_multiscale_prog_vs_b2_film_nwd']['n_positive']}/3 胜出）；"
            f"在全未见最大房间 **S8N6** 上从 {bv_sp['b2_film_nwd']['S8N6']['success_mean']:.3f} 提升至 **{bv_sp['film_multiscale_mix']['S8N6']['success_mean']:.3f}** / {bv_sp['film_multiscale_prog']['S8N6']['success_mean']:.3f}，"
            f"5 环境宏平均达 **{bv_sp['film_multiscale_mix']['macro_mean']:.3f}**。"
        )
        concat_cols = [bv_se[g]["color_sensitivity_mean"] for g in ORDER if g in bv_se and not bv_se[g]["use_film"]]
        L.append(
            f"4. **机制证据链：为什么只有高密度干扰物/FiLM 才能撬动 S7N4/S8N6？**："
            f"语义接地探针表明，Concat 下所有课程变体颜色敏感度仍停留在 `{min(concat_cols):.3f}–{max(concat_cols):.3f}`（`env_easy_cumul` 更因无干扰预热跌至 `{bv_se['env_easy_cumul']['color_sensitivity_mean']:.3f}`），"
            f"而 `film_multiscale_*` 颜色敏感度维持在 `{bv_se['film_multiscale_prog']['color_sensitivity_mean']:.3f}–{bv_se['film_multiscale_mix']['color_sensitivity_mean']:.3f}`，"
            f"同时借助 S6N4/S8N6 的大房间与密集干扰物将 S7N4 失败总数从 `b2_film_nwd` 的 {f7['aggregate']['b2_film_nwd']['n_failure_mean']:.1f}（`attr_conf` {f7['aggregate']['b2_film_nwd']['failure_types_mean'].get('attribute_confusion', 0):.1f}）"
            f"进一步压低至 **{f7['aggregate']['film_multiscale_prog']['n_failure_mean']:.1f}**（`attr_conf` **{f7['aggregate']['film_multiscale_prog']['failure_types_mean'].get('attribute_confusion', 0):.1f}**）。"
        )
    L.append("")

    out = R / "b4_curriculum_summary.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
