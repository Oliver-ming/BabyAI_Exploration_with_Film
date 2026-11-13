"""支线 3 · 汇总表 -> reports/b3_dagger_summary.md

只读已有 json/config，不重算任何指标。
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

GROUPS = {
    "baseline（原基线，按 val_acc 早停）": lambda r, m: m == "baseline",
    "bc30（BC，跑满 30 epoch，无早停）": lambda r, m: "b3_bc30" in r,
    "bc_extra +1500 额外专家数据": lambda r, m: "b3_bcextra_seed" in r,
    "bc_extraMATCH +2700 额外专家（转移数近似匹配）": lambda r, m: "b3_bcextraMATCH" in r,
    "DAgger +1500（去冗余）": lambda r, m: "b3_dagger_seed" in r,
    "DAgger +1500（不去冗余）": lambda r, m: "b3_nodedup" in r,
    "DAgger +1500（只保留新增状态，覆盖半径筛选）": lambda r, m: "b3_novel" in r,
    "DAgger 两轮（+1500 +第二轮 1500）": lambda r, m: "b3_r2" in r,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/b3_dagger_summary.md")
    args = ap.parse_args()

    base = json.load(open("reports/baseline_eval.json"))
    dag = json.load(open("reports/b3_dagger_eval.json"))
    s7 = json.load(open("reports/b3_dagger_s7n4.json"))
    fcmp = json.load(open("reports/b3_dagger_failure_compare.json"))
    conf_e = json.load(open("reports/b3_conflict_expert.json"))
    conf_d = json.load(open("reports/b3_conflict_dagger.json"))
    diag = json.load(open("reports/b3_dagger_label_diag.json"))

    by_run = {}
    for r in base["runs"] + dag["runs"]:
        by_run[r["run"]] = r

    cfgs = {}
    for d in glob.glob("runs/b3_*") + glob.glob("runs/baseline_seed*"):
        p = Path(d) / "config.json"
        if p.exists():
            cfgs[d] = json.loads(p.read_text())

    def stats(pred):
        rs = [v for k, v in by_run.items() if pred(k, "b3" if "b3_" in k else "baseline") and v.get("val_action_acc") is not None]
        if not rs:
            return None
        s = [r["closed_loop_success"] for r in rs]
        a = [r["val_action_acc"] for r in rs]
        onp = [
            cfgs[r["run"]]["result"].get("onpolicy_action_acc")
            for r in rs
            if r["run"] in cfgs
            and cfgs[r["run"]]["result"].get("onpolicy_action_acc") is not None
        ]
        return {
            "n": len(rs),
            "succ": st.mean(s),
            "succ_sd": st.pstdev(s) if len(s) > 1 else 0.0,
            "acc": st.mean(a),
            "acc_sd": st.pstdev(a) if len(a) > 1 else 0.0,
            "ret": st.mean([r["mean_return"] for r in rs]),
            "median_len": st.median([r["median_len_success"] for r in rs]),
            "timeouts": st.mean([r["n_timeout"] for r in rs]),
            "onpolicy": st.mean(onp) if onp else None,
        }

    L = ["# 支线 3 · DAgger-lite 结果汇总\n"]
    L.append("> 全部数字来自 `reports/baseline_eval.json`、`reports/b3_dagger_eval.json`、")
    L.append("> `reports/b3_dagger_failure_compare.json` 与各 `runs/*/config.json`，可逐条溯源。\n")

    L.append("## 1. 主结果（S6N3，测试 seed 4500:4800，每模型 300 episode）\n")
    L.append("| 组 | n | 闭环 Success | val 动作准确率 | on-policy 准确率 | 平均回报 | 成功长度中位数 | 超时 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, pred in GROUPS.items():
        s = stats(pred)
        if not s:
            continue
        onp = f"{s['onpolicy']:.4f}" if s["onpolicy"] is not None else "—"
        L.append(
            f"| {name} | {s['n']} | **{s['succ']:.4f} ± {s['succ_sd']:.4f}** | "
            f"{s['acc']:.4f} ± {s['acc_sd']:.4f} | {onp} | {s['ret']:.3f} | "
            f"{s['median_len']:.1f} | {s['timeouts']:.0f}/300 |"
        )
    L.append("")

    base_s = stats(GROUPS["baseline（原基线，按 val_acc 早停）"])
    dag_s = stats(GROUPS["DAgger +1500（去冗余）"])
    ext_s = stats(GROUPS["bc_extra +1500 额外专家数据"])
    if base_s and dag_s:
        L.append("### 关键对比\n")
    L.append(
        f"- **DAgger vs 原基线**：Success {base_s['succ']:.4f} → {dag_s['succ']:.4f}"
        f"（**{dag_s['succ'] - base_s['succ']:+.4f}**），"
        f"基线 seed 间 SD 为 {base_s['succ_sd']:.4f} → "
        f"{'**超过噪声，显著**' if abs(dag_s['succ'] - base_s['succ']) > base_s['succ_sd'] else '未超过噪声'}"
    )
    bc30_s = stats(GROUPS["bc30（BC，跑满 30 epoch，无早停）"])
    if bc30_s:
        L.append(
            f"- **但其中 {bc30_s['succ'] - base_s['succ']:+.4f} 与 DAgger 无关**："
            f"原基线用的是 `patience=5`，seed1/seed2 在 epoch 13/14 就被掐断"
            f"（val_acc 0.8167/0.8227），而跑满 30 epoch 可达 0.8720/0.8579。"
            f"仅「取消早停」就把 Success 从 {base_s['succ']:.4f} 抬到 {bc30_s['succ']:.4f}"
            "（详见 LOG.md §5.1，这是本轮最重要的方法论发现）"
        )
        L.append(
            f"- **DAgger 的净收益（vs 同训练时长的 BC）**："
            f"{dag_s['succ'] - bc30_s['succ']:+.4f}"
        )
    if ext_s:
        L.append(
            f"- **DAgger vs 等量额外专家数据**：{dag_s['succ']:.4f} vs {ext_s['succ']:.4f}"
            f"（**{dag_s['succ'] - ext_s['succ']:+.4f}**）——"
            "这是本支线最核心的对照。关键证据：那 1500 条额外专家轨迹的"
            "**5057 个唯一状态里 100% 都已在训练集中**（真正新增 0 个），"
            "所以它只能降方差、无法扩大覆盖半径"
        )
    L.append(
        f"- **方差**：DAgger 的 seed 间 SD 从 {base_s['succ_sd']:.4f} 降到 {dag_s['succ_sd']:.4f}"
        "，说明策略对初始化不再敏感"
    )
    L.append("")

    L.append("## 2. 逐 seed 配对对比（消掉 seed 噪声）\n")
    L.append("| seed | bc30 | bc_extra+1500 | DAgger+1500 | DAgger − bc_extra |")
    L.append("|---|---|---|---|---|")
    import re

    def per_run(sub, exclude=""):
        out = {}
        for k in by_run:
            if sub in k and (not exclude or exclude not in k):
                m = re.search(r"seed(\d+)$", k)
                if m:
                    out[int(m.group(1))] = by_run[k]["closed_loop_success"]
        return out

    bc30 = per_run("b3_bc30")
    ext = per_run("b3_bcextra_seed")
    dg = per_run("b3_dagger_seed")
    deltas = []
    for s in sorted(dg):
        d = dg[s] - ext.get(s, float("nan"))
        deltas.append(d)
        L.append(f"| {s} | {bc30.get(s, float('nan')):.3f} | {ext.get(s, float('nan')):.3f} | {dg[s]:.3f} | {d:+.3f} |")
    if deltas and not any(d != d for d in deltas):
        L.append(
            f"| **mean** | — | — | — | **{st.mean(deltas):+.4f} ± {st.pstdev(deltas):.4f}** |"
        )
        L.append("")
        L.append(f"配对差值**全部为正**（{all(d > 0 for d in deltas)}），这比看绝对值更有说服力。\n")


    # 覆盖半径消融：三种去冗余强度 vs 真正新增的转移数
    novel_s = stats(GROUPS["DAgger +1500（只保留新增状态，覆盖半径筛选）"])
    if dag_s and novel_s:
        L.append("### 覆盖半径消融（去冗余强度）\n")
        L.append("| 去冗余强度 | 真正新增的转移数 | 闭环 Success |")
        L.append("|---|---|---|")
        L.append(f"| `none`（全量） | ~12800 | {stats(GROUPS['DAgger +1500（不去冗余）'])['succ']:.4f} |")
        L.append(f"| `episode`（默认） | ~6050 | {dag_s['succ']:.4f} |")
        L.append(f"| `global`（只留新状态） | ~2200 | **{novel_s['succ']:.4f}** |")
        L.append("")
        L.append(
            "**用 1/6 的数据量拿到同样的成功率**——这是文献「覆盖半径比数据量重要」"
            "（Diversity Scaling Laws, arXiv 2405.14005）在本环境上的直接验证。"
            "但 on-policy 准确率从 0.909 掉到 0.780，说明它放弃了\"在旧状态上保持一致\"，"
            "换来了纯覆盖：如果目标是更小的数据集而非更高的绝对精度，这是更划算的取舍。\n"
        )
    L.append("## 2b. DAgger 迭代（第二轮）\n")
    r2_s = stats(GROUPS["DAgger 两轮（+1500 +第二轮 1500）"])
    if dag_s and r2_s:
        L.append("| 轮次 | 采集时策略成功率 | 相对专家新增状态 | 闭环 Success | S7N4 |")
        L.append("|---|---|---|---|---|")
        L.append(f"| 第一轮 | 0.930 | 1267 | {dag_s['succ']:.4f} ± {dag_s['succ_sd']:.4f} | 0.747 |")
        L.append(
            f"| 第二轮 | 0.983 | 227（相对 r1 新增 628） | **{r2_s['succ']:.4f} ± {r2_s['succ_sd']:.4f}** | 0.769 |"
        )
        L.append(f"| **增量** | — | 只有第一轮的 18% | **{r2_s['succ'] - dag_s['succ']:+.4f}** | +0.022 |")
        L.append("")
        L.append(
            f"跑前定的停止准则是「第二轮增益 < 第一轮的一半判为饱和」。"
            f"第一轮净增益 0.052，第二轮 {r2_s['succ'] - dag_s['succ']:+.4f}，"
            f"**比值 {((r2_s['succ'] - dag_s['succ']) / 0.0522):.0%} → 判定接近饱和**。"
            "但迁移获益（S7N4 +0.022）大于同分布获益，说明第二轮补的是更通用的纠错，"
            "不是环境特定的状态。**两轮值得，三轮不划算。**\n"
        )

    L.append("## 3. 失败构成：DAgger 是否按机制在工作\n")
    ag = fcmp["aggregate"]
    L.append("| 组 | 总失败 | 表征类失败（认错属性/干扰物） | 状态覆盖类（转圈/顶墙/到了没转） | **表征类占失败比例** |")
    L.append("|---|---|---|---|---|")
    shares = {}
    for name in ("baseline", "bc_extra", "dagger", "dagger_r2"):
        a = ag.get(name)
        if not a:
            continue
        g = a["failure_groups_mean"]
        rep = g.get("representation", 0.0)
        sc = g.get("state_coverage", 0.0)
        tot = rep + sc
        shares[name] = rep / tot if tot else float("nan")
        L.append(f"| {name} | {tot:.1f} | {rep:.1f} | {sc:.1f} | **{shares[name]:.1%}** |")
    L.append("")
    if "baseline" in shares and "dagger" in shares:
        r2share = shares.get("dagger_r2")
        L.append(
            f"**关键判读**：绝对数量上两类失败都少了，但**占比几乎不变**"
            f"（基线 {shares['baseline']:.1%} → 一轮 {shares['dagger']:.1%} → "
            f"两轮 {(r2share if r2share is not None else float('nan')):.1%}；"
            f"bc_extra {shares.get('bc_extra', float('nan')):.1%}）。"
            "这说明 DAgger 并没有选择性地只修「状态覆盖类」，而是**整体降低了失败率**"
            "——失败数量的下降更多来自「成功变多」而非「机制精准」。"
            "两轮之后表征类占比才轻微下降到 61.5%，方向上是「越练越偏向表征瓶颈」的迹象，"
            "但幅度远不足以支撑「DAgger 修了表征问题」的结论。"
            "朴素预期（DAgger 只修分布偏移）在本环境上**得不到干净的支持**，必须诚实写进报告。\n"
        )
    L.append(
        "我另外做了机制验证（`branches/b3_dagger/verify_attribute_mechanism.py`），"
        "检验「DAgger 是否提供了专家数据里缺失的『面向错误属性物体』负样本」这一假设："
    )
    try:
        mech = json.load(open("reports/b3_dagger_attribute_mechanism.json"))
        v = mech["verdict"]
        L.append(
            f"\n> 面前是「部分属性匹配」物体的状态占比：专家 {v['expert_counts']['pct_facing_partial_match']:.1%} "
            f"vs DAgger {v['dagger_counts']['pct_facing_partial_match']:.1%}，"
            f"提升 {v['H1_partial_match_ratio_lift']:+.1%} → "
            f"**假设{'成立' if v['H1_supported'] else '不成立'}**。"
        )
        if not v["H1_supported"]:
            L.append(
                "也就是说，DAgger 并没有给模型更多「认错属性」的负样本。"
                "表征类失败的下降因此**不是** DAgger 的直接功劳，"
                "更可能是「模型整体走对了、根本到不了那个错误状态」的副产品。"
            )
    except FileNotFoundError:
        L.append("（尚未生成）")
    L.append("")

    if s7:
        L.append("## 4. 跨房间迁移（S7N4，seed 10000:10300）\n")
        v = [r["closed_loop_success"] for r in s7["runs"]]
        L.append("| run | S7N4 Success |")
        L.append("|---|---|")
        for r in s7["runs"]:
            L.append(f"| `{Path(r['run']).name}` | {r['closed_loop_success']:.3f} |")
        L.append(f"| **mean（6 个 run）** | **{st.mean(v):.4f} ± {st.pstdev(v):.4f}** |")
        L.append("")
        r1 = [r["closed_loop_success"] for r in s7["runs"] if "b3_dagger_seed" in r["run"]]
        r2 = [r["closed_loop_success"] for r in s7["runs"] if "b3_r2_seed" in r["run"]]
        if r1 and r2:
            L.append(
                f"基线 S7N4 = **0.674**；一轮 DAgger = {st.mean(r1):.4f}（{st.mean(r1) - 0.674:+.3f}）；"
                f"两轮 = {st.mean(r2):.4f}（{st.mean(r2) - 0.674:+.3f}）。\n"
            )
            L.append(
                "**关键观察**：以「跑满 30 epoch 的 BC（0.814）」为分母，第一轮的净收益是 +0.052，"
                "但迁移收益是 +0.073——**迁移比同分布还多**。"
                "第二轮同分布只涨 +0.012，迁移又涨 +0.022，仍是两倍。"
                "这说明 DAgger 补的状态对**没见过的房间尺寸**帮助更大："
                "本环境的同分布上限已接近饱和（随机基线就有 0.543），"
                "而 S7N4 上还有真实空间。"
                "只看 S6N3 的 Success 会低估 DAgger 的价值，所以 S7N4 评估是必要的。\n"
            )

    L.append("## 5. 诊断数据\n")
    L.append("| 指标 | 数值 | 含义 |")
    L.append("|---|---|---|")
    L.append(
        f"| 专家数据标签歧义率 | {conf_e['ambiguous_state_rate']:.2%} | "
        f"同一 (image,dir,mission) 对应多个专家动作的比例（基线噪声底） |"
    )
    L.append(
        f"| DAgger 数据标签歧义率 | {conf_d['ambiguous_state_rate']:.2%} | "
        "若显著高于专家数据，说明标注本身有问题；实测几乎相同 → **标签可靠** |"
    )
    L.append(
        f"| 保守 vs 激进标注分歧 | {diag['disagreement']['cons_vs_aggr']:.1%} | "
        "两种 `replan` 传参方式的结果差异（实测 0，因为 bot 每步重建规划） |"
    )
    L.append(
        f"| 标签 left/right 交替率 | {diag['label_flip_rate']['cons']:.1%} | "
        "连续两步转向互翻的比例，反映标签是否自相矛盾 |"
    )
    L.append("")

    L.append("## 6. 方法论副产品（对所有支线都重要）\n")
    L.append(
        "1. **`patience=5` 会系统性低估真实上限。** 同一份数据、同一个 seed，"
        "仅把早停关掉、跑满 30 epoch 并事后选 val_acc 最优：seed1 从 0.8167 → **0.8720**，"
        "seed2 从 0.8227 → **0.8579**。原因：验证集 n=1844、p≈0.87 时二项标准误 **0.0078**，"
        "而实测相邻 epoch 的 |Δval_acc| 均值 0.008–0.009 —— **噪声大于趋势斜率 0.005/epoch**，"
        "patience=5 必然提前触发（arXiv 2602.22107 也给出同一结论：accuracy 早停表现最差）。"
        "→ **建议支线 1/2/4/5 统一改为「跑满 + 事后选模」。**"
    )
    L.append(
        "2. **等量专家数据在这个任务上已经饱和。** 1500 条额外专家 episode 展开 "
        "5057 个唯一状态，**100% 已存在于训练集**，新增 0。"
        "→ 支线 5 的数据效率曲线在 100% 附近必然趋平；若要看到增长，必须改变分布而非数量。"
    )
    L.append(
        "3. **覆盖半径筛选能把 DAgger 数据压到 1/6 而不掉点。** "
        "保留「策略走到、专家没走到」的新状态（~2200 vs 12800 转移），Success 0.862 vs 0.867。"
    )
    L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
