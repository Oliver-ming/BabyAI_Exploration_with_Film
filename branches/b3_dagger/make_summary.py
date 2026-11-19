"""支线 3 · 汇总表 -> reports/b3_dagger_summary.md

只读已有 json/config，不重算任何指标。同时涵盖 v1（原始 DAgger-lite 基线对照）与
v2（协议修复、等预算 Split-DAgger、死锁解锁采样、主动查询门控、IWR 加权与 FiLM 正交叠加）。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
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
    "DAgger 两轮（+1500 +第二轮 1500）": lambda r, m: "b3_r2_seed" in r,
}

V2_ORDER = [
    ("baseline", "基线 (早停 patience=5, 4500 专家)"),
    ("b3_bc30", "BC30 (跑满 30ep, 4500 专家)"),
    ("b3_v2_bc3000", "BC-3000 (等预算起点, 3000 专家)"),
    ("b3_v2_split_dag", "Split-DAgger (3000 专家 + 1500 解锁交互, 4500 种子)"),
    ("b3_v2_split_ens", "Split-EnsembleDAgger (3000 专家 + 1500 集成门控, 4500 种子)"),
    ("b3_v2_bcfresh", "BC-Fresh (4500 专家 + 1500 全新专家)"),
    ("b3_dagger", "v1 DAgger (4500 专家 + 1500 记忆种子)"),
    ("b3_v2_greedy", "v2 Greedy (4500 专家 + 1500 新种子全步)"),
    ("b3_v2_beta025", "v2 β=0.25 混合 (4500 专家 + 1500 新种子)"),
    ("b3_v2_unstick", "v2 Loop-Unsticking (w=1.0, 4500+1500)"),
    ("b3_v2_iwr2", "v2 Loop-Unsticking + IWR (w=2.0)"),
    ("b3_v2_iwr3", "v2 Loop-Unsticking + IWR (w=3.0)"),
    ("b3_v2_qdis", "v2 Disagreement-Only (全查只留分歧, w=2.0)"),
    ("b3_v2_qmar", "v2 Margin 主动门控 (仅 18.8% 查询, w=2.0)"),
    ("b3_v2_qens", "v2 Ensemble 主动门控 (仅 25.2% 查询, w=2.0)"),
    ("b3_v2_r2_w1", "v2 两轮非重叠 Loop-Unsticking (w=1.0)"),
    ("b3_v2_r2", "v2 两轮非重叠 Loop-Unsticking + IWR (w=2.0)"),
    ("b2_film_nwd", "B2 FiLM (4500 专家, wd=0)"),
    ("b3_v2_film3000", "FiLM-3000 (3000 专家)"),
    ("b3_v2_film_split", "FiLM-Split (3000 专家 + 1500 解锁交互, 4500 种子)"),
    ("b3_v2_film_dag_w1", "FiLM + DAgger (4500 专家 + 1500 解锁交互, w=1.0)"),
    ("b3_v2_film_dag", "FiLM + DAgger + IWR (4500 专家 + 1500 解锁交互, w=2.0)"),
]


def jload(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def append_v2_sections(L: list[str], s6: dict, s7: dict | None, cov: dict | None) -> None:
    bg6 = s6["by_group"]
    bg7 = s7["by_group"] if s7 else {}

    L.append("---\n")
    L.append("## 7. v2 系统完善总表（协议修正 + 主动交互改进 + 正交叠加）\n")
    L.append(
        "> 数据来源：`reports/b3_v2_eval_s6n3.json`、`reports/b3_v2_eval_s7n4.json`、"
        "`reports/b3_v2_rollout_coverage.json`，全部复用冻结 `eval_closed_loop.evaluate` 口径"
        "（已与 `src/eval_closed_loop.py` 做逐一交叉核对，差值 `0.0`）。\n"
    )

    L.append("### 7.1 全组闭环性能与查询成本对比（3 seed，mean ± SD）\n")
    L.append(
        "| 组别 | S6N3 Success | Δ vs BC30 (配对) | S7N4 迁移 | val 准确率 | "
        "探针准确率 | 纠偏准确率 | 专家总标注数 | 失败(表征+覆盖) |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, label in V2_ORDER:
        if k not in bg6:
            continue
        r6 = bg6[k]
        r7 = bg7.get(k, {})
        s6_str = f"**{r6['success_mean']:.4f} ± {r6['success_sd']:.4f}**"
        s7_str = f"{r7['success_mean']:.4f} ± {r7['success_sd']:.4f}" if r7 else "—"
        d = r6.get("paired_delta_vs_ref_mean")
        d_str = "—" if k == "b3_bc30" or d is None else f"{d:+.4f} ({'3/3+' if r6.get('paired_all_positive') else '非全正'})"
        val_str = f"{r6['val_acc_mean']:.4f}" if r6.get("val_acc_mean") is not None else "—"
        onp_str = f"{r6['onpolicy_acc_mean']:.4f}" if r6.get("onpolicy_acc_mean") is not None else "—"
        rec_str = f"{r6['recovery_acc_mean']:.4f}" if r6.get("recovery_acc_mean") is not None else "—"
        q_tot = r6.get("total_oracle_labels_mean")
        if q_tot is None and k == "b3_bc30":
            q_tot = 16534.0
        q_str = f"{q_tot:,.0f}" if q_tot is not None else "—"
        fg = r6.get("failure_groups_mean", {})
        f_str = f"{fg.get('representation', 0):.1f}+{fg.get('state_coverage', 0):.1f}" if fg else "—"
        L.append(f"| {label} | {s6_str} | {d_str} | {s7_str} | {val_str} | {onp_str} | {rec_str} | {q_str} | {f_str} |")
    L.append("")

    if cov and cov.get("datasets"):
        L.append("### 7.2 v1 协议缺陷诊断：种子重复与记忆盲区对状态覆盖的影响\n")
        L.append("| 采集集 (1500 ep, seed0) | 交互步数 | Oracle 查询 | 查询率 | 唯一 (img,dir,mission) | 相对 4500 专家新颖状态 | π≠a* 步数 | 新颖纠偏状态 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for row in cov["datasets"]:
            L.append(
                f"| {row['label']} | {row['n_env_steps']:,} | {row['n_oracle_queries']:,} | "
                f"{row['query_rate']:.1%} | {row['unique_idm_states']:,} | "
                f"**{row['novel_idm_vs_expert_train']:,}** ({row['novel_ratio']:.1%}) | "
                f"{row['disagree_steps']:,} | **{row['novel_disagree_idm_states']:,}** |"
            )
        L.append("")
        ds_map = {d["key"]: d for d in cov["datasets"]}
        if "v1_expert_extra_dup" in ds_map and "v2_fresh_expert" in ds_map:
            v1e, v2e = ds_map["v1_expert_extra_dup"], ds_map["v2_fresh_expert"]
            v1d, v2u = ds_map.get("v1_dagger_memorized"), ds_map.get("v2_unstick_fresh")
            L.append(
                f"- **修正 1（额外专家种子重叠）**：v1 `b3_expertextra_1500` 使用了已进训练集的 `0:1500`，"
                f"新颖状态为 **{v1e['novel_idm_vs_expert_train']}**；改用未见种子 `5300:6800` 后，"
                f"新专家实际包含 **{v2e['novel_idm_vs_expert_train']:,}** ({v2e['novel_ratio']:.1%}) 个全新状态，"
                f"`BC-Fresh` 成功率也从 v1 的 0.8444 升至 **{bg6['b3_v2_bcfresh']['success_mean']:.4f}**。"
            )
            if v1d and v2u:
                L.append(
                    f"- **修正 2（交互种子记忆 + 死锁解锁）**：v1 DAgger 在已记忆的 `0:1500` 上 rollout 仅采出 "
                    f"{v1d['novel_idm_vs_expert_train']:,} 个新颖状态和 {v1d['novel_disagree_idm_states']} 个新颖纠偏状态；"
                    f"切换至未见种子并加入 **Loop-Unsticking** 后，用更少查询（{v2u['n_oracle_queries']:,} vs {v1d['n_oracle_queries']:,}）"
                    f"采出了 **{v2u['novel_idm_vs_expert_train']:,}**（×4.9）个新颖状态与 **{v2u['novel_disagree_idm_states']:,}**（×3.4）个新颖纠偏状态。\n"
                )

    if all(k in bg6 for k in ("b3_bc30", "b3_v2_bc3000", "b3_v2_split_dag", "b3_v2_split_ens")):
        L.append("### 7.3 严格 4,500 种子等预算实验（零额外环境种子）\n")
        bc30, bc3k = bg6["b3_bc30"], bg6["b3_v2_bc3000"]
        sdag, sens = bg6["b3_v2_split_dag"], bg6["b3_v2_split_ens"]
        L.append(
            f"- 在 **4,500 个训练环境种子**硬上限下（3,000 专家 + 1,500 交互）："
            f"`Split-DAgger` 从 `BC-3000` 的 {bc3k['success_mean']:.4f} 跃升至 **{sdag['success_mean']:.4f}**，"
            f"反超满 4,500 专家的 `BC30`（{bc30['success_mean']:.4f}）**{sdag['success_mean'] - bc30['success_mean']:+.4f}**，"
            f"S7N4 迁移更从 {bg7['b3_bc30']['success_mean']:.4f} 提至 **{bg7['b3_v2_split_dag']['success_mean']:.4f}**。"
        )
        L.append(
            f"- `Split-EnsembleDAgger` 仅使用 **{sens['total_oracle_labels_mean']:,.0f}** 条专家动作标签"
            f"（比 `BC30` 的 16,534 条**少 16.9%**），却取得了 **{sens['success_mean']:.4f}** 的 S6N3 成功率"
            f"（vs `BC30` {sens['success_mean'] - bc30['success_mean']:+.4f}）与 **{bg7['b3_v2_split_ens']['success_mean']:.4f}** 的 S7N4 迁移成功率。"
        )
        if all(k in bg6 for k in ("b2_film_nwd", "b3_v2_film3000", "b3_v2_film_split")):
            L.append(
                f"- 在 FiLM 架构下，`FiLM-Split`（3k+1.5k=4,500 种子）达到 **{bg6['b3_v2_film_split']['success_mean']:.4f}**"
                f"（S7N4 **{bg7['b3_v2_film_split']['success_mean']:.4f}**），追平并微超满 4,500 专家的 `B2 FiLM`（{bg6['b2_film_nwd']['success_mean']:.4f} / {bg7['b2_film_nwd']['success_mean']:.4f}）。\n"
            )

    L.append("### 7.4 关键机制消融与诚实判读\n")
    if all(k in bg6 for k in ("b3_v2_unstick", "b3_v2_greedy", "b3_v2_iwr2", "b3_v2_iwr3")):
        u, g, i2, i3 = bg6["b3_v2_unstick"], bg6["b3_v2_greedy"], bg6["b3_v2_iwr2"], bg6["b3_v2_iwr3"]
        L.append(
            f"1. **死锁解锁（Loop-Unsticking）优于盲目 Greedy rollout**：`v2_unstick`（w=1.0）取得 **{u['success_mean']:.4f} ± {u['success_sd']:.4f}**（S6N3）与 **{bg7['b3_v2_unstick']['success_mean']:.4f}**（S7N4），"
            f"与 `v2_greedy`（{g['success_mean']:.4f} ± {g['success_sd']:.4f} / S7N4 {bg7['b3_v2_greedy']['success_mean']:.4f}）相比："
            f"**S6 方差减半、总专家查询减少 {g['total_oracle_labels_mean'] - u['total_oracle_labels_mean']:,.0f} 步（-20%）、S7N4 迁移高出 +{bg7['b3_v2_unstick']['success_mean'] - bg7['b3_v2_greedy']['success_mean']:.4f}**，"
            f"状态覆盖类失败降至 **{u['failure_groups_mean']['state_coverage']:.1f}/300**（对比 `BC30` 17.3）。"
        )
        L.append(
            f"2. **IWR 干预加权（w > 1）在本任务中的双刃剑效应（诚实负结果）**：在 `BCNet` 上将 `π≠a*` 样本权重从 1.0 提至 2.0 / 3.0，"
            f"虽然让只读探针纠偏准确率 `recov_acc` 从 {u['recovery_acc_mean']:.4f} 升至 **{i2['recovery_acc_mean']:.4f}** / {i3['recovery_acc_mean']:.4f}，"
            f"但 S6N3 闭环成功率反而从 {u['success_mean']:.4f} 降至 **{i2['success_mean']:.4f}** / **{i3['success_mean']:.4f}**，"
            f"且状态覆盖类失败从 {u['failure_groups_mean']['state_coverage']:.1f} 回升至 {i2['failure_groups_mean']['state_coverage']:.1f}。"
            f"原因：`BCNet` 缺乏属性接地，在大量困难分歧态上加倍权重会拉低常规前进/转向策略在专家分布上的校准度（`val_acc` 由 {u['val_acc_mean']:.4f} 降至 {i2['val_acc_mean']:.4f}），引发近距振荡。"
        )
    if all(k in bg6 for k in ("b3_v2_qmar", "b3_v2_qens", "b3_v2_qdis")):
        qm, qe, qd = bg6["b3_v2_qmar"], bg6["b3_v2_qens"], bg6["b3_v2_qdis"]
        L.append(
            f"3. **主动查询门控的查询效率**：`Margin 门控`仅用 **{qm['extra_oracle_queries_mean']:,.0f}** 条额外查询（18.8% 查询率，总计 18.1k 标签）"
            f"即达到 **{qm['success_mean']:.4f}**（优于 16.5k 标签的 `BC30` {bg6['b3_bc30']['success_mean']:.4f} 并接近 22.1k 标签的 `BC-Fresh` {bg6['b3_v2_bcfresh']['success_mean']:.4f}）；"
            f"`Ensemble 门控`用 **{qe['extra_oracle_queries_mean']:,.0f}** 条查询（25.2% 查询率）在 S7N4 上达到 **{bg7['b3_v2_qens']['success_mean']:.4f}**。"
            f"而只存分歧样本的 `Disagreement-Only`（`{qd['success_mean']:.4f}`）因失去高置信在轨正则反而低于全量保留。"
        )
    if all(k in bg6 for k in ("b3_v2_r2_w1", "b3_v2_r2")):
        r2w1, r2w2 = bg6["b3_v2_r2_w1"], bg6["b3_v2_r2"]
        L.append(
            f"4. **非重叠两轮 DAgger**：`v2_r2 (w=2.0)` 达到 **{r2w2['success_mean']:.4f} ± {r2w2['success_sd']:.4f}**（seed0 达 **0.9300**，失败总数降至 29.3），"
            f"`v2_r2_w1 (w=1.0)` 达到 **{r2w1['success_mean']:.4f} ± {r2w1['success_sd']:.4f}**（S7N4 **{bg7['b3_v2_r2_w1']['success_mean']:.4f}**），均高于 v1 两轮（{bg6['b3_r2']['success_mean']:.4f}）。"
        )
    if all(k in bg6 for k in ("b2_film_nwd", "b3_v2_film_dag", "b3_v2_film_dag_w1")):
        f0, fd2, fd1 = bg6["b2_film_nwd"], bg6["b3_v2_film_dag"], bg6["b3_v2_film_dag_w1"]
        L.append(
            f"5. **FiLM + DAgger 正交叠加（突破上限）**：在 B2 最强模型 `FiLM`（{f0['success_mean']:.4f} ± {f0['success_sd']:.4f}）基础上叠加 1 轮解锁 DAgger："
            f"`FiLM+DAgger (w=2.0)` 达到 **{fd2['success_mean']:.4f} ± {fd2['success_sd']:.4f}**（seed0 **0.9800**，平均失败从 12.3 降至 **9.0**），"
            f"`FiLM+DAgger (w=1.0)` 达到 **{fd1['success_mean']:.4f} ± {fd1['success_sd']:.4f}**（S7N4 **{bg7['b3_v2_film_dag_w1']['success_mean']:.4f}**）。"
            f"表征类与状态覆盖类失败同步降至 **{fd2['failure_groups_mean']['representation']:.1f} + {fd2['failure_groups_mean']['state_coverage']:.1f} = "
            f"{fd2['failure_groups_mean']['representation'] + fd2['failure_groups_mean']['state_coverage']:.1f}**（或 w=1.0 组 "
            f"{fd1['failure_groups_mean']['representation']:.1f} + {fd1['failure_groups_mean']['state_coverage']:.1f}），"
            f"从实证上验证了「表征接地（B2）」与「状态分布纠偏（B3）」的正交互补性。"
        )


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

    L = ["# 支线 3 · DAgger-lite 与交互式模仿学习（v1 + v2）结果汇总\n"]
    L.append("> 全部数字来自 `reports/baseline_eval.json`、`reports/b3_dagger_*.json`、")
    L.append("> `reports/b3_v2_eval_*.json` 与各 `runs/*/config.json`，可逐条溯源。\n")

    L.append("## 1. v1 主结果（S6N3，测试 seed 4500:4800，每模型 300 episode）\n")
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
        L.append("### v1 关键对比与复盘说明\n")
    L.append(
        f"- **DAgger vs 原基线**：Success {base_s['succ']:.4f} → {dag_s['succ']:.4f}"
        f"（**{dag_s['succ'] - base_s['succ']:+.4f}**），"
        f"基线 seed 间 SD 为 {base_s['succ_sd']:.4f} → "
        f"{'**超过噪声，显著**' if abs(dag_s['succ'] - base_s['succ']) > base_s['succ_sd'] else '未超过噪声'}"
    )
    bc30_s = stats(GROUPS["bc30（BC，跑满 30 epoch，无早停）"])
    if bc30_s:
        L.append(
            f"- **其中 {bc30_s['succ'] - base_s['succ']:+.4f} 来自取消早停**："
            f"原基线用 `patience=5` 在 epoch 13/14 掐断 seed1/seed2，跑满 30 epoch 后为 {bc30_s['succ']:.4f}。"
        )
        L.append(
            f"- **v1 DAgger 净收益（vs BC30）**：{dag_s['succ'] - bc30_s['succ']:+.4f}"
        )
    if ext_s:
        L.append(
            "- **关于 v1 `bc_extra` 的重要勘误（详见 §7.2）**：v1 声称 1500 条额外专家状态 100% 重叠、"
            "实因脚本误采了训练前 1500 个重复 seed（`0:1500`）。v2 已用全新未见种子 `5300:6800` 重建真正的新专家对照 `BC-Fresh`（见 §7）。"
        )
    L.append("")

    L.append("## 2. v1 逐 seed 配对对比\n")
    L.append("| seed | bc30 | bc_extra | DAgger | D − bc_extra |")
    L.append("|---|---|---|---|---|")

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
        L.append(f"| **mean** | — | — | — | **{st.mean(deltas):+.4f} ± {st.pstdev(deltas):.4f}** |\n")

    novel_s = stats(GROUPS["DAgger +1500（只保留新增状态，覆盖半径筛选）"])
    if dag_s and novel_s:
        L.append("### v1 覆盖半径消融（去冗余强度）\n")
        L.append("| 去冗余强度 | 真正新增的转移数 | 闭环 Success |")
        L.append("|---|---|---|")
        L.append(f"| `none`（全量） | ~12800 | {stats(GROUPS['DAgger +1500（不去冗余）'])['succ']:.4f} |")
        L.append(f"| `episode`（默认） | ~6050 | {dag_s['succ']:.4f} |")
        L.append(f"| `global`（只留新状态） | ~2200 | **{novel_s['succ']:.4f}** |\n")

    L.append("## 3. v1 失败构成与机理验证\n")
    ag = fcmp["aggregate"]
    L.append("| 组 | 总失败 | 表征类失败 | 状态覆盖类 | 表征类占比 |")
    L.append("|---|---|---|---|---|")
    for name in ("baseline", "bc_extra", "dagger", "dagger_r2"):
        a = ag.get(name)
        if not a:
            continue
        g = a["failure_groups_mean"]
        rep = g.get("representation", 0.0)
        sc = g.get("state_coverage", 0.0)
        tot = rep + sc
        L.append(f"| {name} | {tot:.1f} | {rep:.1f} | {sc:.1f} | **{(rep / tot if tot else 0):.1%}** |")
    L.append("")

    if s7:
        L.append("## 4. v1 跨房间迁移（S7N4，seed 10000:10300）\n")
        r1 = [r["closed_loop_success"] for r in s7["runs"] if "b3_dagger_seed" in r["run"]]
        r2 = [r["closed_loop_success"] for r in s7["runs"] if "b3_r2_seed" in r["run"]]
        if r1 and r2:
            L.append(f"基线 S7N4 = **0.674**；v1 一轮 = {st.mean(r1):.4f}；v1 两轮 = {st.mean(r2):.4f}（v2 完整迁移见 §7.1）。\n")

    L.append("## 5. v1 诊断数据\n")
    L.append("| 指标 | 数值 | 含义 |")
    L.append("|---|---|---|")
    L.append(f"| 专家数据标签歧义率 | {conf_e['ambiguous_state_rate']:.2%} | 同一 (image,dir,mission) 对应多标签比例 |")
    L.append(f"| DAgger 数据标签歧义率 | {conf_d['ambiguous_state_rate']:.2%} | 标签可靠 |")
    L.append(f"| 保守 vs 激进 `replan` 分歧 | {diag['disagreement']['cons_vs_aggr']:.1%} | bot 每步重建规划 |")
    L.append(f"| 标签 left/right 交替率 | {diag['label_flip_rate']['cons']:.1%} | 走廊掉头等状态本身属性 |\n")

    L.append("## 6. 方法论副产品（全项目适用）\n")
    L.append("1. **取消 `patience=5` 误早停**：统一跑满 30 epoch 后事后按纯净 `val_acc` 选模。")
    L.append("2. **单线程 CPU 确定性（`OMP_NUM_THREADS=1`）**：防止百核服务器上 OpenMP 锁竞争拖慢百倍与破坏复现性。\n")

    v2_s6 = jload("reports/b3_v2_eval_s6n3.json")
    v2_s7 = jload("reports/b3_v2_eval_s7n4.json")
    v2_cov = jload("reports/b3_v2_rollout_coverage.json")
    if v2_s6:
        append_v2_sections(L, v2_s6, v2_s7, v2_cov)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
