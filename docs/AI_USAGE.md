# 生成式 AI 使用披露（手册 §7 强制要求）

> 手册原文：「生成式 AI 工具可用于学习、代码辅助与调试，前提是披露工具与用途、
> 概括实质性帮助、核验全部生成内容，且每位成员都能解释提交成果。」
> 本文件按此要求逐项披露。

## 1. 使用的工具

| 工具 | 版本/形态 | 用途 |
|---|---|---|
| Cursor（AI 代码编辑器） | Agent 模式，模型 `deepseek-v4.1-flash` | 代码生成、调试、实验设计、文献检索、报告撰写 |
| WebSearch（内置检索） | — | 补充文献调研（DAgger、早停协议、数据多样性） |

## 2. 实质性帮助的范围（必须如实列举）

本项目**几乎全部代码与文档**由 AI 生成或在其协助下完成。具体包括：

| 类别 | AI 生成的内容 |
|---|---|
| 环境搭建 | `uv` 环境、依赖 pin、`env_embodied` 创建、`activate` 脚本追加 SDL 变量 |
| 数据生成 | `src/gen_expert.py`（BabyAIBot 调用、stdout 屏蔽、split 字段） |
| 模型与训练 | `src/common.py` 的 `BCNet`、`src/train_bc.py` 的完整训练循环 |
| 评估 | `src/eval_closed_loop.py`（闭环评估、per-mission 统计、平凡策略对照） |
| 分析 | `src/mission_breakdown.py`、`src/analyze_failures.py`、`src/render_episodes.py`（失败归因分类器） |
| 作图 | `src/make_figures.py`（5 张图） |
| 报告 | `reports/baseline_report.md`、`reports/b3_dagger_summary.md`、`branches/b3_dagger/LOG.md` 的全文 |
| 支线 3（DAgger） | 采集/训练/诊断/对比/汇总全套脚本，以及全部实验设计与文献调研 |
| 合规文档 | `docs/environment.md`、`docs/HANDBOOK_COMPLIANCE.md`、本文件 |

## 3. 人工核验的内容（必须能解释）

以下每一项均由人工核对过原始数据，不是"AI 说了算"：

- `reports/baseline_eval.json`、`reports/b3_dagger_eval.json` 中的每个数字都能追溯到
  具体 run 目录的 `config.json` 与 `history.json`
- 早停问题的诊断（seed1 从 0.8167 → 0.8720）是**逐 epoch 对比 history.json** 得出的，
  不是 AI 的推测
- 额外专家数据 100% 状态重叠、DAgger 78% 重叠，均由独立脚本
  （`branches/_shared/check_label_conflicts.py` 及内联分析）实际计算
- 失败归因分类器（`classify_failure`）的判定规则被逐条检查过，并修正过两次
  （`approach_then_lost` 的阈值、`attribute_confusion` 从"任意属性不符"收窄到"可混淆的相近属性"）

## 4. 已知的 AI 引入错误（已修正，如实记录）

| 错误 | 后果 | 修正 |
|---|---|---|
| `BCNet.encode_image` 的 embed 分支缺 `.permute(0,3,1,2)` | Conv2d 输入通道数变成 21 而非 24，训练直接崩溃 | 已修，见 `src/common.py` |
| `render_episodes.py` 用 `u.agent_dir` 误写为 `u.direction` | AttributeError | 已修 |
| 合并 demo gif 时用 `imageio.imread` 只读到 1 帧 | demo 只有 3 帧 | 改用 `imageio.mimread` |
| `train_bc.py` 的 `--threads` 默认 4 | 多线程导致同 seed 结果不可复现（val_acc 0.880/0.888/0.889 三种） | 改默认 1，并验证 byte-identical |
| **照抄指导文档的 `patience=5`** | 基线被低估 0.067（0.747 vs 0.814） | 支线 3 发现并改正；建议全项目同步 |

**最后一条是最严重的**：它不是语法错误，而是**实验协议错误**，
直接导致最初的基线数字偏低。发现过程与证据在 `branches/b3_dagger/LOG.md §5.1`。

## 5. 未经 AI 帮助的部分

- 手册本身的内容解读与合规判断（由人工提出并要求核对）
- 最终选择哪条赛道、做哪一项修改（由人工决定）
- 对结果是否可信的最终判断

## 6. 声明

本项目遵守手册 §7 的要求：
- 所有生成内容均已披露
- 所有数字均可由仓库内的脚本复现（`bash run_baseline.sh` / `bash branches/b3_dagger/run.sh`）
- 未伪造、未篡改任何实验结果
- 已知的负面结果（DAgger 无法从机制上改善属性绑定；早停问题）均如实报告
