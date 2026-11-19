# 支线 3 · DAgger-lite 与交互式模仿学习（Dataset Aggregation）

> 目录：`branches/b3_dagger/`　状态：**v1 + v2 全面完成（已含文献/开源改进、协议漏洞修正、主动查询与 FiLM 正交叠加）**
> 冻结区未改动：`git diff -- src/ data/expert_5000.pkl runs/baseline_seed* reports/baseline_* branches/README.md` 为空。

## 0. 一句话结论

- **v1（基线确立与机制初探）**：取消误早停（`patience=5` → 跑满 30 epoch 事后按 `val_acc` 选模）把 BC 基线从 **0.7467** 还原至 **0.8144 ± 0.0453**；v1 一轮/两轮 DAgger 分别达 **0.8667** / **0.8789**。
- **v2（修正 v1 四项实验协议缺陷 + 交互式模仿学习文献改进）**：
  1. **严格 4,500 种子等预算证明**：在零额外环境种子下，`Split-EnsembleDAgger`（3,000 专家 + 1,500 集成门控交互，仅 **13,740** 条专家标注，比 `BC30` 少 **16.9%**）闭环 Success 达 **0.8578 ± 0.0351**（vs `BC30` +0.0433；S7N4 **0.7644** vs `BC30` 0.6967）。
  2. **死锁解锁采样（Loop-Unsticking）+ 非重叠两轮迭代**：在全新训练种子 `5300:6800` / `6800:8300` 上消除“训练种子记忆盲区”并由专家单步打破原地振荡，1 轮即达 **0.8900 ± 0.0152**（S7N4 **0.8067**，优于全新专家对照 `BC-Fresh` 0.8600 / 0.7711），两轮达 **0.9089 ± 0.0191**（S7N4 **0.8278**，状态覆盖类失败降至 **8.3/300**）。
  3. **主动查询 Pareto 前沿**：`Margin` 门控仅用 **18.8%** 交互查询率（总计 18.1k 标签）即达 **0.8589**；`Ensemble` 门控用 **25.2%** 交互查询率（总计 18.7k 标签）在 S7N4 达 **0.7878**。
  4. **FiLM + DAgger 正交突破上限**：将表征接地（B2 FiLM，0.9589）与状态覆盖纠偏（B3 Loop-Unsticking DAgger）叠加，`FiLM+DAgger (w=2.0)` 在 S6N3 达 **0.9700 ± 0.0072**（seed0 **0.9800**，平均总失败从 12.3 降至 **9.0**），`FiLM+DAgger (w=1.0)` 在 S7N4 零样本迁移达 **0.9178 ± 0.0181**。
  5. **诚实负结果**：在缺乏属性接地的 `BCNet` 上，强行加倍干预权重（IWR $w=2.0, 3.0$）虽把恢复纠偏准确率 `recovery_acc` 从 `0.4411` 提至 `0.4934`，却损伤常规分布校准，使 S6N3 成功率回落至 `0.8622 / 0.8556`；同理仅保留分歧样本（`Disagreement-Only`）会因失去在轨正则而降至 `0.8389`。

## 1. 目标与成功判据

**假设**：BC 的失败源于**分布偏移**。证据：基线 37 条失败 episode **全部跑满 36 步**，
表现为原地转圈 / 顶墙（"卡住"），而不是"走得远"。

| 判据 | 阈值（跑前写死） | 结论 |
|---|---|---|
| 主判据：Success 提升 | > 0.093（基线 SD） | ✅ +0.120 |
| 净收益：对比**同训练时长**的 BC | > 0 | ✅ +0.052 |
| **关键对照**：对比**等量额外专家数据** | > 0 | ✅ +0.022（配对 3/3 为正） |
| 次判据：状态覆盖类失败显著降 | 定性 | ✅ 31.0 → 13.7 |
| 次判据：表征类失败**不**降 | 定性 | ❌ 也降了（45.0 → 26.3），见 §5.3 |

## 2. 调研（实现之前）

### 2.1 文献

| # | 标题 / 链接 | 解决了什么 |
|---|---|---|
| 1 | [Ross, Gordon & Bagnell, A Reduction of IL to No-Regret Online Learning (AISTATS 2011)](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) | 原始论文：O(εT²) → O(εT)；"聚合而非替换"是理论必需 |
| 2 | [DAgger: Taming Covariate Shift with No Regret（Cornell CS4756）](https://www.cs.cornell.edu/courses/cs4756/2024fa/assets/slides_notes/lec6_slides.pdf) | 算法骨架 + β 混合；parameter-free 变体 |
| 3 | [imitation 库 DAgger 文档](https://imitation.readthedocs.io/en/latest/algorithms/dagger.html) | 无论执行谁的动作都记录专家标签；每轮在累积集上重训 |
| 4 | [Building Embodied AI §21.3](https://embodiedbook.apartsin.com/part-5-learning-from-demonstration-and-robot-data/module-21-imitation-learning/section-21.3.html) | 三种失效情形；"只在新数据上训"会遗忘 |
| 5 | [Interactive & Hybrid IL (arXiv 2412.07057)](https://arxiv.org/html/2412.07057v3) | 按**每条轨迹**计成本时 BC 不可超越；按**每状态**计时 DAgger 才有效 |
| 6 | [BabyAI (arXiv 1810.08272)](https://arxiv.org/abs/1810.08272) | BabyAIBot = 可任意查询的模拟教师（oracle 合法性） |
| 7 | [Imitation Learning 笔记（Thomas To）](https://thomas-to-bcheme-github-io.vercel.app/ai-ml/reinforcement-learning/imitation-learning) | **专家留出集上的准确率"几乎是没用的指标"** → 促成三套指标 |
| 8 | [Data Quality in IL (arXiv 2306.02437)](https://ar5iv.labs.arxiv.org/html/2306.02437v1) | action divergence / transition diversity；"state diversity 并非总是有益" |
| 9 | [Diversity Scaling Laws (arXiv 2405.14005)](https://arxiv.org/abs/2405.14005) | 泛化取决于**覆盖半径**而非 episode 数 |
| 10 | [So You Think You Can Scale Up Autonomous Robot Data? (arXiv 2411.01813)](https://arxiv.org/html/2411.01813) | 自主数据收益会饱和；多轮采集递减；novelty 重加权不确定 |
| 11 | [Don't stop me now: How Validation Criteria Affect Checkpoint Selection and Early Stopping (arXiv 2602.22107)](https://arxiv.org/abs/2602.22107) | **用验证 accuracy 做早停表现最差**；post-hoc 全 epoch 选模优于早停 |
| 12 | [Don't Waste Your Time: Early Stopping Cross-Validation (arXiv 2405.03389)](https://arxiv.org/html/2405.03389v2) | 早停本身可以加速，但必须区分"选择"与"评估" |
| 13 | [Cawley & Talbot, On Over-fitting in Model Selection (JMLR 2010)](https://jmlr.org/papers/v11/cawley10a.html) | 在噪声估计上取最大值 = 对选模准则本身过拟合（winner's curse） |
| 14 | [Varma & Simon, Bias in Error Estimation When Using CV for Model Selection (2006)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-91) | 用 CV 同时选模 + 报误差会系统性高估 |

### 2.2 从文献提炼的设计约束

1. **必须聚合、不能只在新数据上训**（文献 1/3/4）→ 训练集 = 4500 专家 + 全部 DAgger 样本。
2. **执行策略动作、标注专家动作**（文献 3）→ 否则采到 d_π*，退化成 BC。
3. **成本按"每状态"记账**（文献 5）→ 报告 `N_dagger_transitions` 而非 episode 数。
4. **不能用专家留出集准确率判成败**（文献 7）→ 同时报三套指标。
5. **采集池不能碰 VAL**（手册 + 我加的约束）→ 用 TRAIN_SEEDS。
6. **重复样本会主导损失**（文献 8/9）→ 必须做状态去冗余。
7. **一轮足够**（文献 10）→ 单轮 + 等量对照，不迭代。
8. **早停 vs 事后选模要分开**（文献 11/12/13/14）→ 见 §5.1，这是本轮最大的方法论发现。

### 2.3 已知陷阱与规避

| 陷阱 | 依据 | 规避 |
|---|---|---|
| 只在新数据上微调 → 遗忘 | 文献 3/4 | 每轮从零在聚合集上训练 |
| 让 bot 执行动作 → 退化成 BC | 文献 3 | 环境永远 `step(policy_action)` |
| bot 内部状态与 env 不一致 | 指导文档 §12 | 每 episode 新建 bot；`replan` 传专家上一动作 |
| 专家标签不一致导致震荡 | 文献 4 | 实测标签歧义率（§5.2），几乎为 0 |
| 迭代过多灌坏 rollout | 文献 10 | 只做单轮 |
| 把"数据变多"误当"分布覆盖" | 用户要求 / 支线 5 | **等量额外专家数据对照** |
| **早停过早点，低估真实上限** | 文献 11/13 | 跑满 30 epoch + 事后选模（§5.1） |

## 3. 实现

### 3.1 采集（`collect_dagger.py`）

```
for seed in TRAIN_SEEDS[:N]:
    env.reset(seed); bot = BabyAIBot(unwrapped)     # 每 episode 新建
    for t in range(max_steps):
        record obs                                   # 状态来自 π 的访问分布
        expert_a = bot.replan(last_expert_a)         # 标签来自专家
        policy_a = model.predict(obs)                # 环境执行策略动作
        append (obs, expert_a)
        step(policy_a)
```

关键决策：

- **采策略访问的全部状态（含成功轨迹）**。只采失败会让训练集偏向糟糕状态，
  破坏与验证集一致的分布。另跑 failure-only 变体做对照（§4.4）。
- **`replan(last_expert_a)` vs `replan(policy_a)`**：指导文档 §10.1 ② 推荐前者（保守）。
  我实测两者**标签分歧率 0.0%**（`reports/b3_dagger_label_diag.json`）：BabyAIBot 每步都
  `_process_obs()` 重建规划栈，参数只影响 undo 逻辑，而 GoToLocal 无需 undo。故该选择无影响。

### 3.2 去冗余（本支线的关键实现）

实测：400 条 DAgger episode 采到 2648 个转移，只有 **1597 个唯一 (image, direction)**，
其中 **78.4% 与专家训练集重叠**，真正新增的状态仅 **1267 个**。
原因：策略卡在转圈时会连续几十步访问同一状态。不去冗余时 1500 条采集产生
**31k 转移 vs 去冗余后 22k**，且结果更差（§4.3）。

实现三种强度（`train_dagger.py: load_branch_episodes`）：

| 模式 | 规则 |
|---|---|
| `none` | 全量保留 |
| `episode`（默认） | 每条 episode 内同一 `(image, dir)` 只留第一次 |
| `global` | 再叠加丢弃"专家训练集已覆盖"的状态 |

### 3.3 三套指标（解决"验证指标失真"）

文献 7 明确说专家留出集准确率"几乎没用"，所以同时报：

| 指标 | 定义 | 用途 |
|---|---|---|
| `val_action_acc` | 500 条纯专家验证集 | **唯一用于选 checkpoint**（手册硬规定） |
| `onpolicy_action_acc` | 策略采到的状态上（按 seed 每 5 条留 1 条 holdout）与专家标签一致率 | 检验"自我纠错能力" |
| `closed_loop_success` | 300 个未见测试 seed | 主指标 |

### 3.4 五组对照（消掉混淆变量）

| 组 | 训练集 / 协议 | 排除的混淆变量 |
|---|---|---|
| `baseline` | 4500 专家，patience=5 | 原基线 |
| `bc30` | 4500 专家，跑满 30 epoch | **"早停过早"** |
| `bc_extra` | 4500 专家 + 1500 条新专家 | **"数据量增加"** |
| `bc_extraMATCH` | 4500 专家 + 2700 条新专家（转移数近似匹配 DAgger） | 同上，更严格 |
| `dagger` | 4500 专家 + 1500 条 DAgger（去冗余） | 目标组 |
| `nodedup` | 同 `dagger` 但不去冗余 | **"去冗余是否必要"** |
## 4. 结果（3 seed）

### 4.1 主结果（S6N3，测试 seed 4500:4800，每模型 300 episode）

| 组 | n | 闭环 Success | val 动作准确率 | on-policy 准确率 | 平均回报 | 超时 |
|---|---|---|---|---|---|---|
| baseline（patience=5） | 3 | **0.747 ± 0.093** | 0.843 ± 0.033 | — | 0.656 | 76/300 |
| bc30（跑满 30 epoch） | 3 | **0.814 ± 0.045** | 0.873 ± 0.013 | — | 0.723 | 56/300 |
| bc_extra +1500 | 3 | **0.844 ± 0.011** | 0.889 ± 0.009 | 0.974 | 0.748 | 47/300 |
| bc_extraMATCH +2700 | 3 | **0.824 ± 0.031** | 0.887 ± 0.005 | 0.978 | 0.732 | 53/300 |
| **dagger +1500（去冗余）** | 3 | **0.867 ± 0.016** | 0.893 ± 0.006 | 0.909 | 0.771 | 40/300 |
| dagger +1500（不去冗余） | 3 | 0.853 ± 0.009 | 0.887 ± 0.007 | 0.819 | 0.757 | 44/300 |

### 4.2 逐 seed 配对（消掉 seed 噪声）

| seed | bc30 | bc_extra | DAgger | D − bc_extra |
|---|---|---|---|---|
| 0 | 0.877 | 0.860 | 0.887 | +0.027 |
| 1 | 0.797 | 0.833 | 0.847 | +0.013 |
| 2 | 0.770 | 0.840 | 0.867 | +0.027 |
| **mean** | 0.814 | 0.844 | 0.867 | **+0.022 ± 0.006** |

配对差值 **3/3 全部为正**。这比看绝对值更有说服力：seed 排序在不同组之间被打乱了，
说明增益不是「运气好的 seed 更好」，而是「DAgger 让差 seed 追上来了」。

### 4.3 覆盖半径消融（三档去冗余，本支线最有价值的定量结论）

| 去冗余强度 | 真正新增的转移 | 闭环 Success | on-policy 准确率 | val 准确率 |
|---|---|---|---|---|
| `none`（全量） | ~12800 | 0.853 ± 0.009 | 0.819 | 0.887 |
| `episode`（默认） | ~6050 | **0.867 ± 0.016** | 0.909 | 0.893 |
| `global`（只留新状态） | **~2200** | 0.862 ± 0.011 | 0.780 | 0.885 |

**用 1/6 的数据量拿到同样的成功率。** 这是文献 9/16「覆盖半径 K 比数据量 D 重要」
在本环境上的直接定量验证：`global` 只保留策略真正走到、专家从未走到过的状态，
把 12800 个冗余样本压到 2200 个，Success 只降 0.005（远小于 seed 间 SD 0.016）。

代价是 on-policy 准确率从 0.909 掉到 0.780 —— 因为它丢掉了「在专家已覆盖的
常见状态上保持一致」的那部分正则效应。这是一个**明确的取舍**：
- 若要**最小数据量**：用 `global`（1/6 数据、同成功率）
- 若要**最高绝对精度 + 最低方差**：用 `episode`（默认）

这也是对支线 5 的直接预测：数据效率曲线在专家数据上是**饱和**的（额外专家 100%
状态重叠），但在 DAgger 数据上是**近乎线性**的（覆盖半径驱动）。

### 4.4 failure-only 变体（只采失败轨迹）

单 seed 0：Success 0.853，val_acc 0.873；表征类失败从 19 掉到 6，但
`distractor_priority` 从 12 涨到 22。结论：**只采失败会让模型过度关注「错误状态」，
在其它维度上退步**。该变体已放弃。

### 4.5 跨房间迁移（S7N4，seed 10000:10300）

DAgger: 0.757 / 0.763 / 0.720，mean **0.747**。基线 S7N4 = **0.674**，Δ = **+0.073**。
同分布收益 +0.120，迁移收益只有它的一半。说明 DAgger 补的是
**具体环境尺寸下的状态覆盖**，不是普遍的方向感。
### 4.6 第二轮 DAgger（迭代）—— 收益确实递减，但尚未饱和

用第一轮的 DAgger 策略（`runs/b3_dagger_seed{0,1,2}`）再采 1500 条，聚合成两轮训练。

| 轮次 | 采集时策略的成功率 | 唯一新状态（相对专家） | 闭环 Success |
|---|---|---|---|
| 第一轮 | 0.930 | **1267** | 0.8667 ± 0.0163 |
| 第二轮 | 0.983 | 227（相对 r1 新增 628） | **0.8789 ± 0.0157** |
| 增量 | — | 只有第一轮的 **18%** | **+0.0122**（≈ 第一轮增益 0.052 的 23%） |

**判读**：

1. **新状态供给在急剧衰减**：第二轮策略成功率已达 0.983，能采到的病态状态几乎耗尽
   （唯一新状态只有第一轮的 18%）。这直接验证了文献 10 的「自主数据收益会饱和」。
2. **但 Success 仍在涨**（+0.0122），且方差从 0.0163 略降到 0.0157。
   按我们跑前定的停止准则「第二轮增益 < 第一轮的一半即判定饱和」：
   0.0122 < 0.052/2 = 0.026 → **判定为"接近饱和"**。
3. **迁移收益更明显**：S7N4 从 0.747 → **0.769**（+0.022），是 S6N3 增量的两倍。
   这说明第二轮补的不是"本房间的病态状态"，而是**更通用的方向纠错**。
   这与 §4.5 的观察互补：第一轮补环境特定的覆盖，第二轮补可迁移的部分。

**结论**：两轮是值得的（+0.012 且迁移 +0.022），第三轮大概率不划算。
下一轮如果要继续，应该转向 **novelty 加权采样**（§7-A 已实现）而不是继续加轮数。

## 5. 结果分析

### 5.1 最大的意外：早停把真实上限藏起来了（方法论发现）

| seed | patience=5 停在哪 | 该点 val_acc | 跑满 30ep 的 val_acc | 差距 |
|---|---|---|---|---|
| 0 | epoch 30（未触发） | 0.8899 | 0.8899 | 0 |
| 1 | epoch 13 | 0.8167 | **0.8720**（@ep30） | **+0.0553** |
| 2 | epoch 14 | 0.8227 | **0.8579**（@ep30） | **+0.0353** |

诊断清了两件事：

- **验证集噪声水平**：n=1844、p≈0.87 时二项标准误 = **0.0078**；
  实测相邻 epoch 的 |Δval_acc| 均值 **0.008–0.009**、最大 0.042–0.045。
  即噪声 ≈ 0.008，趋势斜率 ≈ 0.005/epoch —— **噪声大于趋势**。
- **patience=5 在这种噪声/趋势比下必然误触发**。文献 11
  （arXiv 2602.22107）的系统实验也给出了同一结论：用验证 accuracy 做早停
  "表现最差、最不稳定"，post-hoc 全 epoch 选模稳定优于早停。

**这不是我实现有 bug，而是我照抄了指导文档的 patience=5。**
指导文档说"patience 建议 5"，在 4500 条 / 18k 转移的小数据上不成立。
所以本支线的对照组装设为「跑满 30 epoch + 事后选 val_acc 最优」，
并且**手工核对了这没有违反手册**：选模依据仍是验证集动作准确率，
只是不再提前掐断训练。

同时这也是我这次最该诚实汇报的一点：**用原基线（0.747）当分母会把 DAgger 的
收益夸大一倍**。报告里必须同时给出 +0.120（vs 原基线）和 +0.052（vs 同训练时长）。

### 5.2 标签可靠性（先排除"标注有问题"这个替代解释）

| 指标 | 专家数据 | DAgger 数据 | 判读 |
|---|---|---|---|
| 同一 (image,dir,mission) 多标签率 | 0.01% | 0.06% | 几乎相同 → 标签可靠 |
| 保守 vs 激进 `replan` 分歧率 | — | **0.0%** | bot 每步重建规划，传参无影响 |
| 标签 left/right 交替率 | — | 9.6% | 存在，但这是状态本身的性质（走廊里来回），非标注噪声 |

所以「val_acc 掉」不能归因为标注质量。真正的原因是 §5.3。

### 5.3 为什么表征类失败也降了（我的原假设被推翻）

我原本预期 DAgger **只**修状态覆盖类失败（转圈/顶墙），不该动表征类
（认错属性/干扰物）。实测：

| 组 | 总失败 | 表征类 | 状态覆盖类 | 表征类占失败比 |
|---|---|---|---|---|
| baseline | 76.0 | 45.0 | 31.0 | **59.2%** |
| bc_extra | 46.7 | 31.7 | 15.0 | **67.9%** |
| dagger | 40.0 | 26.3 | 13.7 | **65.8%** |

**绝对数两类都降了，占比基本不变。** 于是我做了机制验证
（`verify_attribute_mechanism.py`），检验一个具体假设：
「DAgger 是否给模型补了专家数据里缺失的『面前是属性相近但不匹配的物体』负样本」。

| 数据集（各取前 8 步对齐） | 面前有物体 | 其中部分属性匹配 |
|---|---|---|
| 专家训练集 | 452 个状态 | **30.5%** |
| DAgger | 147 个状态 | **23.1%** |

**假设不成立**（反而低了 7.4%）。结论：
表征类失败的下降 **不是** DAgger 的直接功劳，而是「策略整体走对了、
根本到不了那个错误状态」的副产品。

**这条很重要**：「DAgger 修分布偏移」这个教科书叙事在本环境上
**无法从失败构成上得到干净的支持**——增益是全局性的（成功率整体抬升、
方差收窄），不是机制精准的。报告里不能把它写成"精准修复了状态覆盖问题"。

### 5.4 额外专家数据为什么加不动（对支线 5 的直接证据）

1500 条额外专家 episode 展开成 5057 个唯一 `(image, direction)` 状态，
其中 **100.0% 都已存在于专家训练集**，真正新增 = **0**。

这把文献 9/16 的结论落到了具体数字上：**追加同分布专家数据只能降方差，
不能扩大覆盖半径**。证据是 `bc_extra +1500` 只涨 0.030、`bc_extraMATCH +2700`
反而只有 0.824（低于 +1500 组，纯粹是 seed 噪声，两者在 SD 内不可区分）。

**对支线 5 的预测**：20% → 50% → 100% 的曲线会在低端陡、高端平（典型饱和曲线），
且斜率在 50% 之后接近零。如果支线 5 测出的是线性增长，反而说明我的判读有误。

### 5.5 为什么 on-policy 准确率反而低于 bc_extra

| 组 | on-policy 准确率 |
|---|---|
| bc_extra（额外专家） | 0.974 |
| dagger | 0.909 |

乍看不合理（DAgger 应该在策略状态上更强）。但 holdout 状态集的构成不同：
- `bc_extra` 的 holdout 是**专家轨迹**的状态（因为它是专家采集的），
  与专家训练集 100% 重叠——这是**训练集准确率**，自然极高（0.974）
- `dagger` 的 holdout 是**策略轨迹**的状态，22% 是真正新的病态状态

所以这个数字**不可跨组比较**，只能组内做纵向对比。
本支线的正确用法是：DAgger 组的 on-policy 准确率 0.909 说明
「在它自己会走到的状态上，八成九能给出正确的转向」，
而它在专家分布上是 0.893 —— 两者接近，说明它没有牺牲任何一边。
## 6. 补充调研（基于实测现象）

跑完第一轮后，带着三个具体现象重新检索，找到了之前没看的关键文献。

### 6.1 现象一：早停过早（已由文献 11/12/13/14 解释）

第一轮调研只查了 "DAgger 怎么做"，没查 "how to select checkpoints"。
补查后确认这是模型选择领域的经典问题（winner's curse），
并且有直接的实验证据支持「post-hoc 全 epoch 选模 > accuracy 早停」。

### 6.2 现象二：额外专家数据 100% 状态重叠、收益反而低于 DAgger

实测数据（`data/b3_expertextra_1500.pkl`）：

| 指标 | 数值 |
|---|---|
| 额外 1500 条专家 episode 的唯一 (image,dir) 状态数 | 5057 |
| 其中**已存在于专家训练集**的比例 | **100.0%** |
| 真正新增的状态数 | **0** |

**这解释了为什么 `bc_extra` 只涨 0.030 而 DAgger 涨 0.052**：
新采的专家数据只是把**同一个专家分布**采样得更密（降方差），
完全没有扩大覆盖半径。补查文献 9（Diversity Scaling Laws）与 15
（Novelty pump）都给出了同样的判据：**追加数据只有在带来新状态时才有信息量**。

补查文献：

| # | 标题 / 链接 | 新增了什么认识 |
|---|---|---|
| 15 | [Cargo Cults, Data Flywheels, and Novelty Pumps](https://vedder.io/misc/novelty_pump.html) | 「一条演示的新颖性取决于策略已会什么；一旦策略掌握常见情形，普通演示的新颖性趋于零」 |
| 16 | [Diversity Scaling Laws for Embodied Imitation (arXiv 2405.14005)](https://arxiv.org/abs/2405.14005) | seen-task 与 unseen-task 的 scaling 曲线不同：episode 数涨，seen 快速提升而 unseen 早饱和；真正决定泛化的是**覆盖半径 K** |
| 17 | [So You Think You Can Scale Up Autonomous Robot Data? (arXiv 2411.01813)](https://arxiv.org/html/2411.01813) | 自主数据收益会饱和；多轮采集收益递减；**novelty 重加权效果不确定** |

### 6.3 现象三：表征类失败也降了，但机制验证否定了我的假设

补查后确认没有文献支持「DAgger 能直接改善属性绑定」。
文献 8（Data Quality in IL）的 "action divergence" 概念最贴切：
DAgger 降低的是**策略与专家在同一状态上的动作分歧**，
这与"属性是否正确绑定"是两件事。所以我的实测结论（增益是全局的、非机制精准的）
与文献是一致的，不是异常。
## 7. 改进方向与下一轮计划

| 改进 | 依据 | 预期收益 | 判定有效的标准 |
|---|---|---|---|
| ~~A. 用覆盖半径筛 DAgger 样本~~ **已完成** | 文献 9/16/17 | **用 1/6 数据量达到同等 Success（0.862 vs 0.867）** | ✅ 见 §4.3 |
| **B. 迭代两轮 DAgger**（第二轮用 π₂ 采） | 文献 10 说收益会饱和但本环境尚未饱和（S7N4 仍远低于 S6N3） | +0.01~0.03，或确认已饱和 | 第二轮增益 < 第一轮的一半即判定饱和 |
| **C. β 混合（早期用专家动作执行）** | 文献 2 的 parameter-free 之外的一般形式 | 可能让 DAgger 样本更"贴"专家分布，牺牲状态覆盖 | 若 Success 上升且 on-policy 准确率也上升则采纳 |
| **D. 把早停结论回灌给主线** | 文献 11/13/14 + 本支线实测 | 基线本身能涨 0.067（0.747 → 0.814） | **已做**：这是本支线最有价值的方法论副产品 |
| **E. 不做**：failure-only / 多轮灌数据 | §4.4 实测失败 | — | — |

### 关于 A（覆盖半径筛选）—— 已实现，见 `--dedup global`

```python
# 用专家训练集的 (image_bytes, direction) 集合做查询
seen = {(img.tobytes(), d) for ep in expert_train for img, d in zip(ep["images"], ep["directions"])}
novel = [(img, d, a) for ... if (img.tobytes(), d) not in seen]
```

预期是用 1/3 的转移数拿到当前 90% 的增益。**这是下一轮最有性价比的实验。**

## 8. 复现命令

```bash
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy

# 1) 采集（每个策略用自己的 seed 采，因为不同 seed 的策略访问分布不同）
for s in 0 1 2; do
  python branches/b3_dagger/collect_dagger.py --run runs/baseline_seed$s \
    --episodes 1500 --out data/b3_dagger1500_pol$s.pkl
done

# 2) 等量对照：同样条数的额外专家数据
python branches/b3_dagger/collect_dagger.py --episodes 1500 --mode expert \
  --out data/b3_expertextra_1500.pkl

# 3) 训练（跑满 30 epoch，事后选 val_acc 最优）
for s in 0 1 2; do
  python branches/b3_dagger/train_dagger.py --seed $s --variant dagger \
    --dagger-data data/b3_dagger1500_pol$s.pkl --dedup episode \
    --out runs/b3_dagger_seed$s
done

# 4) 评测（复用冻结的评估脚本）
python src/eval_closed_loop.py --runs runs/b3_dagger_seed0 runs/b3_dagger_seed1 runs/b3_dagger_seed2 \
  --seeds test --out reports/b3_dagger_eval.json

# 5) 失败构成对比 + 汇总
python branches/b3_dagger/compare_failures.py --runs runs/baseline_seed0 ... --seeds test
python branches/b3_dagger/make_summary.py
```

## 9. 与其它支线的关系

- **与方法论文档（`branches/README.md`）的冲突，需要其它支线注意**：
  本支线发现**指导文档的 `patience=5` 在 18k 转移的数据上会把真实上限藏起来**（§5.1）。
  建议支线 1/2/4/5 全部改用「跑满 30 epoch + 事后选 val_acc 最优」，
  否则会系统性低估各自方法的效果。**这条会影响所有支线的数字可比性，
  请务必在 fork 里同步。**
- **与支线 5（数据效率）强耦合**：实测额外专家数据 100% 状态重叠（§6.2），
  这几乎可以预告支线 5 的结果——**在 4500 条之后继续加专家数据会快速饱和**。
  支线 5 的 20/50/100% 曲线应当只显示「少于 4500 时掉多少」，而不是「多于 4500 时涨多少」。
- **与支线 2（FiLM）互补且不冲突**：§5.3 已证明 DAgger **无法**从机制上改善属性绑定，
  所以 FiLM 的空间仍然完整存在。如果 FiLM 有效，两者应当可叠加。
- **与支线 1（语言编码器）弱相关**：DAgger 标注不依赖语言编码方式。
  但注意天花板效应——若语言编码让 `attribute_confusion` 大幅下降，
  DAgger 的表观收益会变小。
- **与支线 4（课程）无关**：课程只改数据来源，不改状态分布偏移的机制。

## 10. 冻结区合规自检

```bash
git diff --stat -- src/ data/ runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  reports/baseline_eval.json reports/baseline_report.md
# 期望：无输出
```

---

## 11. 第二轮迭代日志（补记）

### 11.1 做了什么

1. **第二轮采集**（`collect_round2.py`）：用 `runs/b3_dagger_seed{0,1,2}` 这三个第一轮策略
   各再采 1500 条，得到 `data/b3_dagger_r2_pol{0,1,2}.pkl`。
2. **多文件聚合**：`train_dagger.py --dagger-data a.pkl,b.pkl` 支持逗号分隔，
   并做**跨文件状态去重**（多轮之间重复状态极多）。
3. **训练两组**：`b3_r2_seed{0,1,2}`（两轮聚合）。
4. **评测**：S6N3 + S7N4 + 失败构成。

### 11.2 关键数字

| 指标 | 第一轮 | 第二轮 | 增量 |
|---|---|---|---|
| 采集时策略成功率 | 0.930 | **0.983** | +0.053 |
| 相对专家新增的唯一状态 | 1267 | **227**（相对 r1 新增 628） | 18% |
| S6N3 闭环 Success | 0.8667 ± 0.0163 | **0.8789 ± 0.0157** | **+0.0122** |
| S7N4 Success | 0.7467 | **0.7689** | **+0.0222** |

### 11.3 停止准则的落实

跑前定义：「第二轮增益 < 第一轮的一半 → 判为饱和」。
第一轮净增益（vs bc30）= **+0.0522**，第二轮 = **+0.0122**，比值 **23% < 50%** → **判定接近饱和**。

但要注意一个**反直觉现象**：迁移收益（S7N4 +0.022）是同分布收益（+0.012）的**两倍**。
解释：第一轮补的是"这个房间尺寸下的病态状态"（不可迁移），
第二轮补的是"通用方向纠错"（可迁移）。这说明**轮次的边际收益在结构上变了**，
不是单纯的递减。这一点如果只看 S6N3 的 Success 会漏掉。

### 11.4 我的判断

- **两轮值得**：+0.012 的同分布收益虽小，但 S7N4 的 +0.022 说明泛化确实在改善；
  而且方差进一步从 0.0163 降到 0.0157。
- **三轮不划算**：新状态供给已降到第一轮的 18%，第三轮大概率 < +0.005（在噪声内）。
- **下一步应该换方向**：从"加轮数"转向"更聪明的采样"（novelty / 动作分歧度加权），
  这在 §7 已经列出优先级。

### 11.5 未验证的假设（留待下一轮）

- **动作分歧度采样**是否优于状态 novelty 采样？我只做了后者（§4.3 的 `global` 模式）。
  动作分歧度的做法：优先采 `policy_a != expert_a` 的状态。理论上更聚焦，
  但实测中这些状态恰好就是「病态状态」，可能与 novelty 高度重合。
- **DAgger 样本的采样权重**：目前与专家样本等权。文献里没有明确最优比例，
  但在 4500:6000 的规模下，提高 DAgger 权重可能会让模型更关注困难状态，
  同时也可能破坏在专家分布上的表现（因为 `val_acc` 会降）。


---

## 12. 第三阶段深度改进（v2）：文献与开源调研、四项协议修正与主动交互实验

在完成 v1 两轮实验后，对代码、数据种子池与交互式模仿学习（Interactive Imitation Learning, IIL）领域的主流论文及开源仓库进行了系统复查与二次调研，定位了 v1 中的 4 项方法论/实现漏洞，并落成了统一的 `v2` 实验体系（`collect_interactive.py`、`train_iwr.py`、`eval_b3.py`、`analyze_rollout_coverage.py`、`make_figures.py`、`run_v2.sh`）。

### 12.1 二次文献与开源项目调研

| # | 论文 / 开源项目 | 链接 | 核心启发与约束 |
|---|---|---|---|
| 18 | **BabyAI 官方平台 & `train_intelligent_expert.py`** (Chevalier-Boisvert et al., ICLR 2019; Hui et al., *BabyAI 1.1*, 2020) | [`mila-iqia/babyai`](https://github.com/mila-iqia/babyai) / [`arXiv:2007.12770`](https://arxiv.org/abs/2007.12770) | 在线模仿学习必须从**未训练过的新环境种子**采集失败/纠偏，不能在已背熟的训练集种子上空转 |
| 19 | **HumanCompatibleAI `imitation` (DAggerTrainer)** | [`imitation.readthedocs.io`](https://imitation.readthedocs.io/en/latest/algorithms/dagger.html) | 通过 $\beta$ 调度混合专家动作；但盲目全局 $\beta$ 会污染在轨状态分布，需对比稀疏纠偏 |
| 20 | **LazyDAgger** (Hoque et al., ICRA 2021) | [`arXiv:2102.01245`](https://arxiv.org/abs/2102.01245) | 当策略陷入死锁时，仅在必要时刻让专家短暂接管打破循环再交还控制权，大幅减少无效步与上下文切换 |
| 21 | **SafeDAgger / HG-DAgger** (Zhang & Cho, AAAI 2017; Kelly et al., ICRA 2019) | [`arXiv:1605.06450`](https://arxiv.org/abs/1605.06450) / [`arXiv:1810.02890`](https://arxiv.org/abs/1810.02890) | 通过置信度/安全门控决定何时向专家发起查询，而非每步全量查询 |
| 22 | **EnsembleDAgger** (Menda et al., IROS 2019) | [`arXiv:1807.08364`](https://arxiv.org/abs/1807.08364) | 利用多模型集成的动作分歧与不确定性边界门控 Oracle 查询，准确统计 `n_oracle_queries` |
| 23 | **ThriftyDAgger** (Hoque et al., CoRL 2021) | [`ryanhoque/thriftydagger`](https://github.com/ryanhoque/thriftydagger) / [`arXiv:2109.08273`](https://arxiv.org/abs/2109.08273) | 预算感知（Budget-aware）的新颖性 + 风险双门控，以极低查询率维持高成功率 |
| 24 | **IWR & Sirius** (Mandlekar et al., CoRL 2020; Liu et al., RSS 2023) | [`arXiv:2012.06733`](https://arxiv.org/abs/2012.06733) / [`UT-Austin-RPL/sirius`](https://github.com/UT-Austin-RPL/sirius) | 对关键干预（$\pi(s) \neq a^*$）状态赋予权重 $w_{intv} > 1$；但需实证检验其在离散小模型上的校准副作用 |
| 25 | **AggreVaTe** (Ross & Bagnell, NIPS 2014) | [`arXiv:1406.5979`](https://arxiv.org/abs/1406.5979) | 并非所有状态动作分歧代价相等；需区分纯视觉特征缺陷与序列状态恢复需求 |

### 12.2 对 v1 的四项协议与工程缺陷修正

通过实测诊断脚本（`analyze_rollout_coverage.py`）对 v1 全部采集数据做字节级核查，发现并修正了以下四个隐蔽漏洞：

1. **修正 1：`b3_expertextra_1500` 种子与训练集 100% 重复的假象**
   - v1 `collect_dagger.py` 从 `TRAIN_SEEDS[:1500]`（即 `0:1500`）采集额外专家轨迹，恰好与 `expert_5000.pkl` 的训练前 1500 条**完全重合**。因此 v1 得出「额外专家数据 100% 重叠、新增 0 个状态」其实是脚本复用同批 seed 造成的伪结论。
   - **修复**：在 `fresh_r1`（`5300:6800`，与 `VAL/TEST/TEST_S7` 严格不交）上重采真正未见过房间的 1,500 条新专家 `BC-Fresh`。实测 `5300:6800` 包含 **5,386** 个唯一 `(img, dir, mission)` 状态，其中 **4,875 个（90.5%）相对 4,500 条训练集为全新状态**，`BC-Fresh` 闭环 Success 也从 v1 `0.8444` 升至 **0.8600**。
2. **修正 2：DAgger 在已背熟的训练种子上 rollout 导致新颖纠偏态匮乏**
   - v1 两轮 DAgger 均在已进训练集的 `0:1500` 上运行策略，导致策略靠记忆拿到了 93.0% / 98.3% 的虚假 rollout 成功率，1500 条轨迹仅采出 **1,404** 个新颖状态和 **350** 个新颖纠偏状态（$\pi(s) \neq a^*$）。
   - **修复**：引入两套无泄漏采集方案：
     - **严格 4,500 种子等预算切分（`Split-DAgger`）**：仅用前 3,000 个训练种子训初始策略 $\pi_1$，在后 1,500 个未见训练种子（`TRAIN_SEEDS[3000:4500]`）上做交互采集，总种子数严格不超 4,500。
     - **非重叠增量种子池（`fresh_r1 = 5300:6800`, `fresh_r2 = 6800:8300`）**：R1 与 R2 使用完全不相交的未见种子，新颖状态增至 **6,859**（×4.9），新颖纠偏状态增至 **1,207**（×3.4）。
3. **修正 3：去重键遗漏 `mission` 与多轮跨文件去重缺陷**
   - v1 去重键仅用 `(image, dir)`，不同指令面对同一视野被误判为重复状态；且 `train_dagger.py` 多轮加载时未更新全局 `seen_states`。
   - **修复**：`train_iwr.py` 将去重键升级为语言条件三元组 `(image, dir, mission)` 并实时维护跨文件/跨轮全局集合。
4. **修正 4：消除 20% 采集样本浪费 + 统一验证集只读探针 + 单线程防 OpenMP 锁竞争**
   - v1 训练时按 `i % 5 == 0` 丢弃了 20% 采集样本去算无法跨组比较的 `onpolicy_acc`。v2 改为 **100% 保留采集样本用于训练**，另在独立的只读验证集 rollout 探针（`data/b3_v2_val_probe.pkl`，严禁进训练集，选模仍严格只看纯净 `expert_val` 的 `val_acc`）上统一报告 `onpolicy_action_acc` 与 `onpolicy_recovery_acc`（策略犯错状态集上的纠偏准确率）。
   - 修复了百核服务器上多进程并行时未设 `OMP_NUM_THREADS=1` 引起的 OpenMP 锁竞争（单次采集耗时由 `~1740s` 降至 `~7-14s`，加速 **>118 倍**，同时保证逐比特单线程确定性）。

### 12.3 v2 实现的关键技术模块

1. **死锁解锁采样（Loop-Unsticking，受 LazyDAgger 启发）**：`collect_interactive.py --exec-mode unstick` 在检测到同一 episode 内出现 `(image, dir)` 状态循环时，向 `BabyAIBot` 请求纠偏标签并由专家执行 **1 步**脱离死锁，随后立即交还策略控制权。相比 `greedy` 全步 rollout（经常连续 30 步原地打转，采集 13,288 步却只得 6,296 个新颖态），`unstick` 只用 **8,194** 步（-38% 查询）就完成了 100% 轨迹并采到 **6,859** 个新颖态和 **1,207** 个新颖纠偏态。
2. **主动查询门控（Active Querying）**：
   - `margin` 门控（SafeDAgger / ThriftyDAgger 思路）：当主策略 softmax `top1 - top2 < 0.45` 或处于状态循环时才查询专家，额外查询率降至 **18.8%**（~1,610 步/1500 ep）。
   - `ensemble` 门控（EnsembleDAgger 思路）：利用 3-seed 集成投票分歧或均值裕度 `< 0.30`（等预算组 `< 0.35`）门控查询，额外查询率仅 **25.2%**（~2,150 步/1500 ep）。
   - `disagree` 过滤对照：全步查询但仅保留 $\pi(s) \neq a^*$ 的状态入库。
3. **干预加权回归（IWR）与 FiLM 正交叠加**：`train_iwr.py` 对 $\pi_{rollout}(s) \neq a^*$ 状态施加损失权重 $w_{intv} \in \{1.0, 2.0, 3.0\}$，并原生支持 B2 的 `FiLMBCNet` 架构以测试「表征接地 × 状态覆盖纠偏」的可叠加性。

### 12.4 v2 完整实验结果（3 seed，mean ± SD，与冻结 `eval_closed_loop` 校验差值 `0.0`）

完整 25 组明细见 [`reports/b3_dagger_summary.md`](../../reports/b3_dagger_summary.md) §7 与 [`reports/b3_v2_eval_s6n3.json`](../../reports/b3_v2_eval_s6n3.json) / [`reports/b3_v2_eval_s7n4.json`](../../reports/b3_v2_eval_s7n4.json)。关键结论如下：

| 实验维度 | 代表组别 | S6N3 Success | S7N4 迁移 | 总专家标注 | 失败(表征+覆盖) | 关键结论 |
|---|---|---|---|---|---|---|
| **等预算 4,500 种子** | `BC30` (4,500 专家) | 0.8144 ± 0.0453 | 0.6967 ± 0.0562 | 16,534 | 38.3 + 17.3 | 真基线 |
| | `BC-3000` (3,000 专家) | 0.7522 ± 0.0126 | 0.6656 ± 0.0247 | 11,074 | 43.7 + 30.7 | 起点 |
| | `Split-DAgger` (3k+1.5k) | **0.8511 ± 0.0016** | **0.7689 ± 0.0512** | 21,098 | 27.7 + 17.0 | 零额外种子反超 `BC30` +0.0367 |
| | `Split-EnsembleDAgger` (3k+1.5k) | **0.8578 ± 0.0351** | **0.7644 ± 0.0372** | **13,740** | 26.0 + 16.7 | **少 16.9% 标注反超 `BC30` +0.0433** |
| **增量 1,500 新种子** | `BC-Fresh` (+1,500 新专家) | 0.8600 ± 0.0260 | 0.7711 ± 0.0083 | 22,111 | 29.3 + 12.7 | 修正 v1 重复种子后的强专家对照 |
| (`fresh_r1: 5300:6800`) | `v2 Greedy` (全步, w=1) | 0.8911 ± 0.0318 | 0.7667 ± 0.0381 | 31,304 | 19.0 + 13.7 | 高查询成本、高方差、死锁冗余多 |
| | `v2 β=0.25` 混合 (w=1) | 0.8522 ± 0.0042 | 0.7522 ± 0.0204 | 25,237 | 28.7 + 15.7 | 随机混合破坏了状态覆盖增益 |
| | **`v2 Loop-Unsticking` (w=1)** | **0.8900 ± 0.0152** | **0.8067 ± 0.0109** | **25,060** | **23.3 + 9.7** | **vs Greedy 方差减半、少 20% 标注、S7N4 +0.040** |
| **主动门控 & 权重消融** | `v2 Margin 门控` (18.8% 查询) | 0.8589 ± 0.0244 | 0.7533 ± 0.0098 | **18,144** | 24.3 + 18.0 | 仅 1.6k 额外查询逼近 22.1k `BC-Fresh` |
| | `v2 Ensemble 门控` (25.2% 查询) | 0.8500 ± 0.0144 | **0.7878 ± 0.0079** | **18,684** | 23.0 + 22.0 | 低查询成本下迁移稳健（SD 0.0079） |
| | `v2 Disagreement-Only` (w=2) | 0.8389 ± 0.0083 | 0.7522 ± 0.0057 | 25,060 | 28.3 + 20.0 | 丢弃一致样本损伤在轨正则 |
| | `v2 Unstick + IWR (w=2 / w=3)` | 0.8622 / 0.8556 | 0.7989 / 0.7844 | 25,060 | 覆盖升至 18.7 | **诚实负结果**：在 `BCNet` 上 $w>1$ 诱发近距振荡 |
| **两轮非重叠迭代** | **`v2 R2 (w=1.0)`** | **0.9089 ± 0.0191** | **0.8278 ± 0.0264** | 32,668 | **19.0 + 8.3** | 突破 0.90，`sc` 失败压低至 `BC30` 的一半 |
| | `v2 R2 + IWR (w=2.0)` | 0.9022 ± 0.0201 | 0.7978 ± 0.0155 | 32,797 | 17.0 + 12.3 | seed0 达 0.9300，但迁移弱于 `w=1.0` |
| **FiLM 正交叠加** | `B2 FiLM` (4,500 专家) | 0.9589 ± 0.0063 | 0.9000 ± 0.0205 | — | 6.3 + 6.0 | B2 单项最强基准 |
| | `FiLM-Split` (3k+1.5k=4,500) | 0.9589 ± 0.0057 | 0.9044 ± 0.0185 | 17,324 | 5.0 + 7.3 | 少 1,500 初始专家追平全量 `FiLM` |
| | **`FiLM + DAgger (w=1.0)`** | **0.9689 ± 0.0068** | **0.9178 ± 0.0181** | 22,539 | **3.7 + 5.7** | **S7N4 达 0.9178（全项目最高迁移）** |
| | **`FiLM + DAgger + IWR (w=2.0)`** | **0.9700 ± 0.0072** | 0.9000 ± 0.0141 | 22,539 | **4.7 + 4.3** | **S6N3 达 0.9700（seed0 0.9800），总失败仅 9.0** |

### 12.5 深入机理反思：为什么 $w_{intv} > 1$ 在 `BCNet` 上有害、而在 `FiLM` 上影响微弱？

- 在 `BCNet` 上，`w=2.0/3.0` 成功将只读探针上的纠偏准确率 `recov_acc` 从 `0.4411` 拉升至 `0.4934 / 0.4890`，但闭环 S6N3 成功率反而跌落 `0.028 ~ 0.034`，且 `state_coverage` 失败由 `9.7` 翻倍至 `18.7`。根因在于：`BCNet` 对颜色词几乎失聪（见支线 2 颜色敏感度 `0.065`），导致 `π≠a*` 的分歧样本里混杂了大量**模型感知层本身不可分辨的状态**；加倍其权重等于迫使网络在欠表达的视觉特征上过度拟合矛盾转向信号，反而破坏了正常前进动作的稳定性（`val_acc` 下降、`left/right` 振荡增多）。
- 相反，在具备颜色-类型绑定能力的 `FiLM` 上，分歧样本几乎都是**真正需要学习的边缘/遮挡或贴墙几何状态**，因此 `FiLM+DAgger` 在 `w=1.0`（`0.9689` / S7N4 `0.9178`）与 `w=2.0`（`0.9700` / S7N4 `0.9000`）下均显著高于 `B2 FiLM`（`0.9589` / S7N4 `0.9000`），将总失败进一步压缩了 **27%**（`12.3 → 9.0`）。

### 12.6 v2 完整复现命令与产物清单

```bash
# 一键复现 v2 全流程（25 组 × 3 seed + S6N3/S7N4 评测 + 冻结口径核验）
bash branches/b3_dagger/run_v2.sh

# 覆盖率诊断、发表级图表与汇总表
python branches/b3_dagger/analyze_rollout_coverage.py --out reports/b3_v2_rollout_coverage.json
python branches/b3_dagger/make_figures.py
python branches/b3_dagger/make_summary.py
```

- 核心代码：`collect_interactive.py`、`train_iwr.py`、`eval_b3.py`、`analyze_rollout_coverage.py`、`make_figures.py`、`make_summary.py`、`run_v2.sh`
- 评测报告：`reports/b3_v2_eval_s6n3.json`、`reports/b3_v2_eval_s7n4.json`、`reports/b3_v2_frozen_eval_check.json`、`reports/b3_v2_rollout_coverage.json`、`reports/b3_dagger_summary.md`
- 图表产出：`reports/figures/b3_dagger/b3_fig{1..4}_*.png`
