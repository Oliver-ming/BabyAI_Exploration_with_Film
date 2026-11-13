# 支线隔离契约（五条支线并行开发的前置约定）

本项目的主线（**基准复现**）已经完成并**冻结**。之后五个优化项各自是一条独立支线，
分布在五个 fork 里并行推进。本文件是这五条支线都必须遵守的边界规则。

---

## 1. 冻结区（任何支线都不得修改）

| 路径 | 内容 | 为什么冻结 |
|---|---|---|
| `src/common.py` | seed 分段、词表、`BCNet`、编码工具 | **实验协议的唯一出处**。任一支线改它，其它四条的数据划分与模型定义立刻失效，跨支线结果不可比 |
| `src/train_bc.py` | BC 基线训练循环 | 基线数字 0.843 / 0.747 由它产出，改动会让历史结果无法复现 |
| `src/eval_closed_loop.py` | 闭环评估 | 主指标口径。五个支线必须用同一套评估才算得出"有没有提升" |
| `src/gen_expert.py` `src/audit_data.py` | 专家数据生成与审计 | 5000 条专家 episode 是固定预算，不许任何支线重新生成 |
| `src/mission_breakdown.py` `src/analyze_failures.py` `src/render_episodes.py` `src/make_figures.py` `src/make_report.py` `src/check_env.py` | 分析/可视化/报告 | 只读复用 |
| `data/expert_5000.pkl` | 5000 条专家数据 | 支线 4 需要额外数据时**新建文件**，不许改写它 |
| `runs/baseline_seed{0,1,2}` `runs/ablation_*` | 基线 checkpoint | 支线的对照基准 |
| `reports/baseline_*` `reports/data_audit.md` `reports/mission_breakdown.*` `reports/s7n4_transfer.json` `reports/failure_analysis.*` `reports/failures/` `reports/figures/**` `videos/**` | 基线产物 | 同上 |

**允许的唯一"侵入"方式**：`import` 冻结模块并使用它（只读复用）。**不允许**编辑、monkey-patch、
或通过环境变量改变冻结模块的行为。

`git diff` 对上述路径必须始终为空。支线提交前后的自检：

```bash
git diff --stat -- src/ data/ runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  reports/baseline_eval.json reports/data_audit.md
# 期望：无输出
```

---

## 2. 支线的专属区（自由改动）

```
branches/
├── README.md                 ← 本文件（契约，冻结）
├── _template/                ← 新建支线的骨架
│   ├── LOG.md
│   └── run.sh
├── b1_langenc/               ← 支线 1：语言编码器
├── b2_film/                  ← 支线 2：FiLM 跨模态融合
├── b3_dagger/                ← 支线 3：DAgger-lite
├── b4_curriculum/            ← 支线 4：课程学习
└── b5_data_eff/              ← 支线 5：数据效率
```

每条支线**只写自己的目录**，产物命名一律带支线前缀：

| 类型 | 位置 | 命名 |
|---|---|---|
| 代码 | `branches/bN_xxx/*.py` | 自由 |
| 训练产物 | `runs/bN_xxx/<variant>_seed<k>/` | **必须以 `bN_` 开头** |
| 结果数据 | `reports/bN_xxx_<name>.json` | 同上 |
| 图表 | `reports/figures/bN_xxx/*.png` | 子目录隔离 |
| 新增数据 | `data/bN_xxx_*.pkl` | 同上 |
| 过程记录 | `branches/bN_xxx/LOG.md` | 必需 |

这样任何一条支线被单独 merge 或单独丢弃，都不会覆盖别人的文件。

---

## 3. 每条支线必须走的四步

用户明确要求：**先调研 → 再实现 → 跑一轮 → 再补文献 → 记录并给理由**。

### 步骤 A：文献调研（实现之前）

写进 `LOG.md` 的「调研」小节，至少包含：

1. **3–5 篇可引用的文献/文档**（论文、官方文档、权威教程），每条给出：标题、链接、一句话它解决了什么
2. **从文献里提炼的 3–6 条具体设计约束**（不是"要小心"，而是"因为 X，所以我要用 Y"）
3. **已知陷阱清单**：文献里明确说"这样做会失败"的地方，以及本支线如何规避

目的：防止一头扎进实现然后卡在某个点上，或者做出一个文献里早就证明无效的东西。

### 步骤 B：实现

- 所有新代码放在 `branches/bN_xxx/`
- 优先**复用冻结模块**（例如直接调用 `src/eval_closed_loop.py` 的 `evaluate()`），
  而不是复制一份改一改——复制会立刻产生口径漂移
- 关键设计选择写注释，指明它对应调研里的哪条约束

### 步骤 C：跑一轮 + 结果

- 默认 **3 个 seed**（与基线同协议：seed 0/1/2，早停只看验证集动作准确率，测试 `4500:4800`）
- 与基线对比必须报告：`val_action_acc`、`closed_loop_success`、`mean_return`、
  `median_len_success`、`timeout 数`
- 提升是否可信：与基线 seed 间 SD（acc 0.033 / success 0.093）比较，小于 SD 的一律标注"不显著"

### 步骤 D：跑完之后补充文献

基于**实测看到的具体现象**再检索一次。这一步不是形式主义，而是因为：

- 第一轮跑出来的失败模式往往和预期不同（例如本项目实测发现随机策略就有 54% 成功率，
  整个改进空间被压缩——这在动手前并不知道）
- 只有在看到具体瓶颈之后，才知道该去查"如何解决 X"而不是泛泛地查领域综述

补完文献后，写「改进方向」小节：下一轮具体改什么、预期收益多少、怎么判定成功/失败。

---

## 4. LOG.md 的标准结构

```markdown
# 支线 N · <名字>

## 0. 一句话结论
## 1. 目标与成功判据（跑之前就写好）
## 2. 调研（实现之前）
### 2.1 文献
### 2.2 提炼的设计约束
### 2.3 已知陷阱
## 3. 实现（含关键决策与理由）
## 4. 结果（第一轮）
## 5. 结果分析：哪里符合预期、哪里不符合
## 6. 补充调研（基于实测现象）
## 7. 改进方向与下一轮计划
## 8. 复现命令
## 9. 与其它支线的关系（谁的结论会影响谁）
```

第 9 节很重要：五条支线看似独立，但结论会互相影响。例如
"DAgger 收益小" 与 "随机基线 0.54 已很高" 是同一件事的两面，
如果不交叉说明，评审会认为两个结论互相矛盾。

---

## 5. 共享的评估口径（不许另写一套）

主指标一律来自 `src/eval_closed_loop.py`：

```python
import sys; sys.path.insert(0, "src")
from eval_closed_loop import evaluate, parse_seeds
from common import ENV_ID, TEST_SEEDS

res = evaluate(policy_fn, ENV_ID, parse_seeds("test"))
# policy_fn: obs -> int，必须是无状态的单步函数（内部可以包模型）
```

- 测试 seed 固定 `4500:4800`（300 个），S7N4 固定 `10000:10300`
- 成功判定用 `terminated`，超时是 `truncated`
- **绝不用闭环 Success 选 checkpoint**（手册硬规定）
- **绝不让测试 seed 进任何训练集**（`common.assert_no_leak` 已在 `train_bc.py` 里硬断言）

---

## 6. 当前支线状态

| 支线 | 目录 | 状态 | 一句话结论 |
|---|---|---|---|
| 3 · DAgger-lite | `b3_dagger/` | **✅ 完成（两轮）** | Success 0.747 → **0.879**；净收益 +0.052；1/6 数据量可达同等效果；附带发现「patience=5 会低估真实上限」 |
| 2 · FiLM 融合 | `b2_film/` | ✅ 完成 | Success 0.814（BC30 对照）→ **0.959**（+0.144，3/3 seed）；颜色敏感度 0.065→0.268（×4.1）；表征类失败 −78%；S7N4 迁移 +0.203 |
| 1 · 语言编码器 | `b1_langenc/` | 未开始（建议先测语言敏感度，见 b2 §9） | — |
| 4 · 课程学习 | `b4_curriculum/` | 未开始 | — |
| 5 · 数据效率 | `b5_data_eff/` | 未开始 | — |

### 已沉淀的跨支线方法论（后续支线直接继承，不必重新踩坑）

1. **不要用 `patience=5` 早停**（b3 实测）：在 18k 转移的数据上，验证集噪声（≈0.008）
   大于趋势斜率（≈0.005/epoch），早停会系统性低估真实上限达 0.067。
   **统一改为「跑满 30 epoch + 事后选 val_acc 最优」**（选模依据仍是验证集动作准确率，不违反手册）。
2. **必须设计一个能证伪自己假设的探针**（b3 §5.3 + b2 §4.2）：
   b3 的机制验证推翻了「DAgger 精准修复状态覆盖」的叙事；
   b2 用反事实探针（改 mission 看动作是否变化）证明 FiLM 的收益来自真实语言接地而非容量。
   只看 Success 会被"参数变多""训练更久"这类混淆变量骗过。
3. **参数量匹配对照是必需的**（b2）：`concat_wide` 参数量只比 `film` 少 500，
   但 Success 低 0.241 —— 这是排除"只是多参数"的唯一硬证据。
4. **正则要加在新增分支上，不要无差别压全网**（b2 §5.1）：
   wd=1e-4 让基线 Success 掉 0.129（因为 90% 参数挤在视觉第一层），
   而对 FiLM 组几乎无影响（它有自己的 `film_l2`）。
5. **数据/标签的可靠性要先排除**（b3 §5.2）：在讨论"模型不行"之前，
   先量化标签歧义率、重复率、状态重叠率。b3 用 `check_label_conflicts.py` 做过这件事。
