# P0-3 · 专家数据集审计（BabyAI-GoToLocalS6N3）

> 数据文件 `data/expert_5000.pkl`（按 seed 的 `split` 字段分词；训练 4500 / 验证 500）　生成脚本 `src/gen_expert.py`　审计脚本 `src/audit_data.py`

## 0. 实验协议与实际 seed 划分（必读）

### 0.1 为什么不能照抄 §1.3 的 seed 配置

指导文档 §1.3 写 `TRAIN_SEEDS = range(0, 4000)` 用来生成 **5000 条**专家演示。这里有两处冲突：

1. **数量不足**：同一个 env seed 只能产出一条确定性 episode，`range(0,4000)` 最多 4000 条互不重复样本。用循环抽样凑到 5000 条会注入 1000 条完全重复数据，直接污染 §10.2 的数据效率消融。
2. **验证集也要专家**：§1.2 ① 强制用**验证集动作准确率**选 checkpoint，算准确率必须知道 VAL seed 上的专家动作，所以要额外跑 500 条专家 episode。若训练集仍占满 5000 条，专家 episode 总数 = 5500 > 手册硬上限 5000。

### 0.2 本实现的划分（`src/common.py` 是唯一定义处）

| 用途 | seed 段 | 条数 | 与文档的关系 |
|---|---|---|---|
| 训练专家（主段） | `0..4000` | 4000 | 与文档一致 |
| 训练专家（补足段） | `4800..5300` | 500 | **文档未覆盖的区间** |
| 验证专家（选 checkpoint） | `4000..4500` | 500 | 与文档一致 |
| 测试（最终闭环评估） | `4500..4800` | 300 | 与文档一致 |
| 迁移测试 S7N4 | `10000..10300` | 300 | 与文档一致 |

**专家 episode 总预算**：4500（训练）+ 500（验证）= **5000** 条，正好等于手册上限 5000，未超。

补足段 `4800..5300` 与 VAL(`4000..4500`)、TEST(`4500..4800`)、TEST_S7(`10000..10300`) **全不相交**，且从未参与任何评估，因此「测试 seed 绝不进训练」这条硬性规定仍然成立。

### 0.3 实际数据规模

| 数据 | episode | 转移 |
|---|---|---|
| 训练集（split=train） | 4500 | 16534 |
| 验证集（split=val） | 500 | 1844 |
| 合计（= 专家 episode 总预算） | **5000** | 18378 |

## 1. 训练集规模与 episode 长度

| 指标 | 值 |
|---|---|
| 成功专家 episode 数 | **4500** |
| 转移（(obs, action) 训练样本）总数 | **16534** |
| 平均长度 | 3.67 步 |
| 长度 min / median / max | 1 / 3.0 / 13 |
| 专家失败 episode | 0（BabyAIBot 在 S6N3 上 100% 成功） |
| 验证集 episode 数 | 500（36 种 mission） |

长度分布：

| 长度（步） | episode 数 | 占比 |
|---|---|---|
| 1 | 215 | 4.8% |
| 2 | 927 | 20.6% |
| 3 | 1123 | 25.0% |
| 4 | 1033 | 23.0% |
| 5 | 622 | 13.8% |
| 6 | 328 | 7.3% |
| 7 | 164 | 3.6% |
| 8 | 65 | 1.4% |
| 9 | 17 | 0.4% |
| 10 | 4 | 0.1% |
| 12 | 1 | 0.0% |
| 13 | 1 | 0.0% |

**关键含义**：4500 条 episode 只展开成 **16534** 个 (obs, action) 样本。这个量级意味着 BC 训练几十秒就能收敛，主要风险是**过拟合专家访问过的状态分布**，而不是欠拟合。这也是 §7 动作准确率与闭环 Success 会出现巨大差距的根本原因。

episode 长度 = 1 的样本占 4.8%：agent 出生时就已「站在目标旁边并面向它」，一步都不用动。这类样本对训练几乎只提供「当前朝向已正确」这一条信息，却在数据里占固定比例，会让验证准确率虚高。

## 2. 动作分布（对照 §4.1 ④）

| 动作 | 次数 | 占比 |
|---|---|---|
| `left` | 3644 | 22.0% |
| `right` | 3558 | 21.5% |
| `forward` | 9332 | 56.4% |
| `pickup` | 0 | 0.0% |
| `drop` | 0 | 0.0% |
| `toggle` | 0 | 0.0% |
| `done` | 0 | 0.0% |

**平凡基线**：「永远预测 forward」可拿到 **56.4%** 的 teacher-forcing 动作准确率，但闭环 Success 恒为 0（不会转向就永远到不了目标）。报告必须把这组数字与闭环 Success 并列呈现。
`pickup/drop/toggle/done` 合计 **0.0%**　GoToLocal 只需移动+转向，符合预期。

## 3. mission 覆盖（36 种字符串）

训练集出现 **36/36** 种 mission；≥20 条的有 **34/36** 种。未出现 0 种，低覆盖 2 种。

| mission | 训练 episode 数 | 占比 | 覆盖判定 |
|---|---|---|---|
| `go to the red ball` | 264 | 5.87% | OK |
| `go to the red box` | 199 | 4.42% | OK |
| `go to the red key` | 234 | 5.20% | OK |
| `go to the green ball` | 216 | 4.80% | OK |
| `go to the green box` | 227 | 5.04% | OK |
| `go to the green key` | 203 | 4.51% | OK |
| `go to the blue ball` | 214 | 4.76% | OK |
| `go to the blue box` | 223 | 4.96% | OK |
| `go to the blue key` | 226 | 5.02% | OK |
| `go to the purple ball` | 241 | 5.36% | OK |
| `go to the purple box` | 219 | 4.87% | OK |
| `go to the purple key` | 220 | 4.89% | OK |
| `go to the yellow ball` | 227 | 5.04% | OK |
| `go to the yellow box` | 214 | 4.76% | OK |
| `go to the yellow key` | 236 | 5.24% | OK |
| `go to the grey ball` | 225 | 5.00% | OK |
| `go to the grey box` | 227 | 5.04% | OK |
| `go to the grey key` | 221 | 4.91% | OK |
| `go to a red ball` | 24 | 0.53% | OK |
| `go to a red box` | 15 | 0.33% | **低覆盖** |
| `go to a red key` | 21 | 0.47% | OK |
| `go to a green ball` | 22 | 0.49% | OK |
| `go to a green box` | 27 | 0.60% | OK |
| `go to a green key` | 21 | 0.47% | OK |
| `go to a blue ball` | 34 | 0.76% | OK |
| `go to a blue box` | 23 | 0.51% | OK |
| `go to a blue key` | 33 | 0.73% | OK |
| `go to a purple ball` | 20 | 0.44% | OK |
| `go to a purple box` | 38 | 0.84% | OK |
| `go to a purple key` | 19 | 0.42% | **低覆盖** |
| `go to a yellow ball` | 31 | 0.69% | OK |
| `go to a yellow box` | 34 | 0.76% | OK |
| `go to a yellow key` | 23 | 0.51% | OK |
| `go to a grey ball` | 21 | 0.47% | OK |
| `go to a grey box` | 28 | 0.62% | OK |
| `go to a grey key` | 30 | 0.67% | OK |

### 低覆盖 mission（<20 条）

这些 mission 的**成功率估计方差极大**，P2-1 的分解表里必须单独标注，不可当作结论。

- `go to a red box`：15 条
- `go to a purple key`：19 条

## 4. 冠词分布（`the` vs `a`）

| 冠词 | 训练 episode 数 | 占比 | 语义（`surface()` 逻辑） |
|---|---|---|---|
| `the` | 4036 | 89.7% | 场景里只有 1 个匹配物体 |
| `a` | 464 | 10.3% | 场景里有 >1 个匹配物体，任一个都算成功 |

`a` 只占 10.3%，但它天然更难：存在多个候选目标，模型必须靠「哪个更近 / 哪个在视野里」这类隐式线索挑一个，挑错就退化成 `distractor_priority` 失败。因此 36 种 mission 的成功率不会均匀，`a` 系列应当显著更低。

## 5. 每个 episode 的匹配物体数量（`obj_poss`）分布

| 匹配物体数 | episode 数 | 占比 |
|---|---|---|
| 1 | 4036 | 89.7% |
| 2 | 454 | 10.1% |
| 3 | 10 | 0.2% |

匹配物体数 = 1 的有 4036 条（89.7%），>1 的有 464 条（10.3%）。这与 §4 的冠词分布严格对应（`surface()` 用 `len(obj_set) > 1` 决定加 `a`），可作为数据一致性交叉验证：两项比例应一致。

## 6. 观测唯一性

| 指标 | 值 |
|---|---|
| 训练转移总数 | 16534 |
| 唯一 `(image, direction)` 组合 | **14374**（86.9%） |
| 唯一 `image` 帧 | **13263**（80.2%） |

唯一 `(image, direction)` 占 86.9%，重复主要来自「转一圈回到同一朝向」这类状态。更关键的是唯一 `image` 帧数（13263）明显少于唯一组合数（14374）——**同一个 7×7 patch 可以对应多个绝对朝向**，这直接证明了 §4.1 ⑤ 的结论：`direction` 是不能省的输入，只给 patch 无法唯一确定「该往哪边转」。

## 7. 验收结论

- ✅ 专家 episode 总数 = 5000（手册上限 5000）
- ✅ 训练集无 pickup/drop/toggle/done 动作
- ✅ 36 种 mission 全部在训练集出现（实际 36）
- ✅ ≥20 条的 mission 有 34/36（§5 要求 ≥30）
- ✅ 训练集内部无重复 seed（无重复样本）
- ✅ 训练集与验证集均不含任何 TEST / TEST_S7 seed
- ✅ 训练集与验证集 seed 不相交
- ✅ 训练集 seed 全部落在约定的 TRAIN_SEEDS 内

对照 §5 的六项要求：1 规模/长度 ✅　2 动作分布 ✅　3 mission 覆盖表 ✅　4 冠词分布 ✅　5 `obj_poss` 分布 ✅　6 观测唯一性 ✅
