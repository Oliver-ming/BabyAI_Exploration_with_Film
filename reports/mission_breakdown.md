# P2-1 · mission 类型分解

> 数据源：`reports/baseline_eval.json`（3 个 run，测试 seed 4500:4800）
> 训练样本数来自 `data/expert_5000.pkl (split=train)`

**加权总成功率：0.747**（覆盖 32/36 种 mission，共 900 个测试 episode）

## 按冠词分解

| 冠词 | 测试 episode | 成功率 | 说明 |
|---|---|---|---|
| `the` | 804 | **0.724** | 场景只有 1 个匹配物体 |
| `a` | 96 | **0.937** | 场景有多个匹配物体，任一个都算成功 |

## 按物体类型分解

| 类型 | 测试 episode | 成功率 |
|---|---|---|
| `ball` | 333 | **0.733** |
| `box` | 249 | **0.755** |
| `key` | 318 | **0.755** |

## 按颜色分解

| 颜色 | 测试 episode | 成功率 |
|---|---|---|
| `red` | 138 | **0.746** |
| `green` | 135 | **0.741** |
| `blue` | 126 | **0.738** |
| `purple` | 174 | **0.764** |
| `yellow` | 174 | **0.730** |
| `grey` | 153 | **0.758** |

## 低覆盖 mission（训练样本 < 20，成功率不可靠）

| mission | 训练样本 | 测试 episode | 成功率 |
|---|---|---|---|
| `go to a red box` | 15 | 15 | 1.000 |
| `go to a purple key` | 19 | 6 | 1.000 |

这些 mission 在测试集里只有个位数 episode，单个 episode 的成败就能让成功率跳动 ±20 个百分点以上，**不能据此下任何结论**。报告中引用时应标注为「样本不足，仅供参考」。

## 成功率最高 / 最低的 mission（仅列测试 episode ≥ 5 的）

| | mission | n_train | n_test | success |
|---|---|---|---|---|
| 最高 | `go to a green key` | 21 | 6 | 1.000 |
| 最高 | `go to a grey ball` | 21 | 9 | 1.000 |
| 最高 | `go to a grey box` | 28 | 6 | 1.000 |
| 最高 | `go to a purple ball` | 20 | 6 | 1.000 |
| 最低 | `go to a green ball` | 22 | 6 | 0.500 |
| 最低 | `go to the blue box` | 223 | 21 | 0.524 |
| 最低 | `go to the red ball` | 264 | 39 | 0.590 |
| 最低 | `go to the grey key` | 221 | 42 | 0.643 |
