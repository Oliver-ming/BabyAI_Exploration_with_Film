# 基线结果汇总表

## S6N3 闭环评估（4500:4800，每模型 300 episode）

| run | 验证动作准确率 | 闭环 Success | 平均回报 | 成功长度中位数 | 超时数 |
|---|---|---|---|---|---|
| `baseline_seed0` | 0.890 | **0.877** | 0.773 | 4.0 | 37/300 |
| `baseline_seed1` | 0.817 | **0.663** | 0.584 | 4.0 | 101/300 |
| `baseline_seed2` | 0.823 | **0.700** | 0.612 | 4.0 | 90/300 |
| **mean ± SD** | 0.843 ± 0.033 | **0.747 ± 0.093** | 0.656 | 4.0 | — |

## 平凡策略（证明「准确率 ≠ 成功率」）

| 策略 | teacher-forcing 准确率 | 闭环 Success |
|---|---|---|
| `always_forward` | 0.564 | 0.060 |
| `always_left` | 0.220 | 0.000 |
| `random` | ≈ 1/3（均匀随机） | 0.543 |

## S7N4 迁移（10000:10300）

| run | S6N3 Success | S7N4 Success | Δ |
|---|---|---|---|
| `baseline_seed0` | 0.877 | **0.773** | -0.103 |
| `baseline_seed1` | 0.663 | **0.573** | -0.090 |
| `baseline_seed2` | 0.700 | **0.677** | -0.023 |

## 失败归因（`runs/baseline_seed0`，300 episode）

成功 263/300，失败构成：

| 类型 | 数量 |
|---|---|
| `attribute_confusion` | 19 |
| `distractor_priority` | 12 |
| `turn_oscillation` | 4 |
| `orientation_lock` | 2 |

- 数据审计：已完成 `reports/data_audit.md`