# 基线结果汇总表

## S6N3 闭环评估（4500:4800，每模型 300 episode）

| run | 验证动作准确率 | 闭环 Success | 平均回报 | 成功长度中位数 | 超时数 |
|---|---|---|---|---|---|
| `baseline_seed0` | 0.881 | **0.837** | 0.742 | 4.0 | 49/300 |
| `baseline_seed1` | 0.823 | **0.797** | 0.689 | 5.0 | 61/300 |
| `baseline_seed2` | 0.823 | **0.710** | 0.623 | 4.0 | 87/300 |
| **mean ± SD** | 0.842 ± 0.027 | **0.781 ± 0.053** | 0.685 | 4.0 | — |

## 平凡策略（证明「准确率 ≠ 成功率」）

| 策略 | teacher-forcing 准确率 | 闭环 Success |
|---|---|---|
| `always_forward` | 0.564 | 0.060 |
| `always_left` | 0.220 | 0.000 |
| `random` | ≈ 1/3（均匀随机） | 0.543 |

## S7N4 迁移（10000:10300）

| run | S6N3 Success | S7N4 Success | Δ |
|---|---|---|---|
| `baseline_seed0` | 0.837 | **0.773** | -0.063 |
| `baseline_seed1` | 0.797 | **0.670** | -0.127 |
| `baseline_seed2` | 0.710 | **0.660** | -0.050 |

## 失败归因（`runs/baseline_seed0`，300 episode）

成功 251/300，失败构成：

| 类型 | 数量 |
|---|---|
| `attribute_confusion` | 18 |
| `distractor_priority` | 15 |
| `turn_oscillation` | 9 |
| `orientation_lock` | 5 |
| `forward_stall` | 1 |
| `wander_timeout` | 1 |

- 数据审计：已完成 `reports/data_audit.md`