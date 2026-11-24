# ABC attention fusion — one primary comparison

**Primary modification:** single-frame language-conditioned spatial fusion with raw-feature/map concatenation and relative-bias relational attention (ABC).
**Primary comparison:** BC30 vs ABC (`s3_map1h_relbias`). A, AB and FiLM are supporting controls.

4,500 training + 500 validation expert episodes; Adam lr=1e-3, wd=0; 3 initialization seeds; 300 unseen episodes per environment. Values are mean ± population SD (ddof=0).
**Protocol deviation:** 30 full epochs without early stopping, checkpoint selected by validation action accuracy only. Original early-stopped baseline: `baseline_report.md`.

## 1. Primary comparison

| Method | Params | S6N3 success | Mean return | Median successful length (steps) | Val action accuracy | S7N4 success | S8N6 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| BC30 | 223,071 | 0.814 ± 0.045 | 0.723 ± 0.037 | 4.000 ± 0.000 | 0.873 ± 0.013 | 0.697 ± 0.056 | 0.490 ± 0.019 |
| ABC: Map + RelBias | 230,436 | **0.974 ± 0.002** | 0.886 ± 0.002 | 3.000 ± 0.000 | 0.983 ± 0.002 | 0.944 ± 0.009 | 0.814 ± 0.009 |

| Method | Seed 0 S6N3 | Seed 1 S6N3 | Seed 2 S6N3 |
|---|---:|---:|---:|
| BC30 | 0.877 | 0.797 | 0.770 |
| ABC: Map + RelBias | 0.973 | 0.977 | 0.973 |

## 2. Component ablations and alternative fusion

A applies multiplicative weighting. AB replaces it with `[H; M] → Conv2`; ABC adds C after Conv2. AB vs ABC isolates C.

| Method | Role | S6N3 success | S7N4 success | S8N6 success | 20% S6N3 | 20% S7N4 |
|---|---|---:|---:|---:|---:|---:|
| BC30 | Baseline | 0.814 ± 0.045 | 0.697 ± 0.056 | 0.490 ± 0.019 | 0.742 ± 0.026 | 0.627 ± 0.021 |
| ABC: Map + RelBias | Primary method | 0.974 ± 0.002 | 0.944 ± 0.009 | 0.814 ± 0.009 | 0.876 ± 0.024 | 0.802 ± 0.062 |
| A: Spatial-Softmax | Fusion-design ablation | 0.971 ± 0.003 | 0.964 ± 0.010 | 0.887 ± 0.018 | 0.898 ± 0.009 | 0.854 ± 0.021 |
| AB: Map-Inject | Remove C from ABC | 0.974 ± 0.004 | 0.960 ± 0.003 | 0.829 ± 0.023 | 0.881 ± 0.013 | 0.858 ± 0.053 |
| FiLM | Alternative fusion control | 0.959 ± 0.006 | 0.900 ± 0.021 | 0.738 ± 0.031 | 0.814 ± 0.023 | 0.750 ± 0.052 |

20% means 900 unique training episodes and 16,534 sampled transitions per epoch (130 optimizer updates at batch size 128). AB/ABC currently have 20% and 100% results; no 50% result is claimed.

**Interpretation:** compare the full method against BC30 first. To assess C separately, ABC minus AB is S6N3 +0.00 pp, S7N4 -1.56 pp, S8N6 -1.44 pp. A monotonic improvement is not assumed; three training seeds do not establish significance for small differences.

## 3. Failures and diagnostic cases

Failure categories are heuristic behavior diagnoses, not proofs of exclusive causes.

| Method | Total failures /300 | Grounding-related | Coverage/loop-related |
|---|---:|---:|---:|
| BC30 | 55.667 ± 13.597 | 38.333 ± 5.249 | 17.333 ± 8.994 |
| ABC: Map + RelBias | 7.667 ± 0.471 | 1.667 ± 1.247 | 6.000 ± 1.633 |
| A: Spatial-Softmax | 8.667 ± 0.943 | 3.667 ± 0.471 | 5.000 ± 0.816 |
| AB: Map-Inject | 7.667 ± 1.247 | 2.667 ± 0.471 | 5.000 ± 0.816 |
| FiLM | 12.333 ± 1.886 | 6.333 ± 1.247 | 6.000 ± 0.816 |

| Case | BC30 | ABC: Map + RelBias | A: Spatial-Softmax | AB: Map-Inject | FiLM |
|---|---:|---:|---:|---:|---:|
| 4501 | 1/3 | 2/3 | 1/3 | 3/3 | 2/3 |
| 4502 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| 4516 | 0/3 | 2/3 | 3/3 | 2/3 | 3/3 |
| 4560 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 4568 | 1/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 4582 | — | 3/3 | — | 3/3 | — |

Counts come from raw per-seed records; `—` means missing evidence. Cases 4502/4516/4560 diagnose attribute/distractor behavior, 4501 tests detouring, and 4568 illustrates a blind spot. BC30 solves 4568 in 1/3 runs: no universal impossibility claim follows.

## 4. Targeted probes

- **Language response:** change target words while holding observation/direction fixed. Action flips measure response, not correctness.
- **Navigation information:** H is preserved at concatenation; detour outcomes test whether the downstream policy uses useful information. Preservation does not guarantee navigation success.
- **Relation contribution:** compare AB vs ABC on task success and failures; a positive effect is not assumed.

| Method | Color-swap response | Type-swap response |
|---|---:|---:|
| AB: Map-Inject | 0.290 ± 0.016 | 0.339 ± 0.000 |
| ABC: Map + RelBias | 0.302 ± 0.019 | 0.328 ± 0.016 |

Feature geometry remains supplementary in `b6_georel_results.json#/probe2_feature_geometry`. The norm of `[H; M]` is not directly comparable to `M*H` as a universal grounding score.

## 5. Evidence and provenance

- Focused measurements and per-run records: `b6_georel_results.json`. Run configuration and training/validation curves: `runs/b6_georel_{s2_map1h,s3_map1h_relbias}*`.
- Frozen references: `b2_attn_v2_eval.json`, `b2_attn_v2_f020_eval.json`, `b2_attn_v2_failures.json`. Reference case counts are computed from `references.cases[model][seed].per_seed`.
- Original baseline reproduction, curves and cases: `baseline_report.md`, `baseline_eval.json`, `mission_breakdown.md`, `figures/` and `videos/`.
- Retained new models' per-mission breakdowns are in full-data evaluation run records. Raw source reports for historical controls remain available.
- Parent snapshot: `paradigm1-georel-unified@2609629`. Historical `git_commit=unknown` is not rewritten. Reruns record source hashes and environment versions; see the result JSON provenance.
- The existing presentation/report files describe previous experiments. This branch's narrative is specified in `docs/ATTENTION_STORY.md`.
