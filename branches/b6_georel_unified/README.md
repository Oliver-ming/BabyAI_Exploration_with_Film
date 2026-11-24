# Focused ABC fusion implementation

Primary model: `s3_map1h_relbias` (ABC). Direct ablation: `s2_map1h` (AB).
A and FiLM reuse frozen reference implementations; BC30 is the baseline.

- `georel_net.py`: checkpoint-compatible retained network definitions.
- `train_georel.py`: stateless trainer, source/data provenance, protected outputs.
- `run_attention_all.sh`: 12 retained runs (2 models × 2 fractions × 3 seeds), then evaluation.
- `eval_georel.py`: requires complete runs; saves per-episode rollouts, mission results and probes.
- `study.py`: five-model scope and source adapters; reference cases are read from per-seed JSON.
- `make_summary.py`: focused tables and comparison figure; no simulator required.

No GRU or residual alternative is exposed here. The old directory name remains to keep checkpoint/run paths compatible. See ../../docs/ATTENTION_STORY.md and ../../reports/b6_georel_summary.md.
