"""Render the focused ABC study from saved results; no training dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from study import CASE_SEEDS, MODELS, NEW_VARIANTS, PRIMARY, RESULT_PATH, ROOT, assemble_results, validate_results


def fmt(metric: dict) -> str:
    return f"{metric['mean']:.3f} ± {metric['sd']:.3f}"


def render_summary(res: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    validate_results(res)
    out_dir = ROOT / "reports" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {k: label for k, label, _ in MODELS}
    c_deltas = ", ".join(
        f"{env} {100 * (res['eval']['s3_map1h_relbias'][env]['closed_loop_success']['mean'] - res['eval']['s2_map1h'][env]['closed_loop_success']['mean']):+.2f} pp"
        for env in ("S6N3", "S7N4", "S8N6")
    )
    lines = [
        "# ABC attention fusion — one primary comparison", "",
        "**Primary modification:** single-frame language-conditioned spatial fusion with raw-feature/map concatenation and relative-bias relational attention (ABC).",
        "**Primary comparison:** BC30 vs ABC (`s3_map1h_relbias`). A, AB and FiLM are supporting controls.", "",
        "4,500 training + 500 validation expert episodes; Adam lr=1e-3, wd=0; 3 initialization seeds; 300 unseen episodes per environment. Values are mean ± population SD (ddof=0).",
        "**Protocol deviation:** 30 full epochs without early stopping, checkpoint selected by validation action accuracy only. Original early-stopped baseline: `baseline_report.md`.", "",
        "## 1. Primary comparison", "",
        "| Method | Params | S6N3 success | Mean return | Median successful length (steps) | Val action accuracy | S7N4 success | S8N6 success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in PRIMARY:
        e = res["eval"][key]
        s = e["S6N3"]
        success = fmt(s["closed_loop_success"])
        if key == PRIMARY[1]:
            success = f"**{success}**"
        lines.append(f"| {labels[key]} | {s['n_params']:,} | {success} | {fmt(s['mean_return'])} | {fmt(s['median_len_success'])} | {fmt(s['val_action_acc'])} | {fmt(e['S7N4']['closed_loop_success'])} | {fmt(e['S8N6']['closed_loop_success'])} |")
    lines += ["", "| Method | Seed 0 S6N3 | Seed 1 S6N3 | Seed 2 S6N3 |", "|---|---:|---:|---:|"]
    for key in PRIMARY:
        vals = res["eval"][key]["S6N3"]["closed_loop_success"]["values"]
        lines.append(f"| {labels[key]} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += [
        "", "## 2. Component ablations and alternative fusion", "",
        "A applies multiplicative weighting. AB replaces it with `[H; M] → Conv2`; ABC adds C after Conv2. AB vs ABC isolates C.", "",
        "| Method | Role | S6N3 success | S7N4 success | S8N6 success | 20% S6N3 | 20% S7N4 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, label, role in MODELS:
        e, f = res["eval"][key], res["f020"][key]
        values = [fmt(e[tag]["closed_loop_success"]) for tag in ("S6N3", "S7N4", "S8N6")]
        values += [fmt(f[tag]["closed_loop_success"]) for tag in ("S6N3", "S7N4")]
        lines.append(f"| {label} | {role} | " + " | ".join(values) + " |")
    lines += [
        "", "20% means 900 unique training episodes and 16,534 sampled transitions per epoch (130 optimizer updates at batch size 128). AB/ABC currently have 20% and 100% results; no 50% result is claimed.",
        "", f"**Interpretation:** compare the full method against BC30 first. To assess C separately, ABC minus AB is {c_deltas}. A monotonic improvement is not assumed; three training seeds do not establish significance for small differences.",
        "", "## 3. Failures and diagnostic cases", "",
        "Failure categories are heuristic behavior diagnoses, not proofs of exclusive causes.", "",
        "| Method | Total failures /300 | Grounding-related | Coverage/loop-related |", "|---|---:|---:|---:|",
    ]
    for key, label, _ in MODELS:
        f = res["failures"][key]
        lines.append(f"| {label} | {fmt(f['total_failures'])} | {fmt(f['repr_failures'])} | {fmt(f['cov_failures'])} |")
    lines += ["", "| Case | " + " | ".join(labels[k] for k, _, _ in MODELS) + " |", "|---|" + "---:|" * len(MODELS)]
    for seed in CASE_SEEDS:
        cells = []
        for key, _, _ in MODELS:
            c = res["failures"][key]["cases"].get(str(seed))
            cells.append(f"{c['solved']}/{c['total']}" if c else "—")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines += [
        "", "Counts come from raw per-seed records; `—` means missing evidence. Cases 4502/4516/4560 diagnose attribute/distractor behavior, 4501 tests detouring, and 4568 illustrates a blind spot. BC30 solves 4568 in 1/3 runs: no universal impossibility claim follows.",
        "", "## 4. Targeted probes", "",
        "- **Language response:** change target words while holding observation/direction fixed. Action flips measure response, not correctness.",
        "- **Navigation information:** H is preserved at concatenation; detour outcomes test whether the downstream policy uses useful information. Preservation does not guarantee navigation success.",
        "- **Relation contribution:** compare AB vs ABC on task success and failures; a positive effect is not assumed.", "",
        "| Method | Color-swap response | Type-swap response |", "|---|---:|---:|",
    ]
    for key in NEW_VARIANTS:
        p = res["probe1_counterfactual"][key]
        lines.append(f"| {labels[key]} | {fmt(p['color_swap_rate'])} | {fmt(p['type_swap_rate'])} |")
    lines += [
        "", "Feature geometry remains supplementary in `b6_georel_results.json#/probe2_feature_geometry`. The norm of `[H; M]` is not directly comparable to `M*H` as a universal grounding score.",
        "", "## 5. Evidence and provenance", "",
        "- Focused measurements and per-run records: `b6_georel_results.json`. Run configuration and training/validation curves: `runs/b6_georel_{s2_map1h,s3_map1h_relbias}*`.",
        "- Frozen references: `b2_attn_v2_eval.json`, `b2_attn_v2_f020_eval.json`, `b2_attn_v2_failures.json`. Reference case counts are computed from `references.cases[model][seed].per_seed`.",
        "- Original baseline reproduction, curves and cases: `baseline_report.md`, `baseline_eval.json`, `mission_breakdown.md`, `figures/` and `videos/`.",
        "- Retained new models' per-mission breakdowns are in full-data evaluation run records. Raw source reports for historical controls remain available.",
        "- Parent snapshot: `paradigm1-georel-unified@2609629`. Historical `git_commit=unknown` is not rewritten. Reruns record source hashes and environment versions; see the result JSON provenance.",
        "- The existing presentation/report files describe previous experiments. This branch's narrative is specified in `docs/ATTENTION_STORY.md`.", "",
    ]
    md = out_dir / "b6_georel_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), dpi=180)
    keys = [k for k, _, _ in MODELS]
    names = ["BC30", "ABC\n(primary)", "A", "AB", "FiLM"]
    x = np.arange(len(keys))
    for j, (env, color) in enumerate(zip(("S6N3", "S7N4", "S8N6"), ("#2355A4", "#19806B", "#AD651D"))):
        v = [res["eval"][k][env]["closed_loop_success"] for k in keys]
        axes[0].bar(x+(j-1)*.25, [m["mean"] for m in v], .25, yerr=[m["sd"] for m in v], capsize=2, label=env, color=color)
    axes[0].set_title("Primary comparison and supporting controls")
    axes[0].set_ylabel("Closed-loop success (mean ± SD)")
    axes[0].set_ylim(0, 1.08)
    axes[0].legend(fontsize=8)
    for j, env in enumerate(("S6N3", "S7N4")):
        v = [res["f020"][k][env]["closed_loop_success"] for k in keys]
        axes[1].bar(x+(j-.5)*.34, [m["mean"] for m in v], .34, yerr=[m["sd"] for m in v], capsize=2, label=env, color=("#2355A4", "#19806B")[j])
    axes[1].set_title("20% training episodes; update-matched")
    axes[1].set_ylabel("Closed-loop success (mean ± SD)")
    axes[1].set_ylim(0, 1.08)
    axes[1].legend(fontsize=8)
    rep = [res["failures"][k]["repr_failures"]["mean"] for k in keys]
    cov = [res["failures"][k]["cov_failures"]["mean"] for k in keys]
    axes[2].bar(x, rep, color="#AD514A", label="Grounding-related")
    axes[2].bar(x, cov, bottom=rep, color="#638EBD", label="Coverage / loops")
    axes[2].set_title("Heuristic failure categories (S6N3)")
    axes[2].set_ylabel("Mean failures / 300 episodes")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, names)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("ABC fusion · 3 initialization seeds · 300 unseen episodes / environment", fontsize=13)
    fig.tight_layout()
    png = out_dir / "figures/b6_georel_unified/b6_attention_comparison.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return md, png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus-results", action="store_true", help="Filter inherited JSON and repair reference adapters")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    res = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if args.focus_results:
        res = assemble_results(res)
        RESULT_PATH.write_text(json.dumps(res, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    for path in render_summary(res, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
