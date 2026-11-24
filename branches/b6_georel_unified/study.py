"""Scope and source adapters for the five-model ABC fusion study (stdlib only)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "reports/b6_georel_results.json"
PRIMARY = ("bc30", "s3_map1h_relbias")
NEW_VARIANTS = ("s2_map1h", "s3_map1h_relbias")
MODELS = (
    ("bc30", "BC30", "Baseline"),
    ("s3_map1h_relbias", "ABC: Map + RelBias", "Primary method"),
    ("spatial_softmax_attn", "A: Spatial-Softmax", "Fusion-design ablation"),
    ("s2_map1h", "AB: Map-Inject", "Remove C from ABC"),
    ("film_nwd", "FiLM", "Alternative fusion control"),
)
MODEL_KEYS = tuple(k for k, _, _ in MODELS)
CASE_SEEDS = (4501, 4502, 4516, 4560, 4568, 4582)


def read_report(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def reference_results() -> dict:
    """Read numeric references from saved experiments; never hard-code case outcomes."""
    full = read_report("b2_attn_v2_eval.json")
    small = read_report("b2_attn_v2_f020_eval.json")
    failures = read_report("b2_attn_v2_failures.json")
    result = {"eval": {}, "f020": {}, "failures": {}}
    aliases = {
        "bc30": ("bc30", "bc_step_f020"),
        "spatial_softmax_attn": ("v1_spatial_softmax_attn", "v1_spatial_softmax_attn"),
        "film_nwd": ("film_nwd", "film_step_f020"),
    }
    for key, (full_key, small_key) in aliases.items():
        result["eval"][key] = copy.deepcopy(full["references"][full_key])
        container = small["references"] if small_key in small["references"] else small["variants"]
        envs = copy.deepcopy(container[small_key])
        result["f020"][key] = {
            env: value if "closed_loop_success" in value else {
                "closed_loop_success": {"mean": value["success_mean"], "sd": value["success_sd"]}
            }
            for env, value in envs.items() if env in ("S6N3", "S7N4")
        }
        stats = failures["references"]["aggregate"][key]
        raw_cases = failures["references"]["cases"][key]
        cases = {}
        for seed in CASE_SEEDS:
            raw = raw_cases.get(str(seed))
            if raw is None:
                continue
            runs = raw["per_seed"]
            solved = sum(bool(run["success"]) for run in runs)
            if solved != raw["n_solved"]:
                raise ValueError(f"Inconsistent source case count: {key}/{seed}")
            cases[str(seed)] = {
                "solved": solved, "total": len(runs),
                "steps": [run["episode_len"] for run in runs],
                "per_seed": copy.deepcopy(runs),
            }
        result["failures"][key] = {
            "total_failures": copy.deepcopy(stats["n_failure"]),
            "repr_failures": copy.deepcopy(stats["representation"]),
            "cov_failures": copy.deepcopy(stats["state_coverage"]),
            "cases": cases,
            "source": f"reports/b2_attn_v2_failures.json#/references/aggregate/{key}",
        }
    return result


def assemble_results(saved: dict) -> dict:
    """Select AB/ABC without changing their measurements; attach frozen controls."""
    result = reference_results()
    for section in ("eval", "f020", "failures"):
        for key in NEW_VARIANTS:
            result[section][key] = copy.deepcopy(saved[section][key])
        result[section] = {key: result[section][key] for key in MODEL_KEYS}
    result["probe1_counterfactual"] = {
        key: copy.deepcopy(saved["probe1_counterfactual"][key]) for key in NEW_VARIANTS
    }
    result["probe2_feature_geometry"] = {
        env: {key: copy.deepcopy(saved["probe2_feature_geometry"][env][key]) for key in NEW_VARIANTS}
        for env in ("S6N3", "S8N6")
    }
    result["study"] = {
        "branch": "zez-attention-method", "primary_comparison": list(PRIMARY),
        "models": [{"key": k, "label": label, "role": role} for k, label, role in MODELS],
        "sd_convention": "population SD, ddof=0, 3 initialization seeds",
        "reference_sources": ["reports/b2_attn_v2_eval.json", "reports/b2_attn_v2_f020_eval.json", "reports/b2_attn_v2_failures.json"],
        "interpretation": [
            "A uses multiplicative weighting; AB replaces it with raw-feature/map concatenation.",
            "AB vs ABC isolates the relation block; results need not improve monotonically.",
            "Full 30-epoch selection without early stopping is a disclosed handbook deviation.",
            "Feature norms across multiplicative and concatenated representations are not a common quality score.",
            "Action flips measure responsiveness, not correctness; failure classes are heuristic diagnoses.",
        ],
    }
    result["provenance"] = copy.deepcopy(saved.get("provenance", {
        "retained_measurements": "paradigm1-georel-unified@2609629:reports/b6_georel_results.json",
        "note": "Existing measurements selected without retraining; historical git_commit=unknown remains disclosed.",
    }))
    validate_results(result)
    return result


def validate_results(result: dict) -> None:
    for section in ("eval", "f020", "failures"):
        if set(result[section]) != set(MODEL_KEYS):
            raise ValueError(f"Unexpected or missing models in {section}")
    for key in MODEL_KEYS:
        for env in ("S6N3", "S7N4", "S8N6"):
            metric = result["eval"][key][env]["closed_loop_success"]
            if metric["n"] != 3 or len(metric["values"]) != 3:
                raise ValueError(f"Expected all three seeds: {key}/{env}")
        for env in ("S6N3", "S7N4"):
            metric = result["f020"][key][env]["closed_loop_success"]
            if not 0 <= metric["mean"] <= 1 or metric["sd"] < 0:
                raise ValueError(f"Invalid low-data metric: {key}/{env}")
        fail = result["failures"][key]
        expected = 300 * (1 - result["eval"][key]["S6N3"]["closed_loop_success"]["mean"])
        if abs(fail["total_failures"]["mean"] - expected) > 0.002:
            raise ValueError(f"Failure count / success mismatch: {key}")
        if abs(fail["repr_failures"]["mean"] + fail["cov_failures"]["mean"] - fail["total_failures"]["mean"]) > 0.002:
            raise ValueError(f"Failure categories do not sum to total: {key}")
        for seed, case in fail["cases"].items():
            if case["total"] != 3 or not 0 <= case["solved"] <= 3:
                raise ValueError(f"Invalid case count: {key}/{seed}")
    if set(result["probe1_counterfactual"]) != set(NEW_VARIANTS):
        raise ValueError("Counterfactual probes must contain AB and ABC only")
    for env in ("S6N3", "S8N6"):
        if set(result["probe2_feature_geometry"][env]) != set(NEW_VARIANTS):
            raise ValueError(f"Feature probes must contain AB and ABC only: {env}")
