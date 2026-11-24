"""Evaluate AB / ABC and their language/feature/behavior diagnostics.

Frozen BC30, A and FiLM results are imported with explicit source paths.
The primary comparison is BC30 vs ABC; supporting comparisons are not a monotonic ladder.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import collections
import concurrent.futures
import contextlib
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b6_georel_unified"))
sys.path.insert(0, str(ROOT / "branches" / "b2_attn"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    ENV_ID,
    ENV_ID_S7,
    TEST_SEEDS,
    TEST_SEEDS_S7,
    encode_batch,
)
from compare_failures import MECHANISM, run_episode_with_policy  # noqa: E402
from eval_closed_loop import evaluate as evaluate_stateless  # noqa: E402
from eval_v2_and_analyze import ENV_ID_S8, SEEDS, load_run_universal  # noqa: E402
from film_policy import make_policy_fn  # noqa: E402
from georel_net import ALL_GEOREL_VARIANTS, build as build_georel  # noqa: E402
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT  # noqa: E402
from probe_feature_mechanics import extract_h_layers as extract_h_base  # noqa: E402
from probe_language import collect_states  # noqa: E402
from render_episodes import classify_failure  # noqa: E402

from study import CASE_SEEDS, ROOT, assemble_results, MODEL_KEYS

F020_VARIANTS = ALL_GEOREL_VARIANTS


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def load_any_model(run_dir: str | Path):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg.get("branch") == "b6_georel_unified":
        vocab = json.loads((run_dir / "vocab.json").read_text())
        a = cfg["arch"]
        model = build_georel(
            a["variant"],
            vocab_size=len(vocab["vocab"]),
            obj_emb_dim=a["obj_emb_dim"],
            lang_dim=a["lang_dim"],
            hidden=a.get("hidden", 128),
            dropout=a.get("dropout", 0.0),
        )
        state = torch.load(run_dir / "best_model.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, cfg, vocab
    return load_run_universal(run_dir)


def mean_sd(vals: list[float]) -> dict:
    vals = [float(v) for v in vals]
    return {
        "n": len(vals),
        "mean": round(st.mean(vals), 6),
        "sd": round(st.pstdev(vals), 6),
        "values": [round(v, 6) for v in vals],
    }


def _eval_worker(task: tuple[str, str, list[int], str]) -> dict:
    run_dir, env_id, seeds, key = task
    model, cfg, _ = load_any_model(run_dir)
    t0 = time.time()
    res = evaluate_stateless(make_policy_fn(model), env_id, seeds)
    return {
        "_key": key,
        "run": run_dir,
        "variant": cfg.get("arch", {}).get("variant", "concat"),
        "branch": cfg.get("branch", "baseline"),
        "train_seed": cfg.get("seed"),
        "frac": cfg.get("frac", 1.0),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "best_epoch": cfg.get("result", {}).get("best_epoch"),
        "n_params": cfg.get("arch", {}).get("n_params"),
        "train_seconds": cfg.get("result", {}).get("train_seconds"),
        "env_id": env_id,
        "eval_seconds": round(time.time() - t0, 2),
        **res,
    }


def summarize_group(runs: list[dict]) -> dict:
    runs = sorted(runs, key=lambda r: r["train_seed"])
    return {
        "n_params": runs[0]["n_params"],
        "closed_loop_success": mean_sd([r["closed_loop_success"] for r in runs]),
        "val_action_acc": mean_sd([r["val_action_acc"] for r in runs]),
        "mean_return": mean_sd([r["mean_return"] for r in runs]),
        "median_len_success": mean_sd([r["median_len_success"] for r in runs if r["median_len_success"] is not None]),
        "n_timeout": mean_sd([r["n_timeout"] for r in runs]),
        "runs": runs,
    }


# ------------------------------------------------------------------
# Probe 1: Behavioral Counterfactual Swap (183 multi-object states)
# ------------------------------------------------------------------
@torch.no_grad()
def probe_counterfactual_one(model, states: list[dict]) -> dict:
    same_type_tot = same_type_flip = 0
    same_color_tot = same_color_flip = 0
    for st_item in states:
        obs = {
            "image": st_item["image"],
            "direction": st_item["direction"],
            "mission": st_item["mission"],
        }
        base_img, base_tok, base_d = encode_batch([obs])
        base_out = model(base_img, base_tok, base_d)
        if isinstance(base_out, tuple):
            base_out = base_out[0]
        base_a = int(base_out.argmax(-1).item())
        if st_item.get("alt_color_mission"):
            obs2 = {**obs, "mission": st_item["alt_color_mission"]}
            im2, tk2, d2 = encode_batch([obs2])
            out2 = model(im2, tk2, d2)
            if isinstance(out2, tuple):
                out2 = out2[0]
            same_type_tot += 1
            if int(out2.argmax(-1).item()) != base_a:
                same_type_flip += 1
        if st_item.get("alt_type_mission"):
            obs2 = {**obs, "mission": st_item["alt_type_mission"]}
            im2, tk2, d2 = encode_batch([obs2])
            out2 = model(im2, tk2, d2)
            if isinstance(out2, tuple):
                out2 = out2[0]
            same_color_tot += 1
            if int(out2.argmax(-1).item()) != base_a:
                same_color_flip += 1
    return {
        "color_swap_rate": round(same_type_flip / max(1, same_type_tot), 6),
        "type_swap_rate": round(same_color_flip / max(1, same_color_tot), 6),
        "n_color_pairs": same_type_tot,
        "n_type_pairs": same_color_tot,
    }


# ------------------------------------------------------------------
# Probe 2: Hidden-Layer Feature Geometry (SNR, Selectivity, Drift)
# ------------------------------------------------------------------
def extract_h_universal(model, img: torch.Tensor, tok: torch.Tensor, direc: torch.Tensor):
    if hasattr(model, "extract_layers"):
        with torch.no_grad():
            h1, h_mod, h2, _ = model.extract_layers(img, tok)
            return h1, h_mod, h2
    return extract_h_base(model, img, tok, direc)


def collect_probe_scenes(env_id: str, seeds: list[int], max_scenes: int = 150) -> list[dict]:
    scenes = []
    env = gym.make(env_id)
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            mission = obs["mission"]
            parts = mission.split()
            t_color, t_type = parts[-2], parts[-1]
            im = obs["image"]  # (7,7,3) -> (col, row, ch) in MiniGrid!
            targets, distractors, partials, bg = [], [], [], []
            for i in range(7):
                for j in range(7):
                    oid, cid = int(im[i, j, 0]), int(im[i, j, 1])
                    oname = IDX_TO_OBJECT.get(oid, "unseen")
                    cname = IDX_TO_COLOR.get(cid, "red")
                    if oname in ("ball", "box", "key"):
                        if cname == t_color and oname == t_type:
                            targets.append((i, j))
                        else:
                            distractors.append((i, j))
                            if cname == t_color or oname == t_type:
                                partials.append((i, j))
                    else:
                        bg.append((i, j))
            if len(targets) == 1 and len(distractors) >= 1:
                alt_color = next(c for c in ("red", "green", "blue", "purple", "yellow", "grey") if c != t_color)
                alt_mission = " ".join(parts[:-2] + [alt_color, t_type])
                scenes.append(
                    {
                        "seed": seed,
                        "obs": obs,
                        "alt_obs": {**obs, "mission": alt_mission},
                        "target": targets[0],
                        "distractors": distractors,
                        "partials": partials,
                        "bg": bg,
                    }
                )
            if len(scenes) >= max_scenes:
                break
    env.close()
    return scenes


@torch.no_grad()
def probe_feature_geometry_one(model, scenes: list[dict]) -> dict:
    hmod_ratios, h2_ratios, h2_part_ratios, selectivities, drifts = [], [], [], [], []
    for sc in scenes:
        img, tok, d = encode_batch([sc["obs"]])
        _, tok2, _ = encode_batch([sc["alt_obs"]])
        _, h_mod, h2 = extract_h_universal(model, img, tok, d)
        _, _, h2_alt = extract_h_universal(model, img, tok2, d)

        ti, tj = sc["target"]
        # h_mod & h2 shape (1, C, 7, 7) with spatial dims matching im (i, j)
        n_mod = h_mod[0].norm(dim=0).numpy()  # (7, 7)
        n_h2 = h2[0].norm(dim=0).numpy()  # (7, 7)
        dh2 = (h2[0] - h2_alt[0]).norm(dim=0).numpy()  # (7, 7)

        t_mod = float(n_mod[ti, tj])
        d_mod = max(float(n_mod[di, dj]) for di, dj in sc["distractors"])
        hmod_ratios.append(t_mod / max(d_mod, 1e-6))

        t_h2 = float(n_h2[ti, tj])
        d_h2 = max(float(n_h2[di, dj]) for di, dj in sc["distractors"])
        h2_ratios.append(t_h2 / max(d_h2, 1e-6))

        if sc["partials"]:
            p_h2 = max(float(n_h2[pi, pj]) for pi, pj in sc["partials"])
            h2_part_ratios.append(t_h2 / max(p_h2, 1e-6))

        bg_dh2 = float(np.mean([dh2[bi, bj] for bi, bj in sc["bg"]]))
        t_dh2 = float(dh2[ti, tj])
        if bg_dh2 > 1e-6:
            selectivities.append(t_dh2 / bg_dh2)
        bg_norm = float(np.mean([n_h2[bi, bj] for bi, bj in sc["bg"]]))
        if bg_norm > 1e-6:
            drifts.append(bg_dh2 / bg_norm)

    return {
        "h_mod_T_over_maxD": round(float(np.mean(hmod_ratios)), 4),
        "h2_T_over_maxD": round(float(np.mean(h2_ratios)), 4),
        "h2_T_over_partialD": round(float(np.mean(h2_part_ratios)), 4) if h2_part_ratios else None,
        "spatial_selectivity": round(float(np.mean(selectivities)), 4) if selectivities else 0.0,
        "background_drift": round(float(np.mean(drifts)), 4) if drifts else 0.0,
    }


# ------------------------------------------------------------------
# Probe 3: 300-Episode Failure Taxonomy & Diagnostic Cases
# ------------------------------------------------------------------
def _failure_worker(run_dir: str) -> dict:
    model, cfg, _ = load_any_model(run_dir)
    env = gym.make(ENV_ID)
    cats = collections.Counter()
    mechs = collections.Counter()
    cases = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TEST_SEEDS:
            ep = run_episode_with_policy(env, make_policy_fn(model), seed)
            succ = bool(ep["success"])
            cat_raw = None if succ else classify_failure(ep)
            cat = "success" if succ else (cat_raw[0] if isinstance(cat_raw, tuple) else str(cat_raw))
            if not succ:
                cats[cat] += 1
                mechs[MECHANISM.get(cat, "state_coverage")] += 1
            if seed in CASE_SEEDS:
                cases[str(seed)] = {
                    "success": succ,
                    "steps": int(len(ep["steps"])),
                    "category": cat,
                }
    env.close()
    return {
        "run": run_dir,
        "train_seed": cfg.get("seed"),
        "total_failures": int(sum(cats.values())),
        "repr_failures": int(mechs.get("representation", 0)),
        "cov_failures": int(mechs.get("state_coverage", 0)),
        "categories": dict(cats),
        "cases": cases,
    }


def main() -> None:
    import argparse
    import importlib.util
    # Historical branches also expose make_summary; bind this sibling explicitly.
    spec = importlib.util.spec_from_file_location("attention_summary", Path(__file__).with_name("make_summary.py"))
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be positive")
    start = time.time()
    tasks = []
    for mode in ("full", "f020"):
        suffix = "" if mode == "full" else "_f020"
        for variant in ALL_GEOREL_VARIANTS:
            for seed in SEEDS:
                run = ROOT / f"runs/b6_georel_{variant}{suffix}_seed{seed}"
                for name in ("best_model.pt", "config.json", "vocab.json"):
                    if not (run / name).is_file():
                        raise FileNotFoundError(f"Incomplete required run: {run / name}")
                envs = [("S6N3", ENV_ID, TEST_SEEDS), ("S7N4", ENV_ID_S7, TEST_SEEDS_S7)]
                if mode == "full":
                    envs.append(("S8N6", ENV_ID_S8, TEST_SEEDS_S7))
                tasks += [(str(run), eid, seeds, f"{mode}|{variant}|{tag}") for tag, eid, seeds in envs]
    grouped = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        for row in pool.map(_eval_worker, tasks):
            mode, variant, env = row.pop("_key").split("|")
            grouped[mode][variant][env].append(row)
            print(f"[eval] {mode} {variant} {env} seed={row['train_seed']} success={row['closed_loop_success']:.4f}", flush=True)
    saved = {
        section: {v: {env: summarize_group(rows) for env, rows in envs.items()} for v, envs in grouped[mode].items()}
        for section, mode in (("eval", "full"), ("f020", "f020"))
    }
    saved["failures"] = {}
    fail_runs = [(variant, ROOT / f"runs/b6_georel_{variant}_seed{seed}") for variant in ALL_GEOREL_VARIANTS for seed in SEEDS]
    by_variant = collections.defaultdict(list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        for (variant, _), row in zip(fail_runs, pool.map(_failure_worker, [str(run) for _, run in fail_runs])):
            by_variant[variant].append(row)
    for variant, rows in by_variant.items():
        rows.sort(key=lambda r: r["train_seed"])
        saved["failures"][variant] = {
            field: mean_sd([row[field] for row in rows])
            for field in ("total_failures", "repr_failures", "cov_failures")
        }
        saved["failures"][variant].update({
            "cases": {str(seed): {
                "solved": sum(bool(row["cases"][str(seed)]["success"]) for row in rows),
                "total": len(rows), "steps": [row["cases"][str(seed)]["steps"] for row in rows],
            } for seed in CASE_SEEDS}, "runs": rows,
        })
    states = collect_states(n_states=300, scan=300)
    scenes = {"S6N3": collect_probe_scenes(ENV_ID, TEST_SEEDS), "S8N6": collect_probe_scenes(ENV_ID_S8, TEST_SEEDS_S7)}
    saved["probe1_counterfactual"] = {}
    saved["probe2_feature_geometry"] = {env: {} for env in scenes}
    for variant in ALL_GEOREL_VARIANTS:
        p1, p2 = [], {env: [] for env in scenes}
        for seed in SEEDS:
            model, _, _ = load_any_model(ROOT / f"runs/b6_georel_{variant}_seed{seed}")
            p1.append(probe_counterfactual_one(model, states))
            for env, inputs in scenes.items():
                p2[env].append(probe_feature_geometry_one(model, inputs))
        saved["probe1_counterfactual"][variant] = {field: mean_sd([p[field] for p in p1]) for field in ("color_swap_rate", "type_swap_rate")}
        for env, rows in p2.items():
            saved["probe2_feature_geometry"][env][variant] = {
                field: mean_sd([r[field] for r in rows if r[field] is not None]) for field in rows[0]
            }
    import platform
    import hashlib
    saved["provenance"] = {
        "kind": "fresh evaluation of retained AB/ABC checkpoints; frozen controls imported from saved source reports",
        "host": platform.node(), "python": platform.python_version(),
        "torch": torch.__version__, "gymnasium": gym.__version__, "minigrid": minigrid.__version__,
        "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.time()-start, 2),
        "checkpoint_sources": {variant: [json.loads((ROOT / f"runs/b6_georel_{variant}_seed{seed}/config.json").read_text()) for seed in SEEDS] for variant in ALL_GEOREL_VARIANTS},
    }
    result = assemble_results(saved)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "b6_georel_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    summary.render_summary(result, args.out_dir)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
