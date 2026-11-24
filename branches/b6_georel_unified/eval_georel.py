"""Branch 6 · Paradigm 1 Unified Evaluation, 3-Probe Mechanistic Analysis, Summary & Figure.

Evaluates the complete Paradigm 1 composable ladder against frozen controls (BC30, FiLM, v1 Spatial-SS,
v2 Map-Inject, v2 MH-Spatial-Relational):
  - Multi-scale closed-loop success: S6N3 (4500:4800), zero-shot S7N4 & S8N6 (10000:10300)
  - 20% step-matched low-data scaling (S6N3 & S7N4)
  - Probe 1: Counterfactual language swap (183 multi-object states)
  - Probe 2: Hidden-layer SNR, spatial selectivity, background drift (150 scenes x 3 seeds)
             + Seed 4501 obstacle multiplier check
  - Probe 3: 300-episode closed-loop failure taxonomy (Representation Grounding vs Coverage Loops)
             + 6 diagnostic case seeds (4501, 4502, 4516, 4560, 4568, 4582)

Usage:
    OMP_NUM_THREADS=1 env_embodied/bin/python branches/b6_georel_unified/eval_georel.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from georel_net import ALL_GEOREL_VARIANTS, RECURRENT_VARIANTS, build as build_georel  # noqa: E402
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT  # noqa: E402
from probe_feature_mechanics import extract_h_layers as extract_h_base  # noqa: E402
from probe_language import collect_states  # noqa: E402
from render_episodes import classify_failure  # noqa: E402

CASE_SEEDS = [4501, 4502, 4516, 4560, 4568, 4582]
F020_VARIANTS = ("s2_map1h", "s3_map1h_relbias", "s3_res1h_relbias", "s4_georel_gru")

# Ordered Paradigm 1 ladder (existing references + new b6 composable variants)
LADDER_MODELS = [
    ("bc30", "runs/b3_bc30", "0. BC30 (Concat)"),
    ("film_nwd", "runs/b2_film_nwd", "0'. Channel FiLM"),
    ("spatial_softmax_attn", "runs/b2_attn_spatial_softmax_attn", "1. +A: 1H Spatial-SS"),
    ("s2_map1h", "runs/b6_georel_s2_map1h", "2. +A+B: 1H Map-Inject"),
    ("s3_map1h_relbias", "runs/b6_georel_s3_map1h_relbias", "3. +A+B+C: Map1H+RelBias"),
    ("s3_res1h_relbias", "runs/b6_georel_s3_res1h_relbias", "3'. +A+B'+C: Res1H+RelBias"),
    ("mh_spatial_relational", "runs/b2_attn_v2_mh_spatial_relational", "ref: v2 4H MH-Rel"),
    ("bc30_gru", "runs/b6_georel_bc30_gru", "ctrl: BC30 + GRU"),
    ("s4_georel_gru", "runs/b6_georel_s4_georel_gru", "4. +A+B+C+D: GeoRel-GRU"),
    ("s4_res1h_gru", "runs/b6_georel_s4_res1h_gru", "4'. +A+B'+C+D: Res1H-GRU"),
]


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


def evaluate_recurrent_closed_loop(model, env_id: str, seeds: list[int]) -> dict:
    """Exact mirror of frozen `src/eval_closed_loop.py:evaluate()` with per-episode hx reset."""
    env = gym.make(env_id)
    successes, lengths, returns, trajectories = [], [], [], []
    action_ctr = collections.Counter()
    per_mission = collections.defaultdict(lambda: {"n": 0, "success": 0})

    with contextlib.redirect_stdout(io.StringIO()):
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            mission = obs["mission"]
            hx = None
            ep_actions, ep_rewards = [], []
            terminated = truncated = False
            for _t in range(env.unwrapped.max_steps):
                img, tok, d = encode_batch([obs])
                with torch.no_grad():
                    logits, hx = model.step(img, tok, d, hx=hx)
                    a = int(logits.argmax(-1).item())
                ep_actions.append(a)
                obs, r, terminated, truncated, _ = env.step(a)
                ep_rewards.append(float(r))
                if terminated or truncated:
                    break

            action_ctr.update(ep_actions)
            m = per_mission[mission]
            m["n"] += 1
            m["success"] += int(terminated)
            successes.append(bool(terminated))
            returns.append(sum(ep_rewards))
            if terminated:
                lengths.append(len(ep_actions))
            trajectories.append(
                {
                    "seed": seed,
                    "mission": mission,
                    "success": bool(terminated),
                    "truncated": bool(truncated),
                    "episode_len": len(ep_actions),
                    "actions": ep_actions,
                }
            )
    env.close()

    n = len(seeds)
    total_actions = sum(action_ctr.values())
    return {
        "n_episodes": n,
        "closed_loop_success": round(float(np.mean(successes)), 6),
        "mean_return": round(float(np.mean(returns)), 6),
        "median_len_success": float(np.median(lengths)) if lengths else None,
        "mean_len_success": round(float(np.mean(lengths)), 3) if lengths else None,
        "mean_len_all": round(float(np.mean([t["episode_len"] for t in trajectories])), 3),
        "n_timeout": int(sum(t["truncated"] for t in trajectories)),
        "per_mission": {
            k: {"n": v["n"], "success": round(v["success"] / v["n"], 6)}
            for k, v in sorted(per_mission.items())
        },
        "action_histogram": {
            str(k): round(action_ctr[k] / max(1, total_actions), 4) for k in range(7)
        },
    }


def _eval_worker(task: tuple[str, str, list[int], str]) -> dict:
    run_dir, env_id, seeds, key = task
    model, cfg, _ = load_any_model(run_dir)
    t0 = time.time()
    if getattr(model, "is_recurrent", False):
        res = evaluate_recurrent_closed_loop(model, env_id, seeds)
    else:
        res = evaluate_stateless(make_policy_fn(model), env_id, seeds)
        res.pop("_trajectories", None)
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
def make_episode_policy(model):
    if not getattr(model, "is_recurrent", False):
        return make_policy_fn(model)
    state = {"hx": None}

    @torch.no_grad()
    def _pol(obs) -> int:
        img, tok, d = encode_batch([obs])
        logits, hx = model.step(img, tok, d, hx=state["hx"])
        state["hx"] = hx
        return int(logits.argmax(-1).item())

    return _pol


def _failure_worker(run_dir: str) -> dict:
    model, cfg, _ = load_any_model(run_dir)
    env = gym.make(ENV_ID)
    cats = collections.Counter()
    mechs = collections.Counter()
    cases = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TEST_SEEDS:
            ep = run_episode_with_policy(env, make_episode_policy(model), seed)
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


def make_ladder_figure(eval_summary: dict, f020_summary: dict, fail_summary: dict, out_path: Path) -> None:
    keys = [k for k, _, _ in LADDER_MODELS if k in eval_summary]
    labels = [lbl for k, _, lbl in LADDER_MODELS if k in eval_summary]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=220)
    x = np.arange(len(keys))

    # Panel A: S6N3, S7N4, S8N6
    ax = axes[0]
    w = 0.26
    for idx, (env_k, col, hatch, name) in enumerate([
        ("S6N3", "#2563EB", "", "S6N3 (6x6, ID)"),
        ("S7N4", "#0F766E", "//", "S7N4 (7x7, OOD)"),
        ("S8N6", "#D97706", "..", "S8N6 (8x8, OOD)"),
    ]):
        means = [eval_summary[k][env_k]["closed_loop_success"]["mean"] for k in keys]
        sds = [eval_summary[k][env_k]["closed_loop_success"]["sd"] for k in keys]
        ax.bar(x + (idx - 1) * w, means, width=w, yerr=sds, capsize=2.5,
               color=col, alpha=0.88 if idx == 0 else 0.78, edgecolor="#0F172A",
               linewidth=0.6, hatch=hatch, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8.5)
    ax.set_ylim(0.42, 1.03)
    ax.set_ylabel("Closed-Loop Success (300 unseen eps)")
    ax.set_title("(a) Paradigm 1 Composable Ladder: S6N3 / S7N4 / S8N6", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right")

    # Panel B: 20% low-data S6N3 & S7N4
    ax = axes[1]
    fkeys = [k for k in keys if k in f020_summary]
    flabels = [lbl for k, _, lbl in LADDER_MODELS if k in fkeys]
    fx = np.arange(len(fkeys))
    w2 = 0.34
    s6m = [f020_summary[k]["S6N3"]["closed_loop_success"]["mean"] for k in fkeys]
    s6s = [f020_summary[k]["S6N3"]["closed_loop_success"]["sd"] for k in fkeys]
    s7m = [f020_summary[k]["S7N4"]["closed_loop_success"]["mean"] for k in fkeys]
    s7s = [f020_summary[k]["S7N4"]["closed_loop_success"]["sd"] for k in fkeys]
    ax.bar(fx - w2 / 2, s6m, width=w2, yerr=s6s, capsize=2.5, color="#1D4ED8", edgecolor="#0F172A", linewidth=0.6, label="20% S6N3")
    ax.bar(fx + w2 / 2, s7m, width=w2, yerr=s7s, capsize=2.5, color="#0D9488", edgecolor="#0F172A", linewidth=0.6, hatch="//", label="20% S7N4 (OOD)")
    ax.set_xticks(fx)
    ax.set_xticklabels(flabels, rotation=32, ha="right", fontsize=8.5)
    ax.set_ylim(0.55, 1.00)
    ax.set_ylabel("Closed-Loop Success (20% = 900 eps)")
    ax.set_title("(b) Step-Matched 20% Low-Data Generalization", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, loc="lower right")

    # Panel C: 300-Episode Failure Taxonomy (Repr vs Coverage)
    ax = axes[2]
    tkeys = [k for k in keys if k in fail_summary]
    tlabels = [lbl for k, _, lbl in LADDER_MODELS if k in tkeys]
    tx = np.arange(len(tkeys))
    repr_m = [fail_summary[k]["repr_failures"]["mean"] for k in tkeys]
    cov_m = [fail_summary[k]["cov_failures"]["mean"] for k in tkeys]
    ax.bar(tx, repr_m, width=0.58, color="#DC2626", alpha=0.88, edgecolor="#0F172A", linewidth=0.6, label="Repr. Grounding")
    ax.bar(tx, cov_m, bottom=repr_m, width=0.58, color="#3B82F6", alpha=0.88, edgecolor="#0F172A", linewidth=0.6, label="POMDP Coverage Loops")
    for i, (r, c) in enumerate(zip(repr_m, cov_m)):
        ax.text(i, r + c + 0.9, f"{r+c:.1f}\n({r:.1f}+{c:.1f})", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.set_xticks(tx)
    ax.set_xticklabels(tlabels, rotation=32, ha="right", fontsize=8.5)
    ax.set_ylim(0, max(r + c for r, c in zip(repr_m, cov_m)) * 1.18 + 3)
    ax.set_ylabel("Mean Failures / 300 unseen S6N3 eps")
    ax.set_title("(c) 300-Ep Failure Funnel: Grounding + Blind-Spot Loops", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t_start = time.time()
    n_workers = min(8, os.cpu_count() or 4)
    print(f"[b6_eval] Starting Paradigm 1 full evaluation with {n_workers} workers...", flush=True)

    # 1. Multi-scale evaluation for all new b6 models (+ load existing refs from reports/b2_attn_v2_eval.json)
    prev_ev2 = json.loads((ROOT / "reports" / "b2_attn_v2_eval.json").read_text())
    prev_f020 = json.loads((ROOT / "reports" / "b2_attn_v2_f020_eval.json").read_text())
    prev_probe = json.loads((ROOT / "reports" / "b2_attn_v2_probe.json").read_text())
    prev_fail = json.loads((ROOT / "reports" / "b2_attn_v2_failures.json").read_text())
    prev_feat = json.loads((ROOT / "reports" / "b2_attn_v2_feature_mechanics.json").read_text())

    eval_tasks = []
    for var in ALL_GEOREL_VARIANTS:
        for seed in SEEDS:
            rdir = ROOT / f"runs/b6_georel_{var}_seed{seed}"
            if not (rdir / "best_model.pt").exists():
                continue
            for env_tag, env_id, s_list in [
                ("S6N3", ENV_ID, TEST_SEEDS),
                ("S7N4", ENV_ID_S7, TEST_SEEDS_S7),
                ("S8N6", ENV_ID_S8, TEST_SEEDS_S7),
            ]:
                eval_tasks.append((str(rdir), env_id, s_list, f"full|{var}|{env_tag}"))

    for var in F020_VARIANTS:
        for seed in SEEDS:
            rdir = ROOT / f"runs/b6_georel_{var}_f020_seed{seed}"
            if not (rdir / "best_model.pt").exists():
                continue
            for env_tag, env_id, s_list in [
                ("S6N3", ENV_ID, TEST_SEEDS),
                ("S7N4", ENV_ID_S7, TEST_SEEDS_S7),
            ]:
                eval_tasks.append((str(rdir), env_id, s_list, f"f020|{var}|{env_tag}"))

    grouped_full = collections.defaultdict(lambda: collections.defaultdict(list))
    grouped_f020 = collections.defaultdict(lambda: collections.defaultdict(list))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        for res in ex.map(_eval_worker, eval_tasks):
            mode, var, env_tag = res["_key"].split("|")
            if mode == "full":
                grouped_full[var][env_tag].append(res)
            else:
                grouped_f020[var][env_tag].append(res)
            print(
                f"  [eval] {mode:4s} {var:18s} {env_tag} seed={res['train_seed']} "
                f"succ={res['closed_loop_success']:.4f} val={res['val_action_acc']:.4f} ({res['eval_seconds']}s)",
                flush=True,
            )

    eval_summary = {}
    for ref_k in ("bc30", "film_nwd", "spatial_softmax_attn", "res_spatial_softmax", "attn_map_inject", "mh_spatial_relational"):
        if ref_k in prev_ev2.get("references", {}):
            eval_summary[ref_k] = prev_ev2["references"][ref_k]
        elif ref_k in prev_ev2.get("variants", {}):
            eval_summary[ref_k] = prev_ev2["variants"][ref_k]
    for var, env_map in grouped_full.items():
        eval_summary[var] = {env_tag: summarize_group(runs) for env_tag, runs in env_map.items()}

    f020_summary = {}
    f020_ref_map = {
        "bc30": "bc_step_f020",
        "film_nwd": "film_step_f020",
        "spatial_softmax_attn": "spatial_softmax_attn",
    }
    for ref_k in ("bc30", "film_nwd", "spatial_softmax_attn", "res_spatial_softmax", "attn_map_inject", "mh_spatial_relational"):
        raw = None
        mapped = f020_ref_map.get(ref_k, ref_k)
        if mapped in prev_f020.get("references", {}):
            raw = prev_f020["references"][mapped]
        elif ref_k in prev_f020.get("variants", {}):
            raw = prev_f020["variants"][ref_k]
        if raw is not None:
            norm_env = {}
            for ek in ("S6N3", "S7N4"):
                if ek not in raw:
                    continue
                d = raw[ek]
                if "closed_loop_success" in d:
                    norm_env[ek] = d
                else:
                    norm_env[ek] = {
                        "closed_loop_success": {"mean": d["success_mean"], "sd": d["success_sd"]},
                    }
            f020_summary[ref_k] = norm_env
    for var, env_map in grouped_f020.items():
        f020_summary[var] = {env_tag: summarize_group(runs) for env_tag, runs in env_map.items()}

    # 2. Probe 3: 300-episode failure taxonomy & 6 diagnostic seeds
    print("[b6_eval] Running Probe 3 (300-episode failure taxonomy & case seeds)...", flush=True)
    fail_tasks = []
    for var in ALL_GEOREL_VARIANTS:
        for seed in SEEDS:
            rdir = ROOT / f"runs/b6_georel_{var}_seed{seed}"
            if (rdir / "best_model.pt").exists():
                fail_tasks.append((var, str(rdir)))

    fail_grouped = collections.defaultdict(list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        futs = [(var, ex.submit(_failure_worker, rdir)) for var, rdir in fail_tasks]
        for var, fut in futs:
            fail_grouped[var].append(fut.result())

    fail_summary = {}
    ref_fail_agg = prev_fail.get("references", {}).get("aggregate", {})
    var_fail_agg = prev_fail.get("variants", {}).get("aggregate", prev_fail.get("variants", {}))
    ref_case_map = prev_fail.get("cases", {})
    for ref_k in ("bc30", "film_nwd", "spatial_softmax_attn", "res_spatial_softmax", "attn_map_inject", "mh_spatial_relational"):
        pf = ref_fail_agg.get(ref_k) or var_fail_agg.get(ref_k)
        if pf:
            cases_norm = {}
            for cseed in CASE_SEEDS:
                cs = str(cseed)
                if cs in ref_case_map and ref_k in ref_case_map[cs]:
                    v = ref_case_map[cs][ref_k]
                    if isinstance(v, list):
                        cases_norm[cs] = {"solved": sum(1 for x in v if x.get("success")), "total": len(v)}
            fail_summary[ref_k] = {
                "total_failures": pf.get("n_failure") or pf.get("total_failures"),
                "repr_failures": pf["representation"],
                "cov_failures": pf["state_coverage"],
                "cases": cases_norm,
            }

    for var, runs in fail_grouped.items():
        runs = sorted(runs, key=lambda r: r["train_seed"])
        case_pass = {}
        for cseed in CASE_SEEDS:
            cs = str(cseed)
            solved = [r["cases"][cs]["success"] for r in runs]
            steps = [r["cases"][cs]["steps"] for r in runs]
            case_pass[cs] = {"solved": int(sum(solved)), "total": len(runs), "steps": steps}
        fail_summary[var] = {
            "total_failures": mean_sd([r["total_failures"] for r in runs]),
            "repr_failures": mean_sd([r["repr_failures"] for r in runs]),
            "cov_failures": mean_sd([r["cov_failures"] for r in runs]),
            "cases": case_pass,
            "runs": runs,
        }

    # 3. Probe 1 (Counterfactual Swap) & Probe 2 (Feature Geometry)
    print("[b6_eval] Running Probe 1 (Counterfactual Swap) & Probe 2 (Feature Geometry)...", flush=True)
    cf_states = collect_states(n_states=300, scan=300)
    scenes_s6 = collect_probe_scenes(ENV_ID, TEST_SEEDS, max_scenes=150)
    scenes_s8 = collect_probe_scenes(ENV_ID_S8, TEST_SEEDS_S7, max_scenes=150)

    probe1_out = {}
    probe2_out = {"S6N3": {}, "S8N6": {}}

    for var in ALL_GEOREL_VARIANTS:
        p1_runs, p2_s6_runs, p2_s8_runs = [], [], []
        for seed in SEEDS:
            rdir = ROOT / f"runs/b6_georel_{var}_seed{seed}"
            if not (rdir / "best_model.pt").exists():
                continue
            model, _, _ = load_any_model(rdir)
            p1_runs.append(probe_counterfactual_one(model, cf_states))
            p2_s6_runs.append(probe_feature_geometry_one(model, scenes_s6))
            p2_s8_runs.append(probe_feature_geometry_one(model, scenes_s8))
        if p1_runs:
            probe1_out[var] = {
                "color_swap_rate": mean_sd([r["color_swap_rate"] for r in p1_runs]),
                "type_swap_rate": mean_sd([r["type_swap_rate"] for r in p1_runs]),
            }
            for env_tag, runs in (("S6N3", p2_s6_runs), ("S8N6", p2_s8_runs)):
                probe2_out[env_tag][var] = {
                    k: mean_sd([r[k] for r in runs if r[k] is not None])
                    for k in ("h_mod_T_over_maxD", "h2_T_over_maxD", "h2_T_over_partialD", "spatial_selectivity", "background_drift")
                }

    # Save JSONs
    out_json = {
        "eval": eval_summary,
        "f020": f020_summary,
        "failures": fail_summary,
        "probe1_counterfactual": probe1_out,
        "probe2_feature_geometry": probe2_out,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    (ROOT / "reports" / "b6_georel_results.json").write_text(json.dumps(out_json, indent=2))

    # Build figure
    fig_path = ROOT / "reports" / "figures" / "b6_georel_unified" / "b6_paradigm1_ladder.png"
    make_ladder_figure(eval_summary, f020_summary, fail_summary, fig_path)

    # Build clean Markdown Summary
    lines = [
        "# Branch 6 · Paradigm 1 Unified Composable Architecture (`GeoRel-BCNet` & `GeoRel-GRU`)",
        "",
        "All models trained strictly on frozen `data/expert_5000.pkl` (4,500 train + 500 val, zero extra expert queries),",
        "30 epochs, `Adam lr=1e-3, wd=0`, checkpoint strictly by `val_action_acc`, evaluated on 300 unseen seeds (`mean ± SD`, `ddof=0`, `n = 3`).",
        "",
        "## 1. Paradigm 1 Ablation Ladder (Multi-Scale, 20% Low-Data, Failures & Key Cases)",
        "",
        "| Stage / Model | Params | `S6N3` (ID) | `S7N4` (OOD) | `S8N6` (OOD) | `20%` S6 / S7 | Fails /300 (`Repr+Cov`) | `4501` / `4502` / `4568` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, _, lbl in LADDER_MODELS:
        if k not in eval_summary:
            continue
        ev = eval_summary[k]
        p = ev["S6N3"]["n_params"]
        s6 = ev["S6N3"]["closed_loop_success"]
        s7 = ev["S7N4"]["closed_loop_success"]
        s8 = ev["S8N6"]["closed_loop_success"]
        if k in f020_summary:
            f6 = f020_summary[k]["S6N3"]["closed_loop_success"]["mean"]
            f7 = f020_summary[k]["S7N4"]["closed_loop_success"]["mean"]
            fstr = f"{f6:.3f} / {f7:.3f}"
        else:
            fstr = "—"
        if k in fail_summary:
            fs = fail_summary[k]
            f_tot = fs["total_failures"]["mean"]
            f_r = fs["repr_failures"]["mean"]
            f_c = fs["cov_failures"]["mean"]
            fail_str = f"{f_tot:.1f} (`{f_r:.1f}+{f_c:.1f}`)"
            cs = fs.get("cases", {})
            c4501 = cs.get("4501", {}).get("solved", "—") if isinstance(cs.get("4501"), dict) else cs.get("4501", "—")
            c4502 = cs.get("4502", {}).get("solved", "—") if isinstance(cs.get("4502"), dict) else cs.get("4502", "—")
            c4568 = cs.get("4568", {}).get("solved", "—") if isinstance(cs.get("4568"), dict) else cs.get("4568", "—")
            case_str = f"{c4501}/3 · {c4502}/3 · {c4568}/3"
        else:
            fail_str = "—"
            case_str = "—"
        lines.append(
            f"| {lbl} | {p:,} | {s6['mean']:.3f} ± {s6['sd']:.3f} | {s7['mean']:.3f} ± {s7['sd']:.3f} | "
            f"{s8['mean']:.3f} ± {s8['sd']:.3f} | {fstr} | {fail_str} | {case_str} |"
        )

    lines += [
        "",
        "## 2. Mechanistic Probes (Probe 1 Counterfactual Swap & Probe 2 Feature Geometry on New Models)",
        "",
        "| Model | Color Swap | Type Swap | S6 `h_mod T/maxD` | S8 `h_mod T/maxD` | S8 `h2 Partial SNR` | S8 Selectivity | S8 Bg Drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for var in ALL_GEOREL_VARIANTS:
        if var not in probe1_out:
            continue
        p1 = probe1_out[var]
        p2s6 = probe2_out["S6N3"][var]
        p2s8 = probe2_out["S8N6"][var]
        lines.append(
            f"| `{var}` | {p1['color_swap_rate']['mean']:.3f} | {p1['type_swap_rate']['mean']:.3f} | "
            f"{p2s6['h_mod_T_over_maxD']['mean']:.3f}× | {p2s8['h_mod_T_over_maxD']['mean']:.3f}× | "
            f"{p2s8['h2_T_over_partialD']['mean']:.3f}× | {p2s8['spatial_selectivity']['mean']:.2f}× | "
            f"{p2s8['background_drift']['mean']:.3f} |"
        )

    (ROOT / "reports" / "b6_georel_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[b6_eval] Done in {time.time() - t_start:.1f}s -> reports/b6_georel_summary.md", flush=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
