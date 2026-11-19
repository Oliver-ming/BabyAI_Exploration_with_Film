"""支线 2 扩展消融 v2 · 隐藏层特征空间信噪比（SNR）、空间选择性与背景漂移机制探针。

目的：从隐藏层特征几何直接量化并解释：
  1. 为什么同样 225,183 参数，Spatial-Softmax Attention 在 S6N3 / S7N4 / S8N6 全面胜过 FiLM；
  2. 为什么通道调制（FiLM / Chaplot GA）随干扰物增多（N=3→4→6）出现累积退化；
  3. 为什么移除 pos_emb（NoPE）能压低 Cross-MHA 在 S8N6 的背景漂移并挽回 +6.2%；
  4. 为什么 Attn-Map-Inject 与 MH-Spatial-Relational 能在保留空间聚焦的同时彻底修复绕障 4501。

运行:
    OMP_NUM_THREADS=1 env_embodied/bin/python branches/b2_attn/probe_feature_mechanics.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import contextlib
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b2_attn"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from common import ENV_ID, ENV_ID_S7, TEST_SEEDS, TEST_SEEDS_S7, encode_batch  # noqa: E402
from eval_v2_and_analyze import ENV_ID_S8, SEEDS, load_run_universal  # noqa: E402
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT  # noqa: E402

MODELS = [
    ("bc30", "runs/b3_bc30"),
    ("film_nwd", "runs/b2_film_nwd"),
    ("gated_attn_chaplot", "runs/b2_attn_v2_gated_attn_chaplot"),
    ("spatial_softmax_attn", "runs/b2_attn_spatial_softmax_attn"),
    ("res_spatial_softmax", "runs/b2_attn_v2_res_spatial_softmax"),
    ("cross_attn_mha", "runs/b2_attn_cross_attn_mha"),
    ("cross_mha_nopos", "runs/b2_attn_v2_cross_mha_nopos"),
    ("attn_map_inject", "runs/b2_attn_v2_attn_map_inject"),
    ("token_coattn_spatial", "runs/b2_attn_v2_token_coattn_spatial"),
    ("mh_spatial_relational", "runs/b2_attn_v2_mh_spatial_relational"),
]

LABELS = {
    "bc30": "BC30\n(Concat)",
    "film_nwd": "FiLM\n(Channel)",
    "gated_attn_chaplot": "Chaplot\nGA",
    "spatial_softmax_attn": "v1:Spatial\nSoftmax",
    "res_spatial_softmax": "v2:Res\nSoftmax",
    "cross_attn_mha": "v1:Cross\nMHA",
    "cross_mha_nopos": "v2:MHA\nNoPE",
    "attn_map_inject": "v2:Map\nInject",
    "token_coattn_spatial": "v2:Token\nCoAttn",
    "mh_spatial_relational": "v2:MH-Res\n+Rel",
}


def mean_sd(vals: list[float]) -> dict:
    vals = [float(v) for v in vals]
    return {
        "n": len(vals),
        "mean": round(st.mean(vals), 6),
        "sd": round(st.pstdev(vals), 6),
        "values": [round(v, 6) for v in vals],
    }


def extract_h_layers(model, img: torch.Tensor, tok: torch.Tensor, direc: torch.Tensor):
    with torch.no_grad():
        cname = model.__class__.__name__
        if cname == "BCNet":
            x = img.permute(0, 3, 1, 2)
            feats = [model.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
            h1 = model.vision[1](model.vision[0](torch.cat(feats, dim=1)))
            h_mod = h1
            h2 = model.vision[3](model.vision[2](h_mod))
            return h1, h_mod, h2
        if cname == "FiLMBCNet":
            lang = model.encode_language(tok)
            x = img.permute(0, 3, 1, 2)
            feats = [model.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
            h1 = model.act(model.conv1(torch.cat(feats, dim=1)))
            dg, db = model.film.forward(lang)
            h_mod = (1.0 + dg).unsqueeze(-1).unsqueeze(-1) * h1 + db.unsqueeze(-1).unsqueeze(-1)
            h2 = model.act(model.conv2(h_mod))
            return h1, h_mod, h2
        if cname == "CrossModalBCNet":
            lang, tok_emb = model.encode_language(tok)
            x = img.permute(0, 3, 1, 2)
            feats = [model.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
            h1 = model.act(model.conv1(torch.cat(feats, dim=1)))
            h_mod = model._modulate(h1, lang, tok_emb, tok)
            h2 = model.act(model.conv2(h_mod))
            return h1, h_mod, h2
        if cname == "CrossModalAttnV2BCNet":
            lang, tok_emb = model.encode_language(tok)
            h1 = model._encode_h1(img)
            h_mod = model._modulate(h1, lang, tok_emb, tok)
            h2 = model.act(model.conv2(h_mod))
            return h1, h_mod, h2
        raise ValueError(f"Unsupported model class: {cname}")


def collect_visible_target_obs(env_id: str, seeds: list[int], limit: int = 150) -> list[dict]:
    env = make_env(env_id)
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        for sd in seeds:
            obs, _ = env.reset(seed=sd)
            for _ in range(4):
                grid = obs["image"]
                parts = obs["mission"].split()
                tcol, ttype = parts[-2], parts[-1]
                t_pos, d_pos, partial_pos, empty_pos = [], [], [], []
                for r in range(7):
                    for c in range(7):
                        oname = IDX_TO_OBJECT.get(int(grid[r, c, 0]), "")
                        cname = IDX_TO_COLOR.get(int(grid[r, c, 1]), "")
                        if oname in ("key", "ball", "box"):
                            if oname == ttype and cname == tcol:
                                t_pos.append((r, c))
                            else:
                                d_pos.append((r, c))
                                if oname == ttype or cname == tcol:
                                    partial_pos.append((r, c))
                        elif oname == "empty":
                            empty_pos.append((r, c))
                if t_pos and d_pos and empty_pos:
                    alt_col = "red" if tcol != "red" else "blue"
                    alt_obs = {**obs, "mission": f"go to the {alt_col} {ttype}"}
                    out.append(
                        {
                            "obs": obs,
                            "alt_obs": alt_obs,
                            "target": t_pos[0],
                            "distractors": d_pos,
                            "partial_distractors": partial_pos,
                            "empty": empty_pos,
                        }
                    )
                    break
                obs, _, term, trunc, _ = env.step(1)
                if term or trunc:
                    break
            if len(out) >= limit:
                break
    env.close()
    return out


def eval_env_mechanics(dataset: list[dict]) -> dict:
    imgs, toks, dirs = encode_batch([x["obs"] for x in dataset])
    _, alt_toks, _ = encode_batch([x["alt_obs"] for x in dataset])
    env_res = {}
    for name, prefix in MODELS:
        seed_metrics = {
            "h_mod_T_over_maxD": [],
            "h2_T_over_maxD": [],
            "h2_T_over_partialD": [],
            "spatial_selectivity_T_over_bg": [],
            "bg_drift_ratio": [],
        }
        for s in SEEDS:
            model, _, _ = load_run_universal(f"{prefix}_seed{s}")
            _, h_mod, h2 = extract_h_layers(model, imgs, toks, dirs)
            _, _, h2_alt = extract_h_layers(model, imgs, alt_toks, dirs)
            e_mod = h_mod.norm(dim=1).numpy()
            e_h2 = h2.norm(dim=1).numpy()
            dh2 = (h2 - h2_alt).norm(dim=1).numpy()

            s_m_max, s_2_max, s_2_part, s_sel, s_bg = [], [], [], [], []
            for i, row in enumerate(dataset):
                tp = row["target"]
                dps = row["distractors"]
                pps = row["partial_distractors"]
                eps = row["empty"]
                tm = e_mod[i, tp[0], tp[1]]
                s_m_max.append(tm / (max(e_mod[i, r, c] for r, c in dps) + 1e-8))
                t2 = e_h2[i, tp[0], tp[1]]
                s_2_max.append(t2 / (max(e_h2[i, r, c] for r, c in dps) + 1e-8))
                if pps:
                    s_2_part.append(t2 / (max(e_h2[i, r, c] for r, c in pps) + 1e-8))
                dt = dh2[i, tp[0], tp[1]]
                dbg = float(np.mean([dh2[i, r, c] for r, c in eps]))
                ebg = float(np.mean([e_h2[i, r, c] for r, c in eps]))
                if dbg > 1e-6:
                    s_sel.append(dt / dbg)
                s_bg.append(dbg / (ebg + 1e-8))

            seed_metrics["h_mod_T_over_maxD"].append(float(np.mean(s_m_max)))
            seed_metrics["h2_T_over_maxD"].append(float(np.mean(s_2_max)))
            seed_metrics["h2_T_over_partialD"].append(float(np.mean(s_2_part)) if s_2_part else 0.0)
            seed_metrics["spatial_selectivity_T_over_bg"].append(float(np.mean(s_sel)) if s_sel else 0.0)
            seed_metrics["bg_drift_ratio"].append(float(np.mean(s_bg)))

        env_res[name] = {k: mean_sd(v) for k, v in seed_metrics.items()}
        print(
            f"  [{name:22s}] h_mod T/maxD={env_res[name]['h_mod_T_over_maxD']['mean']:.3f} | "
            f"h2 T/partD={env_res[name]['h2_T_over_partialD']['mean']:.3f} | "
            f"sel={env_res[name]['spatial_selectivity_T_over_bg']['mean']:.2f}x | "
            f"bg_drift={env_res[name]['bg_drift_ratio']['mean']:.3f}"
        )
    return env_res


def make_figure(out: dict, ev: dict) -> None:
    plt.rcParams.update({"font.size": 9.2, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), dpi=300)

    order = [m[0] for m in MODELS]
    colors = [
        "#D97706",  # bc30
        "#059669",  # film_nwd
        "#94A3B8",  # chaplot
        "#2563EB",  # v1 spatial_softmax
        "#DC2626",  # v2 res_spatial_softmax
        "#818CF8",  # v1 cross_mha
        "#6D28D9",  # v2 nopos
        "#EA580C",  # v2 map_inject
        "#0D9488",  # v2 coattn
        "#0F766E",  # v2 relational
    ]
    compact = ["BC30", "FiLM", "Chaplot", "v1:Softmax", "v2:ResSS", "v1:MHA", "v2:NoPE", "v2:MapInj", "v2:CoAttn", "v2:Rel"]

    # Panel (a): Pre-conv2 h_mod T/maxD on S6N3 & S8N6
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.36
    m6 = [out["by_env"]["S6N3"][k]["h_mod_T_over_maxD"]["mean"] for k in order]
    s6 = [out["by_env"]["S6N3"][k]["h_mod_T_over_maxD"]["sd"] for k in order]
    m8 = [out["by_env"]["S8N6"][k]["h_mod_T_over_maxD"]["mean"] for k in order]
    s8 = [out["by_env"]["S8N6"][k]["h_mod_T_over_maxD"]["sd"] for k in order]
    b6 = ax.bar(x - w / 2, m6, w, yerr=s6, capsize=2, color=colors, alpha=0.92, edgecolor="#1F2937", linewidth=0.65, label="S6N3 (N=3)")
    ax.bar(x + w / 2, m8, w, yerr=s8, capsize=2, color=colors, alpha=0.48, hatch="///", edgecolor="#1F2937", linewidth=0.65, label="S8N6 (N=6)")
    ax.axhline(1.0, color="#DC2626", linestyle="--", linewidth=1.0, label="No separation (1.0×)")
    for b, m, sd in zip(b6, m6, s6):
        if m > 1.5 or m < 1.05:
            ax.text(b.get_x() + b.get_width() / 2, m + sd + 0.18, f"{m:.2f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(compact, fontsize=7.6, rotation=32, ha="right")
    ax.set_ylim(0, 13.8)
    ax.set_ylabel("Pre-Conv2 Target / Max-Distractor Norm Ratio")
    ax.set_title("(a) Fusion-Layer Competition (h_mod T / max D)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")

    # Panel (b): S8N6 Post-Conv2 Partial-Match SNR & Spatial Selectivity
    ax = axes[1]
    p8 = [out["by_env"]["S8N6"][k]["h2_T_over_partialD"]["mean"] for k in order]
    p8s = [out["by_env"]["S8N6"][k]["h2_T_over_partialD"]["sd"] for k in order]
    sel8 = [out["by_env"]["S8N6"][k]["spatial_selectivity_T_over_bg"]["mean"] for k in order]
    sel8s = [out["by_env"]["S8N6"][k]["spatial_selectivity_T_over_bg"]["sd"] for k in order]
    ax.bar(x - w / 2, p8, w, yerr=p8s, capsize=2, color=colors, alpha=0.90, edgecolor="#1F2937", linewidth=0.65, label="h2 T / partial-match D")
    ax.bar(x + w / 2, sel8, w, yerr=sel8s, capsize=2, color="#1D4ED8", alpha=0.38, hatch="...", edgecolor="#1F2937", linewidth=0.65, label="Spatial selectivity Δh2(T)/Δh2(bg)")
    ax.axhline(1.0, color="#6B7280", linestyle=":", linewidth=0.9)
    for xi, (pv, sv) in enumerate(zip(p8, sel8)):
        if xi in (1, 3, 7, 9):
            ax.text(xi - w / 2, pv + p8s[xi] + 0.08, f"{pv:.2f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
            ax.text(xi + w / 2, sv + sel8s[xi] + 0.08, f"{sv:.2f}×", ha="center", va="bottom", fontsize=6.5, color="#1E40AF", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(compact, fontsize=7.6, rotation=32, ha="right")
    ax.set_ylim(0, 5.15)
    ax.set_ylabel("Ratio on Unseen S8N6 (8×8, N=6)")
    ax.set_title("(b) S8N6 Partial-Match SNR & Spatial Selectivity", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")

    # Panel (c): S8N6 Spatial Selectivity vs Closed-Loop Success (excluding bc30=0)
    ax = axes[2]
    def get_s8_succ(k: str) -> float:
        if k in ("bc30", "film_nwd"):
            return ev["references"][k]["S8N6"]["closed_loop_success"]["mean"]
        if f"v1_{k}" in ev["references"]:
            return ev["references"][f"v1_{k}"]["S8N6"]["closed_loop_success"]["mean"]
        return ev["variants"][k]["S8N6"]["closed_loop_success"]["mean"]

    offsets = {
        "film_nwd": (6, -12),
        "gated_attn_chaplot": (6, 4),
        "spatial_softmax_attn": (-58, -14),
        "res_spatial_softmax": (-52, 6),
        "cross_attn_mha": (6, -10),
        "cross_mha_nopos": (6, 4),
        "attn_map_inject": (-52, -13),
        "token_coattn_spatial": (-50, 7),
        "mh_spatial_relational": (-52, 6),
    }
    for idx, k in enumerate(order):
        if k == "bc30":
            continue
        sx = out["by_env"]["S8N6"][k]["spatial_selectivity_T_over_bg"]["mean"]
        sy = get_s8_succ(k)
        bd = out["by_env"]["S8N6"][k]["bg_drift_ratio"]["mean"]
        ax.scatter([sx], [sy], s=95, color=colors[idx], edgecolors="#111827", linewidths=0.9, zorder=4)
        dx, dy = offsets.get(k, (5, 4))
        ax.annotate(f"{compact[idx]}\n(drift {bd:.2f})", (sx, sy), xytext=(dx, dy), textcoords="offset points", fontsize=7.0, fontweight="bold" if k in ("film_nwd", "spatial_softmax_attn", "mh_spatial_relational") else "normal")
    # Arrow showing pos_emb -> NoPE improvement
    x_mha = out["by_env"]["S8N6"]["cross_attn_mha"]["spatial_selectivity_T_over_bg"]["mean"]
    y_mha = get_s8_succ("cross_attn_mha")
    x_np = out["by_env"]["S8N6"]["cross_mha_nopos"]["spatial_selectivity_T_over_bg"]["mean"]
    y_np = get_s8_succ("cross_mha_nopos")
    ax.annotate("", xy=(x_np, y_np), xytext=(x_mha, y_mha), arrowprops=dict(arrowstyle="->", color="#6D28D9", lw=1.6, ls="--"))
    ax.set_xlim(1.65, 3.85)
    ax.set_ylim(0.71, 0.925)
    ax.set_xlabel("Spatial Selectivity Δh2(T) / Δh2(bg) on S8N6")
    ax.set_ylabel("Zero-Shot S8N6 Closed-Loop Success")
    ax.set_title("(c) Why Spatial Selectivity Drives S8N6 Transfer", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/b2_attn/b2_attn_v2_feature_mechanics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    by_env = {}
    for tag, env_id, seeds in [
        ("S6N3", ENV_ID, TEST_SEEDS),
        ("S7N4", ENV_ID_S7, TEST_SEEDS_S7),
        ("S8N6", ENV_ID_S8, TEST_SEEDS_S7),
    ]:
        print(f"== Collecting & probing {tag} ({env_id}) ==")
        ds = collect_visible_target_obs(env_id, seeds, limit=150)
        by_env[tag] = eval_env_mechanics(ds)

    out = {
        "n_scenes_per_env": 150,
        "seeds_evaluated": SEEDS,
        "sd_convention": "ddof=0 (population SD over n=3 init seeds)",
        "by_env": by_env,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (ROOT / "reports/b2_attn_v2_feature_mechanics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    ev = json.loads((ROOT / "reports/b2_attn_v2_eval.json").read_text())
    make_figure(out, ev)
    print(f"FEATURE MECHANICS PROBE & FIGURE COMPLETE IN {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
