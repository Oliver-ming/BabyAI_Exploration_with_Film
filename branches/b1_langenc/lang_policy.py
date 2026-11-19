"""支线 1 · 加载 b1 / b2 / 基线 run，并包装成冻结评估所需的 `obs -> action`。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))

from common import encode_batch  # noqa: E402
from lang_encoders import VARIANTS, build_lang_model  # noqa: E402


def load_lang_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    vocab = json.loads((run_dir / "vocab.json").read_text())
    a = cfg["arch"]
    model = build_lang_model(
        variant=a["variant"],
        vocab_size=len(vocab["vocab"]),
        obj_emb_dim=a["obj_emb_dim"],
        lang_dim=a["lang_dim"],
        hidden=a.get("hidden", 128),
        dropout=a.get("dropout", 0.0),
        widen_to_match=False,  # config 里存的 hidden 已经是最终值
    )
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return model, cfg, vocab


def make_policy_fn(model):
    model.eval()

    def policy(obs) -> int:
        with torch.no_grad():
            img, toks, d = encode_batch([obs])
            return int(model(img, toks, d).argmax(dim=-1).item())

    return policy


def load_any_run(run_dir: str | Path):
    """自动识别基线 / b1 / b2 / b3 的 run 目录。"""
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    branch = cfg.get("branch")
    variant = cfg.get("arch", {}).get("variant")

    if branch == "b1_langenc" or variant in VARIANTS:
        return load_lang_run(run_dir)
    if branch == "b2_film":
        sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
        from film_policy import load_film_run

        return load_film_run(run_dir)

    from common import load_run  # 冻结（baseline 与 b3_bc30 等）

    return load_run(run_dir)
