"""支线 4 · 加载 b4 / b2 / 基线 run，并包装成冻结评估所需的 `obs -> action`。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))

from common import BCNet, encode_batch, load_run  # noqa: E402
from film_net import FiLMBCNet  # noqa: E402


def load_b4_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    vocab = json.loads((run_dir / "vocab.json").read_text())
    a = cfg["arch"]
    if a.get("model_type") == "film" or a.get("use_film"):
        model = FiLMBCNet(
            variant="film",
            vocab_size=len(vocab["vocab"]),
            obj_emb_dim=a["obj_emb_dim"],
            lang_dim=a["lang_dim"],
            hidden=a.get("hidden", 128),
            dropout=a.get("dropout", 0.0),
        )
    else:
        model = BCNet(
            vocab_size=len(vocab["vocab"]),
            obj_emb_dim=a["obj_emb_dim"],
            lang_dim=a["lang_dim"],
            hidden=a.get("hidden", 128),
            dropout=a.get("dropout", 0.0),
            img_mode=a.get("img_mode", "embed"),
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
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    branch = cfg.get("branch")
    if branch == "b4_curriculum":
        return load_b4_run(run_dir)
    if branch == "b2_film":
        from film_policy import load_film_run

        return load_film_run(run_dir)
    return load_run(run_dir)
