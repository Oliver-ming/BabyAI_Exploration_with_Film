"""加载 b2 的 FiLM run 并包装成冻结评估所需的 `obs -> action` 策略函数。

冻结的 `src/eval_closed_loop.evaluate()` 接收无状态单步策略，所以这里只做"模型包装"，
不复制任何评估逻辑 —— 保证与基线、支线的数字口径完全一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))

from common import encode_batch  # noqa: E402
from film_net import build  # noqa: E402


def load_film_run(run_dir: str | Path):
    """返回 (model, cfg, vocab)。与 `common.load_run` 接口兼容，但模型是 FiLM 变体。"""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    vocab = json.loads((run_dir / "vocab.json").read_text()) if (run_dir / "vocab.json").exists() else None
    a = cfg["arch"]
    model = build(
        a["variant"],
        vocab_size=len(vocab["vocab"]) if vocab else 14,
        obj_emb_dim=a["obj_emb_dim"],
        lang_dim=a["lang_dim"],
        hidden=a.get("hidden", 128),
        dropout=a.get("dropout", 0.0),
    )
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg, vocab


def make_policy_fn(model):
    """把模型包成 `obs -> int` 的无状态单步策略（供冻结评估复用）。"""
    model.eval()

    def policy(obs) -> int:
        with torch.no_grad():
            img, toks, d = encode_batch([obs])
            return int(model(img, toks, d).argmax(dim=-1).item())

    return policy


def load_any_run(run_dir: str | Path):
    """自动识别是基线(BCNet)还是 b2(FiLM) 的 run 目录。"""
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    if cfg.get("branch") == "b2_film" or "variant" in cfg.get("arch", {}):
        return load_film_run(run_dir)
    from common import load_run  # 冻结

    return load_run(run_dir)
