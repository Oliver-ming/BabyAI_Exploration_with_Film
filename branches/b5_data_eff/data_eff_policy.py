"""支线 5 · 模型加载与 D4 8× 测试时群等变推理（TTA）封装。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))

from common import BCNet, encode_batch, load_run  # noqa: E402
from film_net import FiLMBCNet  # noqa: E402
from lang_encoders import LangBCNet  # noqa: E402


def load_b5_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    vocab = json.loads((run_dir / "vocab.json").read_text())
    a = cfg["arch"]
    atype = a.get("arch_type", "bcnet")
    if atype == "film" or a.get("use_film"):
        model = FiLMBCNet(
            variant="film",
            vocab_size=len(vocab["vocab"]),
            obj_emb_dim=a["obj_emb_dim"],
            lang_dim=a["lang_dim"],
            hidden=a.get("hidden", 128),
            dropout=a.get("dropout", 0.0),
        )
    elif atype == "gru_small":
        model = LangBCNet(
            variant="gru_small",
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
            img_mode="embed",
        )
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return model, cfg, vocab


def make_policy_fn(model, d4_tta: bool = False):
    model.eval()

    if not d4_tta:
        def policy(obs) -> int:
            with torch.no_grad():
                img, toks, d = encode_batch([obs])
                return int(model(img, toks, d).argmax(dim=-1).item())
        return policy

    def d4_policy(obs) -> int:
        with torch.no_grad():
            img, toks, d = encode_batch([obs])  # (1,7,7,3), (1,5), (1,)
            img_flip = torch.flip(img, dims=[1])
            imgs = torch.cat([img.repeat(4, 1, 1, 1), img_flip.repeat(4, 1, 1, 1)], dim=0)
            toks_b = toks.repeat(8, 1)
            k = torch.arange(4, dtype=d.dtype, device=d.device)
            dirs_c4 = (d.repeat(4) + k) % 4
            dirs_b = torch.cat([dirs_c4, dirs_c4], dim=0)

            probs = F.softmax(model(imgs, toks_b, dirs_b), dim=-1)  # (8, 7)
            # 对镜像分支（后 4 行）把动作概率列 0(left) 与 1(right) 换回原坐标系
            probs_unflipped = probs.clone()
            probs_unflipped[4:, 0] = probs[4:, 1]
            probs_unflipped[4:, 1] = probs[4:, 0]
            mean_prob = probs_unflipped.mean(dim=0)
            return int(mean_prob.argmax(dim=-1).item())

    return d4_policy


def load_any_run(run_dir: str | Path):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    branch = cfg.get("branch")
    if branch == "b5_data_eff":
        return load_b5_run(run_dir)
    if branch == "b2_film":
        from film_policy import load_film_run

        return load_film_run(run_dir)
    return load_run(run_dir)
