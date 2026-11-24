"""ABC attention fusion: AB (Map-Inject) and ABC (+ relative-bias self-attention).

Primary comparison: BC30 vs s3_map1h_relbias. s2_map1h is the AB ablation.
A and FiLM controls reuse their frozen implementations in branches/b2_attn and b2_film.
The retained architectures and state-dict keys match paradigm1-georel-unified.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    N_ACTIONS,
    N_DIRS,
    N_OBJECT_IDS,
    PAD_IDX,
    VIEW_SIZE,
    VOCAB,
)

ALL_GEOREL_VARIANTS = ("s2_map1h", "s3_map1h_relbias")
SINGLE_FRAME_VARIANTS = ALL_GEOREL_VARIANTS


class SpatialPointer1H(nn.Module):
    """Components A/B: language-conditioned spatial map and raw-feature concatenation.

    Produces a single normalized 7x7 pointer map M = 49 * softmax_49(<W_q l, W_k h> / sqrt(C)).
    Matches GoToLocal's single-target retrieval structure with minimal degrees of freedom.
    """

    def __init__(self, lang_dim: int, channels: int):
        super().__init__()
        self.channels = channels
        self.q_proj = nn.Linear(lang_dim, channels)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def compute_map(self, h: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1, H, W) competition map 49 * softmax."""
        B, C, H, W = h.shape
        q = self.q_proj(lang).view(B, C, 1, 1)
        k = self.k_proj(h)
        scores = (q * k).sum(dim=1, keepdim=True) / math.sqrt(C)
        alpha = F.softmax(scores.flatten(2), dim=-1).view(B, 1, H, W)
        return float(H * W) * alpha

    def forward(self, h: torch.Tensor, lang: torch.Tensor, return_map: bool = False):
        M = self.compute_map(h, lang)
        out = torch.cat([h, M], dim=1)
        if return_map:
            return out, M, {"mode": "concat_map"}
        return out


class RelBiasRelationalBlock(nn.Module):
    """Component C: 2D relative-bias relational self-attention.

    Unlike learned absolute 49-cell pos_emb or absolute (x, y) projections, attention logits
    add a learned 2D relative offset bias B[head, dy + (H-1), dx + (W-1)] initialized to 0,
    sharing offset biases across grid locations and enabling non-local information exchange.
    This does not establish translation equivariance of the whole finite-view policy. Residual scale starts at 0.0 (identity init) for safe composition.
    """

    def __init__(self, channels: int = 32, num_heads: int = 4, view_size: int = VIEW_SIZE):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.view_size = view_size

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out_proj = nn.Linear(channels, channels)

        # 2D relative position index table for HxW grid -> (H*W, H*W) in [0, (2H-1)*(2W-1)-1]
        ys = torch.arange(view_size)
        xs = torch.arange(view_size)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)  # (49, 2)
        rel = coords[:, None, :] - coords[None, :, :]  # (49, 49, 2) in [-(V-1), V-1]
        rel[:, :, 0] += view_size - 1
        rel[:, :, 1] += view_size - 1
        rel_idx = rel[:, :, 0] * (2 * view_size - 1) + rel[:, :, 1]  # (49, 49)
        self.register_buffer("rel_idx", rel_idx.long(), persistent=False)

        num_rel = (2 * view_size - 1) * (2 * view_size - 1)
        self.rel_bias_table = nn.Parameter(torch.zeros(num_heads, num_rel))
        self.rel_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, return_map: bool = False):
        B, C, H, W = h.shape
        N = H * W
        x = self.norm(h.flatten(2).transpose(1, 2))  # (B, 49, C)
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, K, d)
        q = q.transpose(1, 2)  # (B, K, N, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, K, N, N)
        bias = self.rel_bias_table[:, self.rel_idx.view(-1)].view(1, self.num_heads, N, N)
        attn = F.softmax(scores + bias, dim=-1)
        out_seq = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out_seq = self.out_proj(out_seq)
        delta = self.rel_scale * out_seq.transpose(1, 2).view(B, C, H, W)
        out = h + delta
        if not return_map:
            return out
        return out, delta.norm(dim=1, keepdim=True), {
            "rel_scale": round(float(self.rel_scale.item()), 5),
            "rel_attn_49x49": attn[0].mean(dim=0).detach().cpu().numpy(),
        }


class GeoRelBCNet(nn.Module):
    """Single-frame AB / ABC fusion network; all retained parameters remain trainable."""

    def __init__(
        self,
        variant: str,
        vocab_size: int = len(VOCAB),
        n_actions: int = N_ACTIONS,
        n_dirs: int = N_DIRS,
        obj_emb_dim: int = 8,
        lang_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.0,
        channels: int = 32,
    ):
        super().__init__()
        if variant not in SINGLE_FRAME_VARIANTS:
            raise ValueError(f"variant must be in {SINGLE_FRAME_VARIANTS}, got {variant}")
        self.variant = variant
        self.hidden = hidden
        self.is_recurrent = False

        # BCNet-aligned visual and language stem (same architecture, trainable weights)
        self.lang_emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)
        self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
        self.conv1 = nn.Conv2d(3 * obj_emb_dim, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        self.use_map_inject = True
        self.use_relational = variant == "s3_map1h_relbias"
        self.pointer = SpatialPointer1H(lang_dim, channels)
        self.conv2 = nn.Conv2d(channels + 1, 32, 3, padding=1)

        if self.use_relational:
            self.rel_block = RelBiasRelationalBlock(channels=32, num_heads=4, view_size=VIEW_SIZE)

        v_dim = 32 * VIEW_SIZE * VIEW_SIZE
        trunk_in = v_dim + lang_dim + n_dirs  # 1604
        self.trunk_fc = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_actions)

    def encode_language(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != PAD_IDX).unsqueeze(-1)
        tok_emb = self.lang_emb(tokens)
        counts = mask.sum(dim=1).clamp(min=1)
        return (tok_emb * mask).sum(dim=1) / counts

    def encode_h1(self, image: torch.Tensor) -> torch.Tensor:
        x = image.permute(0, 3, 1, 2)
        feats = [self.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
        return self.act(self.conv1(torch.cat(feats, dim=1)))

    def extract_layers(self, image: torch.Tensor, tokens: torch.Tensor):
        """Returns (h1, h_mod, h2, lang) for unified hidden-layer feature probing."""
        lang = self.encode_language(tokens)
        h1 = self.encode_h1(image)
        h_mod = self.pointer(h1, lang)
        h2 = self.act(self.conv2(h_mod))
        if self.use_relational:
            h2 = self.rel_block(h2)
        return h1, h_mod, h2, lang

    def encode_trunk(self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        _, _, h2, lang = self.extract_layers(image, tokens)
        v = torch.flatten(h2, 1)
        d = F.one_hot(direction.long(), num_classes=N_DIRS).float()
        return self.trunk_fc(torch.cat([v, lang, d], dim=-1))

    def forward(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        return self.head(self.encode_trunk(image, tokens, direction))

    @torch.no_grad()
    def extract_diagnostics(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> dict:
        lang = self.encode_language(tokens)
        h1 = self.encode_h1(image)
        h_mod, smap, extra = self.pointer(h1, lang, return_map=True)
        out: dict = {
            "variant": self.variant,
            "spatial_map": smap[0, 0].detach().cpu().numpy(),
            **extra,
        }
        if self.use_relational:
            h_geo = self.act(self.conv2(h_mod))
            _, rel_norm, rex = self.rel_block(h_geo, return_map=True)
            out["rel_delta_norm"] = rel_norm[0, 0].detach().cpu().numpy()
            out.update(rex)
        return out

    def scalar_stats(self) -> dict:
        if self.use_relational:
            return {"rel_scale": round(float(self.rel_block.rel_scale.item()), 5)}
        return {}


def build(variant: str, **kw) -> nn.Module:
    return GeoRelBCNet(variant=variant, **kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
