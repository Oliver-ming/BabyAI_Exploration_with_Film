"""支线 2 扩展消融 · FiLM vs Attention 与 FiLM+Attention 融合架构。

严格复用冻结基线 `BCNet` 主干（嵌入 + conv1 + conv2 + flatten + trunk）与 `FiLMGenerator`，
仅在 `ReLU(conv1)` 与 `conv2` 之间插入不同的跨模态调制模块：

  1. spatial_attn_gate    : 纯空间门控注意力（与 FiLM 对称，+2,113 参数，零初始化恒等起步）
  2. spatial_softmax_attn : 经典 Softmax 空间注意力（+2,112 参数，49*softmax 保持尺度）
  3. cross_attn_mha       : 4-head 跨模态多头注意力（49 个视觉格 Query × 5 个指令 Token K/V，零初始化残差）
  4. film_spatial_attn    : 串联融合 FiLM（通道 What）→ Spatial Gate（空间 Where），语言条件化 CBAM（+4,225 参数）
  5. film_cross_mha       : 串联融合 FiLM（通道）→ 4-head Cross-Attention（空间-词元细粒度对齐）
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
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))

from common import (  # noqa: E402
    N_ACTIONS,
    N_DIRS,
    N_OBJECT_IDS,
    PAD_IDX,
    VIEW_SIZE,
    VOCAB,
)
from film_net import FiLMGenerator  # noqa: E402

ATTN_VARIANTS = (
    "spatial_attn_gate",
    "spatial_softmax_attn",
    "cross_attn_mha",
    "film_spatial_attn",
    "film_cross_mha",
)


class SpatialAttentionBlock(nn.Module):
    """语言驱动的 7×7 空间注意力（gate 或 softmax 模式）。"""

    def __init__(self, lang_dim: int, channels: int, mode: str = "gate"):
        super().__init__()
        if mode not in ("gate", "softmax"):
            raise ValueError(mode)
        self.mode = mode
        self.channels = channels
        self.q_proj = nn.Linear(lang_dim, channels)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        if mode == "gate":
            self.gate_scale = nn.Parameter(torch.zeros(1))

    def compute_scores(self, h: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(lang).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        k = self.k_proj(h)  # (B, C, 7, 7)
        return (q * k).sum(dim=1, keepdim=True) / math.sqrt(self.channels)  # (B, 1, 7, 7)

    def spatial_map(self, h: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        scores = self.compute_scores(h, lang)
        if self.mode == "gate":
            return 1.0 + self.gate_scale * torch.tanh(scores)  # (B, 1, 7, 7)
        B, _, H, W = scores.shape
        alpha = F.softmax(scores.flatten(2), dim=-1).view(B, 1, H, W)
        return float(H * W) * alpha

    def forward(self, h: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        return self.spatial_map(h, lang) * h


class CrossAttnMHABlock(nn.Module):
    """4-head 跨模态注意力：49 个 7×7 网格 Query × 指令 Token K/V，零初始化残差起步。"""

    def __init__(self, channels: int = 32, num_heads: int = 4, n_cells: int = VIEW_SIZE * VIEW_SIZE):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty(1, n_cells, channels))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.q_norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        h: torch.Tensor,
        tok_emb: torch.Tensor,
        tokens: torch.Tensor,
        return_map: bool = False,
    ):
        B, C, H, W = h.shape
        q_in = self.q_norm(h.flatten(2).transpose(1, 2) + self.pos_emb)  # (B, 49, C)
        key_padding_mask = tokens == PAD_IDX  # (B, L)
        attn_out, attn_w = self.mha(
            q_in,
            tok_emb,
            tok_emb,
            key_padding_mask=key_padding_mask,
            need_weights=return_map,
            average_attn_weights=True,
        )
        delta = self.res_scale * attn_out.transpose(1, 2).view(B, C, H, W)
        out = h + delta
        if not return_map:
            return out
        # 空间图：每个网格的跨模态残差能量 |res_scale * attn_out|_2 归一化 + token 权重
        delta_norm = delta.norm(dim=1, keepdim=True)  # (B, 1, H, W)
        return out, delta_norm, attn_w


class CrossModalBCNet(nn.Module):
    """与 BCNet/FiLMBCNet 完全对齐的主干，在 conv1 后比较 FiLM vs Attention vs 融合。"""

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
        if variant not in ATTN_VARIANTS:
            raise ValueError(f"variant 必须是 {ATTN_VARIANTS}, got {variant}")
        self.variant = variant
        self.hidden = hidden
        self.use_film = variant in ("film_spatial_attn", "film_cross_mha")
        self.use_spatial = variant in ("spatial_attn_gate", "spatial_softmax_attn", "film_spatial_attn")
        self.use_mha = variant in ("cross_attn_mha", "film_cross_mha")

        # 语言嵌入（与 BCNet/FiLMBCNet 一致）
        self.lang_emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)

        # 视觉前端（与 BCNet/FiLMBCNet 一致）
        self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
        in_ch = 3 * obj_emb_dim
        self.conv1 = nn.Conv2d(in_ch, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, 32, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        if self.use_film:
            self.film = FiLMGenerator(lang_dim, channels)
        if variant in ("spatial_attn_gate", "film_spatial_attn"):
            self.spatial_attn = SpatialAttentionBlock(lang_dim, channels, mode="gate")
        elif variant == "spatial_softmax_attn":
            self.spatial_attn = SpatialAttentionBlock(lang_dim, channels, mode="softmax")
        if self.use_mha:
            self.cross_attn = CrossAttnMHABlock(channels=channels, num_heads=4)

        v_dim = 32 * VIEW_SIZE * VIEW_SIZE
        trunk_in = v_dim + lang_dim + n_dirs  # 与 BCNet / FiLMBCNet 的 film 变体完全一致 (1604)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def encode_language(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (tokens != PAD_IDX).unsqueeze(-1)
        tok_emb = self.lang_emb(tokens)  # (B, L, D)
        counts = mask.sum(dim=1).clamp(min=1)
        lang = (tok_emb * mask).sum(dim=1) / counts
        return lang, tok_emb

    def _modulate(
        self, h1: torch.Tensor, lang: torch.Tensor, tok_emb: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        h = h1
        if self.use_film:
            dg, db = self.film.forward(lang)
            h = (1.0 + dg).unsqueeze(-1).unsqueeze(-1) * h + db.unsqueeze(-1).unsqueeze(-1)
        if self.use_spatial:
            h = self.spatial_attn(h, lang)
        if self.use_mha:
            h = self.cross_attn(h, tok_emb, tokens, return_map=False)
        return h

    def forward(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        lang, tok_emb = self.encode_language(tokens)
        x = image.permute(0, 3, 1, 2)
        feats = [self.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
        h1 = self.act(self.conv1(torch.cat(feats, dim=1)))
        h_mod = self._modulate(h1, lang, tok_emb, tokens)
        h2 = self.act(self.conv2(h_mod))
        v = torch.flatten(h2, 1)
        d = F.one_hot(direction.long(), num_classes=N_DIRS).float()
        return self.trunk(torch.cat([v, lang, d], dim=-1))

    @torch.no_grad()
    def extract_spatial_map(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> dict:
        """返回 7×7 空间注意力/门控图，用于可视化与归因分析。"""
        lang, tok_emb = self.encode_language(tokens)
        x = image.permute(0, 3, 1, 2)
        feats = [self.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
        h = self.act(self.conv1(torch.cat(feats, dim=1)))
        out: dict = {"variant": self.variant}
        if self.use_film:
            dg, db = self.film.forward(lang)
            h = (1.0 + dg).unsqueeze(-1).unsqueeze(-1) * h + db.unsqueeze(-1).unsqueeze(-1)
            out["film_abs_dg"] = float(dg.abs().mean().item())
            out["film_abs_db"] = float(db.abs().mean().item())
        if self.use_spatial:
            smap = self.spatial_attn.spatial_map(h, lang)  # (B, 1, 7, 7)
            out["spatial_map"] = smap[0, 0].cpu().numpy()
            if self.spatial_attn.mode == "gate":
                out["gate_scale"] = float(self.spatial_attn.gate_scale.item())
        if self.use_mha:
            _, delta_norm, attn_w = self.cross_attn(h, tok_emb, tokens, return_map=True)
            # attn_w: (B, 49, L); 取非 pad token 的注意力分布或残差能量图
            m = delta_norm[0, 0].cpu().numpy()
            out["spatial_map"] = m
            out["res_scale"] = float(self.cross_attn.res_scale.item())
            out["token_attn_49xL"] = attn_w[0].cpu().numpy()
        return out


def build(variant: str, **kw) -> CrossModalBCNet:
    return CrossModalBCNet(variant=variant, **kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
