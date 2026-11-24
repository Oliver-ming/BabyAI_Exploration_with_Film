"""Branch 6 · Paradigm 1 Unified Composable Architecture (`GeoRel-BCNet` & `GeoRel-GRU`).

Strict monotonic composable ladder sharing the exact frozen `BCNet` stem
(`obj_emb`, `conv1`, `conv2`, `lang_emb`, `trunk`), strictly respecting AIAA 4220 Track 2 rules
(<= 5,000 expert episodes, <= 30 epochs, val_action_acc checkpoint selection, zero test leak):

Single-frame composable ladder (100% compatible with frozen `src/eval_closed_loop.py`):
  - s2_map1h          : [+A+B]   1-Head Spatial-Softmax (NoPE) + Uncorrupted Map-Inject [h1; M] -> Conv2(33->32)
  - s3_map1h_relbias  : [+A+B+C] s2_map1h + Zero-Init 2D Relative-Bias Relational Self-Attention (49x49)
  - s3_res1h_relbias  : [+A+B'+C] 1-Head Residual Spatial-Softmax + Zero-Init 2D Relative-Bias Relational Self-Attn

Stateful episode-level ladder (same 5,000 expert episodes trained as sequences, zero extra oracle data):
  - bc30_gru          : [0+D]    Frozen BCNet + Residual Episode GRU(128) orthogonal control
  - s4_georel_gru     : [+A+B+C+D] s3_map1h_relbias + Residual Episode GRU(128) (full unified model)
  - s4_res1h_gru      : [+A+B'+C+D] s3_res1h_relbias + Residual Episode GRU(128)
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

SINGLE_FRAME_VARIANTS = (
    "s2_map1h",
    "s3_map1h_relbias",
    "s3_res1h_relbias",
)

RECURRENT_VARIANTS = (
    "bc30_gru",
    "s4_georel_gru",
    "s4_res1h_gru",
)

ALL_GEOREL_VARIANTS = SINGLE_FRAME_VARIANTS + RECURRENT_VARIANTS

_SOFTPLUS_ONE_INIT = math.log(math.e - 1.0)


class SpatialPointer1H(nn.Module):
    """Component A: 1-Head Competitive Spatial-Softmax (NoPE, translation-equivariant).

    Produces a single normalized 7x7 pointer map M = 49 * softmax_49(<W_q l, W_k h> / sqrt(C)).
    Matches GoToLocal's single-target retrieval structure with minimal degrees of freedom.
    """

    def __init__(self, lang_dim: int, channels: int, mode: str = "concat_map"):
        super().__init__()
        if mode not in ("concat_map", "residual"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.channels = channels
        self.mode = mode
        self.q_proj = nn.Linear(lang_dim, channels)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        if mode == "residual":
            self.raw_lambda = nn.Parameter(torch.full((1,), _SOFTPLUS_ONE_INIT))

    def compute_map(self, h: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1, H, W) competition map 49 * softmax."""
        B, C, H, W = h.shape
        q = self.q_proj(lang).view(B, C, 1, 1)
        k = self.k_proj(h)
        scores = (q * k).sum(dim=1, keepdim=True) / math.sqrt(C)
        alpha = F.softmax(scores.flatten(2), dim=-1).view(B, 1, H, W)
        return float(H * W) * alpha

    def forward(self, h: torch.Tensor, lang: torch.Tensor, return_map: bool = False):
        M = self.compute_map(h, lang)  # (B, 1, H, W)
        if self.mode == "concat_map":
            out = torch.cat([h, M], dim=1)  # (B, C+1, H, W), h strictly 1.0x uncorrupted
            if not return_map:
                return out
            return out, M, {"mode": "concat_map"}

        lam = F.softplus(self.raw_lambda).view(1, 1, 1, 1)
        eff_mult = 1.0 + lam * M
        out = eff_mult * h
        if not return_map:
            return out
        return out, eff_mult, {"mode": "residual", "lambda": [round(float(lam.item()), 5)]}


class RelBiasRelationalBlock(nn.Module):
    """Component C: Strictly Translation-Equivariant 2D Relative-Bias Relational Self-Attention.

    Unlike learned absolute 49-cell pos_emb or absolute (x, y) projections, attention logits
    add a learned 2D relative offset bias B[head, dy + (H-1), dx + (W-1)] initialized to 0,
    preserving full 2D translation equivariance while enabling non-local target <-> obstacle
    bearing exchange. Residual scale starts at 0.0 (identity init) for safe composition.
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
    """Unified Single-Frame Paradigm 1 Composable Network (`s2_map1h`, `s3_map1h_relbias`, `s3_res1h_relbias`)."""

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

        # Frozen BCNet-aligned visual & language stem
        self.lang_emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)
        self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
        self.conv1 = nn.Conv2d(3 * obj_emb_dim, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        self.use_map_inject = variant in ("s2_map1h", "s3_map1h_relbias")
        self.use_relational = variant in ("s3_map1h_relbias", "s3_res1h_relbias")

        ptr_mode = "concat_map" if self.use_map_inject else "residual"
        self.pointer = SpatialPointer1H(lang_dim, channels, mode=ptr_mode)

        conv2_in = channels + 1 if self.use_map_inject else channels
        self.conv2 = nn.Conv2d(conv2_in, 32, 3, padding=1)

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
        if self.use_map_inject:
            # h_mod is [h1; M] (33x7x7), Conv2 fuses geometry + pointer, then optional RelBias
            h_mod = self.pointer(h1, lang, return_map=False)
            h2 = self.act(self.conv2(h_mod))
            if self.use_relational:
                h2 = self.rel_block(h2, return_map=False)
        else:
            # Residual 1H spatial softmax -> optional RelBias -> Conv2
            h_mod = self.pointer(h1, lang, return_map=False)
            if self.use_relational:
                h_mod = self.rel_block(h_mod, return_map=False)
            h2 = self.act(self.conv2(h_mod))
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
            if self.use_map_inject:
                h_geo = self.act(self.conv2(h_mod))
                _, rel_norm, rex = self.rel_block(h_geo, return_map=True)
            else:
                _, rel_norm, rex = self.rel_block(h_mod, return_map=True)
            out["rel_delta_norm"] = rel_norm[0, 0].detach().cpu().numpy()
            out.update(rex)
        return out

    def scalar_stats(self) -> dict:
        stats = {}
        if hasattr(self.pointer, "raw_lambda"):
            stats["lambda"] = [round(float(F.softplus(self.pointer.raw_lambda).item()), 5)]
        if self.use_relational:
            stats["rel_scale"] = round(float(self.rel_block.rel_scale.item()), 5)
        return stats


class GeoRelGRUBCNet(nn.Module):
    """Component D: Episode-Level Stateful GRU Policy (`bc30_gru`, `s4_georel_gru`, `s4_res1h_gru`).

    Uses the EXACT same 4,500 train + 500 val episodes (zero extra expert data), adding a
    residual GRU(hidden, hidden) over the trunk representation so the policy remembers prior
    orientations/views within an episode and breaks POMDP blind-spot oscillation loops.
    """

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
        if variant not in RECURRENT_VARIANTS:
            raise ValueError(f"variant must be in {RECURRENT_VARIANTS}, got {variant}")
        self.variant = variant
        self.hidden = hidden
        self.is_recurrent = True

        self.lang_emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)
        self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
        self.conv1 = nn.Conv2d(3 * obj_emb_dim, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        self.use_pointer = variant in ("s4_georel_gru", "s4_res1h_gru")
        self.use_map_inject = variant == "s4_georel_gru"
        self.use_relational = variant in ("s4_georel_gru", "s4_res1h_gru")

        if self.use_pointer:
            ptr_mode = "concat_map" if self.use_map_inject else "residual"
            self.pointer = SpatialPointer1H(lang_dim, channels, mode=ptr_mode)

        conv2_in = channels + 1 if self.use_map_inject else channels
        self.conv2 = nn.Conv2d(conv2_in, 32, 3, padding=1)

        if self.use_relational:
            self.rel_block = RelBiasRelationalBlock(channels=32, num_heads=4, view_size=VIEW_SIZE)

        v_dim = 32 * VIEW_SIZE * VIEW_SIZE
        trunk_in = v_dim + lang_dim + n_dirs  # 1604
        self.trunk_fc = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.gru_norm = nn.LayerNorm(hidden)
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
        lang = self.encode_language(tokens)
        h1 = self.encode_h1(image)
        if not self.use_pointer:
            h_mod = h1
            h2 = self.act(self.conv2(h_mod))
        elif self.use_map_inject:
            h_mod = self.pointer(h1, lang, return_map=False)
            h2 = self.act(self.conv2(h_mod))
            if self.use_relational:
                h2 = self.rel_block(h2, return_map=False)
        else:
            h_mod = self.pointer(h1, lang, return_map=False)
            if self.use_relational:
                h_mod = self.rel_block(h_mod, return_map=False)
            h2 = self.act(self.conv2(h_mod))
        return h1, h_mod, h2, lang

    def _frame_features(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        _, _, h2, lang = self.extract_layers(image, tokens)
        v = torch.flatten(h2, 1)
        d = F.one_hot(direction.long(), num_classes=N_DIRS).float()
        return self.trunk_fc(torch.cat([v, lang, d], dim=-1))

    def forward_sequence(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        directions: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence forward for episode batches.

        Args:
            images:     (B, T, 7, 7, 3)
            tokens:     (B, T, L)
            directions: (B, T)
            hx:         (1, B, hidden) or None
        Returns:
            logits: (B, T, n_actions), h_next: (1, B, hidden)
        """
        B, T = images.shape[:2]
        flat_img = images.reshape(B * T, *images.shape[2:])
        flat_tok = tokens.reshape(B * T, tokens.shape[-1])
        flat_dir = directions.reshape(B * T)
        feats = self._frame_features(flat_img, flat_tok, flat_dir).view(B, T, self.hidden)
        gru_out, h_next = self.gru(feats, hx)
        fused = feats + self.gru_norm(gru_out)
        return self.head(fused), h_next

    def forward(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        direction: torch.Tensor,
        hx: torch.Tensor | None = None,
    ):
        """Single-step forward compatible with stateless probes (hx=None) or stateful rollouts."""
        logits, h_next = self.forward_sequence(
            image.unsqueeze(1), tokens.unsqueeze(1), direction.unsqueeze(1), hx=hx
        )
        if hx is None and not self.training:
            return logits.squeeze(1)
        return logits.squeeze(1), h_next

    @torch.no_grad()
    def step(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        direction: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, h_next = self.forward_sequence(
            image.unsqueeze(1), tokens.unsqueeze(1), direction.unsqueeze(1), hx=hx
        )
        return logits.squeeze(1), h_next

    @torch.no_grad()
    def extract_diagnostics(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> dict:
        if not self.use_pointer:
            return {"variant": self.variant}
        lang = self.encode_language(tokens)
        h1 = self.encode_h1(image)
        h_mod, smap, extra = self.pointer(h1, lang, return_map=True)
        out: dict = {
            "variant": self.variant,
            "spatial_map": smap[0, 0].detach().cpu().numpy(),
            **extra,
        }
        if self.use_relational:
            if self.use_map_inject:
                h_geo = self.act(self.conv2(h_mod))
                _, rel_norm, rex = self.rel_block(h_geo, return_map=True)
            else:
                _, rel_norm, rex = self.rel_block(h_mod, return_map=True)
            out["rel_delta_norm"] = rel_norm[0, 0].detach().cpu().numpy()
            out.update(rex)
        return out

    def scalar_stats(self) -> dict:
        stats = {}
        if self.use_pointer and hasattr(self.pointer, "raw_lambda"):
            stats["lambda"] = [round(float(F.softplus(self.pointer.raw_lambda).item()), 5)]
        if self.use_relational:
            stats["rel_scale"] = round(float(self.rel_block.rel_scale.item()), 5)
        return stats


def build(variant: str, **kw) -> nn.Module:
    if variant in SINGLE_FRAME_VARIANTS:
        return GeoRelBCNet(variant=variant, **kw)
    if variant in RECURRENT_VARIANTS:
        return GeoRelGRUBCNet(variant=variant, **kw)
    raise ValueError(f"Unknown variant {variant!r}; expected one of {ALL_GEOREL_VARIANTS}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
