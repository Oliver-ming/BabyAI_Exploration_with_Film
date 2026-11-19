"""支线 2 扩展消融 v2 · 基于文献归纳偏置与 v1 机制诊断的 7 种 Attention 改进架构。

严格复用冻结基线 `BCNet` 主干（嵌入 + conv1 + conv2 + flatten + trunk），仅替换 `ReLU(conv1)` 与 `conv2` 之间的跨模态注意力模块：
  1. gated_attn_chaplot    : Chaplot et al. (AAAI 2018) 经典通道门控 σ(W_l ℓ) ⊙ h1
  2. cross_mha_nopos       : v1 Cross-MHA 的 NoPE 消融（移除绝对 pos_emb，检验 S7N4/S8N6 空间外推恢复）
  3. res_spatial_softmax   : Wang et al. (CVPR 2017) + Xu et al. (ICML 2015) 残差空间 Softmax (1 + λ·49α) ⊙ h1
  4. mh_res_spatial_softmax: Mott et al. (NeurIPS 2019) 多头竞争 + 残差空间 Softmax（4 头 × 8 通道）
  5. attn_map_inject       : Mott et al. (NeurIPS 2019) / CoordConv 显式空间热图拼接 [h1; M] → conv2(36→32)
  6. token_coattn_spatial  : Lu et al. (NeurIPS 2016) + Kim et al. (NeurIPS 2018) BAN + BabyAI AttGRU 词元-网格协同注意力
  7. mh_spatial_relational : Zambaldi et al. (ICLR 2019) Relational RL（4 头残差空间 Softmax + 自我中心线性坐标关系自注意力）
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

ATTN_V2_VARIANTS = (
    "gated_attn_chaplot",
    "cross_mha_nopos",
    "res_spatial_softmax",
    "mh_res_spatial_softmax",
    "attn_map_inject",
    "token_coattn_spatial",
    "mh_spatial_relational",
)

_SOFTPLUS_ONE_INIT = math.log(math.e - 1.0)


class ChaplotGatedAttnBlock(nn.Module):
    """Chaplot et al. (AAAI 2018): 语义驱动 Sigmoid 通道门控 σ(W_l ℓ) ⊙ h。"""

    def __init__(self, lang_dim: int, channels: int):
        super().__init__()
        self.gate_proj = nn.Linear(lang_dim, channels)

    def forward(self, h: torch.Tensor, lang: torch.Tensor, return_map: bool = False):
        gate = torch.sigmoid(self.gate_proj(lang))  # (B, C)
        out = gate.unsqueeze(-1).unsqueeze(-1) * h
        if not return_map:
            return out
        # 仅作对比可视化：计算门控调制后相对变化，空间图本身按通道均值为标量基准
        delta_map = (out - h).norm(dim=1, keepdim=True)
        return out, delta_map, {"channel_gate": gate[0].detach().cpu().numpy()}


class CrossAttnNoPosBlock(nn.Module):
    """v1 Cross-MHA 移除绝对 2D pos_emb 的对照组，保持零初始化残差。"""

    def __init__(self, channels: int = 32, num_heads: int = 4):
        super().__init__()
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
        q_in = self.q_norm(h.flatten(2).transpose(1, 2))  # (B, 49, C), 无 pos_emb
        key_padding_mask = tokens == PAD_IDX
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
        delta_norm = delta.norm(dim=1, keepdim=True)
        return out, delta_norm, {
            "res_scale": float(self.res_scale.item()),
            "token_attn_49xL": attn_w[0].detach().cpu().numpy(),
        }


class MultiHeadSpatialSoftmaxBlock(nn.Module):
    """统一的多头语言→7×7 空间 Softmax 竞争模块（支持残差乘法或显式热图拼接）。"""

    def __init__(
        self,
        lang_dim: int,
        channels: int,
        num_heads: int = 1,
        mode: str = "residual",
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} 必须能被 num_heads={num_heads} 整除")
        if mode not in ("residual", "concat_map"):
            raise ValueError(f"不支持的 mode: {mode}")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.mode = mode

        self.q_proj = nn.Linear(lang_dim, channels)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        if mode == "residual":
            self.raw_lambda = nn.Parameter(torch.full((num_heads,), _SOFTPLUS_ONE_INIT))

    def compute_maps(self, h: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """返回 (B, K, H, W) 的尺度归一化竞争热图 49 * softmax。"""
        B, C, H, W = h.shape
        q = self.q_proj(lang).view(B, self.num_heads, self.head_dim, 1, 1)
        k = self.k_proj(h).view(B, self.num_heads, self.head_dim, H, W)
        scores = (q * k).sum(dim=2) / math.sqrt(self.head_dim)  # (B, K, H, W)
        alpha = F.softmax(scores.flatten(2), dim=-1).view(B, self.num_heads, H, W)
        return float(H * W) * alpha

    def forward(self, h: torch.Tensor, lang: torch.Tensor, return_map: bool = False):
        B, C, H, W = h.shape
        M = self.compute_maps(h, lang)  # (B, K, H, W)
        if self.mode == "concat_map":
            out = torch.cat([h, M], dim=1)  # (B, C+K, H, W)
            if not return_map:
                return out
            return out, M.mean(dim=1, keepdim=True), {"head_maps": M[0].detach().cpu().numpy()}

        lam = F.softplus(self.raw_lambda).view(1, self.num_heads, 1, 1, 1)
        h_heads = h.view(B, self.num_heads, self.head_dim, H, W)
        out = ((1.0 + lam * M.unsqueeze(2)) * h_heads).reshape(B, C, H, W)
        if not return_map:
            return out
        # 有效调制系数：1 + mean_k(lam_k * M_k)
        eff_mult = (1.0 + lam.squeeze(2) * M).mean(dim=1, keepdim=True)
        return out, eff_mult, {
            "head_maps": M[0].detach().cpu().numpy(),
            "lambda": [round(float(x), 5) for x in F.softplus(self.raw_lambda).detach().cpu().tolist()],
        }


class TokenCoAttnSpatialBlock(nn.Module):
    """词元-网格双向协同注意力：先分头聚合 L 个指令词元，再做 7×7 空间 Softmax 竞争与残差价值注入。"""

    def __init__(self, lang_dim: int, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} 必须能被 num_heads={num_heads} 整除")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.tok_proj = nn.Linear(lang_dim, channels)
        self.tok_score = nn.Linear(channels, num_heads)
        self.q_proj = nn.Linear(lang_dim, channels)
        self.v_proj = nn.Linear(lang_dim, channels)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.raw_lambda = nn.Parameter(torch.full((num_heads,), _SOFTPLUS_ONE_INIT))
        self.val_scale = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        h: torch.Tensor,
        tok_emb: torch.Tensor,
        tokens: torch.Tensor,
        return_map: bool = False,
    ):
        B, C, H, W = h.shape
        valid = tokens != PAD_IDX  # (B, L)

        # (B, L, K) -> (B, K, L)
        tok_logits = self.tok_score(torch.tanh(self.tok_proj(tok_emb))).transpose(1, 2)
        tok_logits = tok_logits.masked_fill(~valid.unsqueeze(1), -1e9)
        w = F.softmax(tok_logits, dim=-1) * valid.unsqueeze(1).float()  # (B, K, L)

        Q_tok = self.q_proj(tok_emb).view(B, -1, self.num_heads, self.head_dim)  # (B, L, K, d)
        V_tok = self.v_proj(tok_emb).view(B, -1, self.num_heads, self.head_dim)
        q = torch.einsum("bkl,blkd->bkd", w, Q_tok)  # (B, K, d)
        v = torch.einsum("bkl,blkd->bkd", w, V_tok)  # (B, K, d)

        k = self.k_proj(h).view(B, self.num_heads, self.head_dim, H, W)
        scores = (q.unsqueeze(-1).unsqueeze(-1) * k).sum(dim=2) / math.sqrt(self.head_dim)
        M = float(H * W) * F.softmax(scores.flatten(2), dim=-1).view(B, self.num_heads, H, W)

        lam = F.softplus(self.raw_lambda).view(1, self.num_heads, 1, 1, 1)
        beta = self.val_scale.view(1, self.num_heads, 1, 1, 1)
        h_heads = h.view(B, self.num_heads, self.head_dim, H, W)
        v_broad = v.unsqueeze(-1).unsqueeze(-1)
        out = ((1.0 + lam * M.unsqueeze(2)) * h_heads + beta * M.unsqueeze(2) * v_broad).reshape(B, C, H, W)

        if not return_map:
            return out
        eff_mult = (1.0 + lam.squeeze(2) * M).mean(dim=1, keepdim=True)
        return out, eff_mult, {
            "head_maps": M[0].detach().cpu().numpy(),
            "token_weights": w[0].detach().cpu().numpy(),
            "lambda": [round(float(x), 5) for x in F.softplus(self.raw_lambda).detach().cpu().tolist()],
            "beta": [round(float(x), 5) for x in self.val_scale.detach().cpu().tolist()],
        }


class EgocentricRelationalBlock(nn.Module):
    """Zambaldi et al. (ICLR 2019) 关系型自注意力：以自我中心归一化线性坐标作为平移一致 2D 几何基底。"""

    def __init__(self, channels: int = 32, num_heads: int = 4, view_size: int = VIEW_SIZE):
        super().__init__()
        ys = torch.linspace(-1.0, 1.0, view_size)
        xs = torch.linspace(-1.0, 1.0, view_size)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        # 注意 MiniGrid obs["image"] 内部顺序是 (H=W=7)，保持展平顺序与 h.flatten(2) 完全一致
        coord_grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)  # (1, 49, 2)
        self.register_buffer("coord_grid", coord_grid, persistent=False)

        self.coord_proj = nn.Linear(2, channels)
        self.rel_norm = nn.LayerNorm(channels)
        self.rel_mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.rel_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, return_map: bool = False):
        B, C, H, W = h.shape
        x_seq = self.rel_norm(h.flatten(2).transpose(1, 2))  # (B, 49, C)
        coord_emb = self.coord_proj(self.coord_grid)  # (1, 49, C)
        qk = x_seq + coord_emb
        rel_out, rel_attn = self.rel_mha(
            qk, qk, x_seq, need_weights=return_map, average_attn_weights=True
        )
        delta = self.rel_scale * rel_out.transpose(1, 2).view(B, C, H, W)
        out = h + delta
        if not return_map:
            return out
        return out, delta.norm(dim=1, keepdim=True), {
            "rel_scale": float(self.rel_scale.item()),
            "rel_attn_49x49": rel_attn[0].detach().cpu().numpy(),
        }


class CrossModalAttnV2BCNet(nn.Module):
    """与冻结 BCNet 主干严格对齐的 7 种 Attention 改进架构对比网络。"""

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
        if variant not in ATTN_V2_VARIANTS:
            raise ValueError(f"variant 必须是 {ATTN_V2_VARIANTS}, got {variant}")
        self.variant = variant
        self.hidden = hidden

        # 语言与视觉前端（与冻结 BCNet 完全一致）
        self.lang_emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)
        self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
        self.conv1 = nn.Conv2d(3 * obj_emb_dim, channels, 3, padding=1)
        conv2_in = channels + 4 if variant == "attn_map_inject" else channels
        self.conv2 = nn.Conv2d(conv2_in, 32, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        # 仅在 conv1 与 conv2 之间切换 Attention 模块
        if variant == "gated_attn_chaplot":
            self.mod = ChaplotGatedAttnBlock(lang_dim, channels)
        elif variant == "cross_mha_nopos":
            self.mod = CrossAttnNoPosBlock(channels=channels, num_heads=4)
        elif variant == "res_spatial_softmax":
            self.mod = MultiHeadSpatialSoftmaxBlock(lang_dim, channels, num_heads=1, mode="residual")
        elif variant == "mh_res_spatial_softmax":
            self.mod = MultiHeadSpatialSoftmaxBlock(lang_dim, channels, num_heads=4, mode="residual")
        elif variant == "attn_map_inject":
            self.mod = MultiHeadSpatialSoftmaxBlock(lang_dim, channels, num_heads=4, mode="concat_map")
        elif variant == "token_coattn_spatial":
            self.mod = TokenCoAttnSpatialBlock(lang_dim, channels, num_heads=4)
        elif variant == "mh_spatial_relational":
            self.mod = MultiHeadSpatialSoftmaxBlock(lang_dim, channels, num_heads=4, mode="residual")
            self.rel_mod = EgocentricRelationalBlock(channels=channels, num_heads=4, view_size=VIEW_SIZE)

        v_dim = 32 * VIEW_SIZE * VIEW_SIZE
        trunk_in = v_dim + lang_dim + n_dirs  # 1604
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

    def _encode_h1(self, image: torch.Tensor) -> torch.Tensor:
        x = image.permute(0, 3, 1, 2)
        feats = [self.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
        return self.act(self.conv1(torch.cat(feats, dim=1)))

    def _modulate(
        self, h1: torch.Tensor, lang: torch.Tensor, tok_emb: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        if self.variant in ("gated_attn_chaplot", "res_spatial_softmax", "mh_res_spatial_softmax", "attn_map_inject"):
            return self.mod(h1, lang, return_map=False)
        if self.variant in ("cross_mha_nopos", "token_coattn_spatial"):
            return self.mod(h1, tok_emb, tokens, return_map=False)
        if self.variant == "mh_spatial_relational":
            h_lang = self.mod(h1, lang, return_map=False)
            return self.rel_mod(h_lang, return_map=False)
        raise RuntimeError(self.variant)

    def forward(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        lang, tok_emb = self.encode_language(tokens)
        h1 = self._encode_h1(image)
        h_mod = self._modulate(h1, lang, tok_emb, tokens)
        h2 = self.act(self.conv2(h_mod))
        v = torch.flatten(h2, 1)
        d = F.one_hot(direction.long(), num_classes=N_DIRS).float()
        return self.trunk(torch.cat([v, lang, d], dim=-1))

    @torch.no_grad()
    def extract_diagnostics(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> dict:
        """提取 7×7 空间权重/有效调制倍数、多头热图、词元权重与可学习标量，用于机制归因。"""
        lang, tok_emb = self.encode_language(tokens)
        h1 = self._encode_h1(image)
        out: dict = {"variant": self.variant}
        if self.variant in ("gated_attn_chaplot", "res_spatial_softmax", "mh_res_spatial_softmax", "attn_map_inject"):
            _, smap, extra = self.mod(h1, lang, return_map=True)
            out["spatial_map"] = smap[0, 0].detach().cpu().numpy()
            out.update(extra)
        elif self.variant in ("cross_mha_nopos", "token_coattn_spatial"):
            _, smap, extra = self.mod(h1, tok_emb, tokens, return_map=True)
            out["spatial_map"] = smap[0, 0].detach().cpu().numpy()
            out.update(extra)
        elif self.variant == "mh_spatial_relational":
            h_lang, smap, extra1 = self.mod(h1, lang, return_map=True)
            _, rel_norm, extra2 = self.rel_mod(h_lang, return_map=True)
            out["spatial_map"] = smap[0, 0].detach().cpu().numpy()
            out["rel_delta_norm"] = rel_norm[0, 0].detach().cpu().numpy()
            out.update(extra1)
            out.update(extra2)
        return out

    @torch.no_grad()
    def extract_spatial_map(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> dict:
        """与 v1 `extract_spatial_map` 保持兼容接口。"""
        return self.extract_diagnostics(image, tokens, direction)

    @torch.no_grad()
    def scalar_stats(self) -> dict:
        stats = {}
        if hasattr(self.mod, "raw_lambda"):
            vals = F.softplus(self.mod.raw_lambda).detach().cpu().tolist()
            stats["lambda"] = [round(float(x), 5) for x in vals]
        if hasattr(self.mod, "val_scale"):
            vals = self.mod.val_scale.detach().cpu().tolist()
            stats["beta"] = [round(float(x), 5) for x in vals]
        if hasattr(self.mod, "res_scale"):
            stats["res_scale"] = round(float(self.mod.res_scale.item()), 5)
        if hasattr(self, "rel_mod"):
            stats["rel_scale"] = round(float(self.rel_mod.rel_scale.item()), 5)
        return stats


def build(variant: str, **kw) -> CrossModalAttnV2BCNet:
    return CrossModalAttnV2BCNet(variant=variant, **kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
