"""支线 1 · 语言编码器与模型变体。

设计约束（见 LOG.md §2.2 与审计修订）：

  1. GRU 按 BabyAI 1.1 的 Small 配置（单向 128 维，无 attention）；另设 gru_small（32 维）
     用于与 Transformer / DAN / PosMLP 在 ~7.8k 语言参数量级做受控对比
  2. Transformer 极小（1 层 2 头 d=32、无预训练），检验小数据下的样本效率
  3. 对照澄清（审计修复）：
       - mean_pos 只是 (emb + pos).mean()，在固定长度 L=5 无非线性时位置向量坍缩为常量偏置，
         对任意词置换完全不敏感（代数对消 bug）。
       - dan（Deep Averaging Network / DeepSets，Iyyer et al. ACL 2015）：词级非线性 MLP
         后池化再投影，严格置换不变（词序不敏感），参数量精确匹配 gru_small（7,855 vs 7,840），
         用来分离「非线性词表征投影」与「词序交互建模」。
       - pos_mlp：在 emb+pos 之后先过非线性 MLP 再池化（7,832 参数，匹配 gru_small），
         修复 mean_pos 的线性对消 bug，使模型无需序列交互即可区分槽位 3/4。
  4. 默认融合方式保持拼接（唯一变量=语言编码器）；另设 film_gru_small / film_gru
     组合组，检验在 FiLM 跨模态融合基础上进一步使用顺序编码器的叠加效应（与 b2_film_nwd 对照）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    MAX_TOKENS,
    N_ACTIONS,
    N_DIRS,
    N_OBJECT_IDS,
    PAD_IDX,
    VIEW_SIZE,
    VOCAB,
)

VARIANTS = (
    "mean",
    "mean_pos",
    "mean_wide",
    "dan",
    "pos_mlp",
    "gru_small",
    "gru",
    "transformer",
    "film_gru_small",
    "film_gru",
)

FILM_VARIANTS = ("film_gru_small", "film_gru")


class MeanPoolEncoder(nn.Module):
    """平均池化（基线）。可选可学习位置编码（mean_pos，固定长度下存在线性对消）。"""

    def __init__(self, vocab_size: int, lang_dim: int = 32, use_pos: bool = False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)
        self.use_pos = use_pos
        if use_pos:
            self.pos = nn.Embedding(MAX_TOKENS, lang_dim)
        self.out_dim = lang_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != PAD_IDX).unsqueeze(-1)  # (B,L,1)
        e = self.emb(tokens)
        if self.use_pos:
            idx = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
            e = e + self.pos(idx)
        counts = mask.sum(dim=1).clamp(min=1)
        return (e * mask).sum(dim=1) / counts


class DANEncoder(nn.Module):
    """非线性词袋编码器（DeepSets / Deep Averaging Network，严格置换不变）。

    公式：rho( mean_{t: valid} phi(emb(w_t)) )
    在池化前加两层 ReLU MLP，但完全不含位置编码或序列交互。
    默认 mlp_hidden=59 时语言分支共 7,855 参数，精确匹配 gru_small（7,840）。
    """

    def __init__(self, vocab_size: int, lang_dim: int = 32, mlp_hidden: int = 59):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)
        self.phi = nn.Sequential(
            nn.Linear(lang_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
        )
        self.rho = nn.Linear(mlp_hidden, lang_dim)
        self.out_dim = lang_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != PAD_IDX).unsqueeze(-1).float()  # (B,L,1)
        h = self.phi(self.emb(tokens))
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = (h * mask).sum(dim=1) / counts
        return self.rho(pooled)


class PosMLPEncoder(nn.Module):
    """位置感知非线性 MLP 编码器（修复 mean_pos 的线性对消 bug，无跨词交互）。

    公式：rho( mean_{t: valid} phi(emb(w_t) + pos(t)) )
    因为 phi 在平均池化之前引入 ReLU 非线性，不同位置对同一词施加不同映射且不再对消；
    同时词与词之间没有任何循环/注意力交互，严格检验「槽位感知（pos）vs 词序交互（GRU/Trf）」。
    默认 mlp_hidden=58 时语言分支共 7,832 参数（含 pos 160），精确匹配 gru_small（7,840）。
    """

    def __init__(self, vocab_size: int, lang_dim: int = 32, mlp_hidden: int = 58):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)
        self.pos = nn.Embedding(MAX_TOKENS, lang_dim)
        self.phi = nn.Sequential(
            nn.Linear(lang_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
        )
        self.rho = nn.Linear(mlp_hidden, lang_dim)
        self.out_dim = lang_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != PAD_IDX).unsqueeze(-1).float()  # (B,L,1)
        idx = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        h = self.phi(self.emb(tokens) + self.pos(idx))
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = (h * mask).sum(dim=1) / counts
        return self.rho(pooled)


class GRUEncoder(nn.Module):
    """单向 GRU，取最后有效步的隐藏态。"""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden: int = 128, out_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        e = self.emb(tokens)
        out, _h = self.gru(e)
        mask = tokens != PAD_IDX
        idx = mask.sum(dim=1).clamp(min=1) - 1  # (B,)
        last = out[torch.arange(out.size(0), device=out.device), idx]
        return self.proj(last)


class TinyTransformerEncoder(nn.Module):
    """1 层 2 头 d=32 的极小 Transformer，加可学习位置编码，无预训练。"""

    def __init__(self, vocab_size: int, d: int = 32, n_heads: int = 2, out_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD_IDX)
        self.pos = nn.Embedding(MAX_TOKENS, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=2 * d,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.proj = nn.Linear(d, out_dim)
        self.out_dim = out_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pad_mask = tokens == PAD_IDX  # True = 忽略
        idx = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        x = self.emb(tokens) + self.pos(idx)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.proj(pooled)


def build_encoder(variant: str, vocab_size: int = len(VOCAB), lang_dim: int = 32) -> nn.Module:
    if variant == "mean":
        return MeanPoolEncoder(vocab_size, lang_dim, use_pos=False)
    if variant == "mean_pos":
        return MeanPoolEncoder(vocab_size, lang_dim, use_pos=True)
    if variant == "mean_wide":
        return MeanPoolEncoder(vocab_size, lang_dim, use_pos=False)
    if variant == "dan":
        return DANEncoder(vocab_size, lang_dim=lang_dim, mlp_hidden=59)
    if variant == "pos_mlp":
        return PosMLPEncoder(vocab_size, lang_dim=lang_dim, mlp_hidden=58)
    if variant in ("gru", "film_gru"):
        return GRUEncoder(vocab_size, embed_dim=lang_dim, hidden=128, out_dim=lang_dim)
    if variant in ("gru_small", "film_gru_small"):
        return GRUEncoder(vocab_size, embed_dim=lang_dim, hidden=32, out_dim=lang_dim)
    if variant == "transformer":
        return TinyTransformerEncoder(vocab_size, d=lang_dim, n_heads=2, out_dim=lang_dim)
    raise ValueError(f"未知 variant {variant}，可选 {VARIANTS}")


class LangBCNet(nn.Module):
    """与 `src/common.BCNet` 结构严格对齐，仅将语言分支替换为可插拔编码器。"""

    def __init__(
        self,
        variant: str = "mean",
        vocab_size: int = len(VOCAB),
        n_actions: int = N_ACTIONS,
        n_dirs: int = N_DIRS,
        obj_emb_dim: int = 8,
        lang_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.0,
        widen_to_match: bool = True,
    ):
        super().__init__()
        self.variant = variant
        self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
        in_ch = 3 * obj_emb_dim
        self.vision = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.lang = build_encoder(variant, vocab_size, lang_dim)
        self.projector = None
        v_dim = 32 * VIEW_SIZE * VIEW_SIZE
        base_hidden = hidden
        if variant == "mean_wide" and widen_to_match:
            extra = 289407 - 223071  # gru 与 mean 的实测参数差
            base_hidden = hidden + max(1, round(extra / (v_dim + lang_dim + n_dirs)))

        self.trunk = nn.Sequential(
            nn.Linear(v_dim + self.lang.out_dim + n_dirs, base_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_hidden, n_actions),
        )
        self.hidden = base_hidden

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        x = image.permute(0, 3, 1, 2)
        feats = [self.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
        return self.vision(torch.cat(feats, dim=1))

    def encode_language(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lang(tokens)

    def forward(self, image, tokens, direction) -> torch.Tensor:
        v = self.encode_image(image)
        l = self.lang(tokens)
        d = nn.functional.one_hot(direction.long(), num_classes=N_DIRS).float()
        return self.trunk(torch.cat([v, l, d], dim=-1))


class FiLMGenerator(nn.Module):
    """与 b2_film 一致的单层线性 FiLM 生成器（零初始化，恒等起步）。"""

    def __init__(self, lang_dim: int, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.proj = nn.Linear(lang_dim, 2 * n_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, lang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.proj(lang)
        return raw[:, : self.n_channels], raw[:, self.n_channels :]

    def deviation_penalty(self, lang: torch.Tensor) -> torch.Tensor:
        dg, db = self.forward(lang)
        return dg.pow(2).mean() + db.pow(2).mean()


class FiLMLangBCNet(nn.Module):
    """FiLM + 顺序语言编码器（与 branches/b2_film/film_net.py 完全同构，只替换语言分支）。"""

    def __init__(
        self,
        variant: str = "film_gru_small",
        vocab_size: int = len(VOCAB),
        n_actions: int = N_ACTIONS,
        n_dirs: int = N_DIRS,
        obj_emb_dim: int = 8,
        lang_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.0,
        film_channels: int = 32,
    ):
        super().__init__()
        if variant not in FILM_VARIANTS:
            raise ValueError(f"FiLMLangBCNet 仅支持 {FILM_VARIANTS}，收到 {variant}")
        self.variant = variant
        self.hidden = hidden
        self.use_film = True
        self.use_concat = True

        self.lang = build_encoder(variant, vocab_size, lang_dim)
        self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
        in_ch = 3 * obj_emb_dim
        self.conv1 = nn.Conv2d(in_ch, film_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(film_channels, 32, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.film = FiLMGenerator(self.lang.out_dim, film_channels)

        v_dim = 32 * VIEW_SIZE * VIEW_SIZE
        trunk_in = v_dim + self.lang.out_dim + n_dirs
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def encode_language(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lang(tokens)

    def encode_image(self, image: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        x = image.permute(0, 3, 1, 2)
        feats = [self.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
        h = self.act(self.conv1(torch.cat(feats, dim=1)))
        dg, db = self.film(lang)
        gamma = (1.0 + dg).unsqueeze(-1).unsqueeze(-1)
        beta = db.unsqueeze(-1).unsqueeze(-1)
        h = gamma * h + beta
        h = self.act(self.conv2(h))
        return torch.flatten(h, 1)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        lang = self.lang(tokens)
        v = self.encode_image(image, lang)
        d = nn.functional.one_hot(direction.long(), num_classes=N_DIRS).float()
        return self.trunk(torch.cat([v, lang, d], dim=-1))

    def film_penalty(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.film.deviation_penalty(self.lang(tokens))

    @torch.no_grad()
    def film_stats(self, tokens: torch.Tensor) -> dict:
        dg, db = self.film(self.lang(tokens))
        return {
            "abs_dgamma_mean": round(float(dg.abs().mean()), 5),
            "abs_dbeta_mean": round(float(db.abs().mean()), 5),
        }


def build_lang_model(
    variant: str = "mean",
    vocab_size: int = len(VOCAB),
    n_actions: int = N_ACTIONS,
    n_dirs: int = N_DIRS,
    obj_emb_dim: int = 8,
    lang_dim: int = 32,
    hidden: int = 128,
    dropout: float = 0.0,
    widen_to_match: bool = True,
) -> nn.Module:
    if variant in FILM_VARIANTS:
        return FiLMLangBCNet(
            variant=variant,
            vocab_size=vocab_size,
            n_actions=n_actions,
            n_dirs=n_dirs,
            obj_emb_dim=obj_emb_dim,
            lang_dim=lang_dim,
            hidden=hidden,
            dropout=dropout,
        )
    return LangBCNet(
        variant=variant,
        vocab_size=vocab_size,
        n_actions=n_actions,
        n_dirs=n_dirs,
        obj_emb_dim=obj_emb_dim,
        lang_dim=lang_dim,
        hidden=hidden,
        dropout=dropout,
        widen_to_match=widen_to_match,
    )


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def lang_params(m: nn.Module) -> int:
    if hasattr(m, "lang"):
        return sum(p.numel() for p in m.lang.parameters())
    if hasattr(m, "lang_emb"):
        return sum(p.numel() for p in m.lang_emb.parameters())
    return 0


def bag_of_words_equivariant(variant: str) -> bool:
    """严格对任意词置换不敏感的变体（数学性质，用于探针自检断言）。"""
    return variant in ("mean", "bc30", "mean_wide", "dan", "b2_film", "b2_film_nwd", "b2_concat", "b2_concat_nwd")


def fixed_length_bow_invariant(variant: str) -> bool:
    """在固定长度 L=5 无非线性前池化条件下因线性对消而置换不变的变体。"""
    return variant == "mean_pos" or bag_of_words_equivariant(variant)
