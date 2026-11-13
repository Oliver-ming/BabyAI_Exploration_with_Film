"""支线 2 · FiLM 跨模态融合的架构变体。

设计约束全部来自 `LOG.md §2.2`（文献 1/2/3/5/7）：

  1. γ 从恒等（1）、β 从 0 起步，参数化为「偏离量」`1+Δγ` / `0+Δβ`
  2. γ/β 生成器是**单层线性**（文献 2 明确说用 RNN 预测 γ/β 是错的）
  3. 默认必须开 weight decay（文献 2：关掉掉 ~10% 准确率）
  4. 只在**早期**放 1 层 FiLM（文献 2：单层早层 ≈ 4 层遍布全网络）
  5. 不用预训练骨干（文献 5），保持 7×7 类别 id 观测的两层小卷积
  6. 「FiLM + 残差」作为**独立消融**，不默认加（文献 3 实测时好时坏）

变体：
  concat       基线（复用 src/common.BCNet 的结构，但这里独立实现以便公平对比）
  film         拼接 + conv1 后 FiLM          ← 严格超集，隔离 FiLM 的额外容量
  film_only    只用 FiLM，不再拼接            ← 检验乘性调制是否已足够
  film_res     film + 残差（独立消融）
  concat_wide  参数量匹配对照：加宽隐层到与 film 等参数
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

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

VARIANTS = ("concat", "film", "film_only", "film_res", "concat_wide")


class FiLMGenerator(nn.Module):
    """从语言嵌入生成逐通道 (Δγ, Δβ)。单层线性，零初始化 => 恒等起步。"""

    def __init__(self, lang_dim: int, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.proj = nn.Linear(lang_dim, 2 * n_channels)
        nn.init.zeros_(self.proj.weight)  # 约束 1：恒等起步
        nn.init.zeros_(self.proj.bias)

    def forward(self, lang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.proj(lang)  # (B, 2C)
        return raw[:, : self.n_channels], raw[:, self.n_channels :]

    def deviation_penalty(self, lang: torch.Tensor) -> torch.Tensor:
        """对偏离量做 L2（文献 2/7 的核心建议）。"""
        dg, db = self.forward(lang)
        return (dg.pow(2).mean() + db.pow(2).mean())


class FiLMBCNet(nn.Module):
    """image + mission + direction -> 7 动作 logits，语言可通过 FiLM 调制视觉。"""

    def __init__(
        self,
        variant: str = "film",
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
        if variant not in VARIANTS:
            raise ValueError(f"variant 必须是 {VARIANTS}")
        self.variant = variant
        self.use_film = variant in ("film", "film_only", "film_res")
        self.use_concat = variant in ("concat", "film", "film_res", "concat_wide")

        # ---- 语言（与基线一致：固定词表 + 平均词嵌入）----
        self.lang_emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)

        # ---- 视觉：3 通道查表 -> conv(32) -> conv(32) ----
        self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
        in_ch = 3 * obj_emb_dim
        self.conv1 = nn.Conv2d(in_ch, film_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(film_channels, 32, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        if self.use_film:
            self.film = FiLMGenerator(lang_dim, film_channels)

        v_dim = 32 * VIEW_SIZE * VIEW_SIZE
        if variant == "concat_wide":
            # 参数量匹配：把隐层加宽，使总参数量与 film 组接近
            hidden = self._matched_hidden(v_dim, lang_dim, n_dirs, 128)
        self.hidden = hidden

        # 视觉特征**永远**进 trunk；是否额外拼接语言向量才是变体差别
        # （film_only = 语言只经 FiLM 进入，视觉仍完整保留）
        trunk_in = v_dim + (lang_dim if self.use_concat else 0) + n_dirs
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    @staticmethod
    def _matched_hidden(v_dim: int, lang_dim: int, n_dirs: int, base_hidden: int) -> int:
        """让 concat_wide 的总参数量 ≈ film 组。

        film 比 concat 多出的参数 = film generator 的 2*C*lang_dim + 2*C
        （C=32, lang_dim=32 -> 2112）。把它换算成 trunk 第一层的加宽量：
        delta_hidden = 2112 / (v_dim + lang_dim + n_dirs)
        """
        extra = 2 * 32 * lang_dim + 2 * 32
        denom = v_dim + lang_dim + n_dirs
        return base_hidden + max(1, round(extra / denom))

    # ------------------------------------------------------------------
    def encode_image(self, image: torch.Tensor, lang: torch.Tensor | None = None):
        """视觉编码；若启用 FiLM，用 lang 生成的 (γ,β) 调制 conv1 的输出。"""
        x = image.permute(0, 3, 1, 2)  # (B,3,7,7) 显式 HWC->CHW
        feats = [self.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)]
        h = self.act(self.conv1(torch.cat(feats, dim=1)))  # (B,C,7,7)

        if self.use_film:
            if lang is None:
                raise ValueError("启用 FiLM 时必须传入 lang")
            dg, db = self.film.forward(lang)  # (B,C), (B,C)
            gamma = (1.0 + dg).unsqueeze(-1).unsqueeze(-1)
            beta = db.unsqueeze(-1).unsqueeze(-1)
            h = h + gamma * h + beta if self.variant == "film_res" else gamma * h + beta

        h = self.act(self.conv2(h))
        return torch.flatten(h, 1)

    def encode_language(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != PAD_IDX).unsqueeze(-1)  # (B,L,1)
        emb = self.lang_emb(tokens)
        counts = mask.sum(dim=1).clamp(min=1)
        return (emb * mask).sum(dim=1) / counts

    def forward(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        lang = self.encode_language(tokens)
        v = self.encode_image(image, lang)

        d = nn.functional.one_hot(direction.long(), num_classes=N_DIRS).float()
        # 视觉特征恒在；film_only 只是不再把 lang 拼进 trunk
        parts = [v] + ([lang] if self.use_concat else []) + [d]
        return self.trunk(torch.cat(parts, dim=-1))

    def film_penalty(self, tokens: torch.Tensor) -> torch.Tensor:
        """训练时加到 loss 上的 FiLM 偏离正则（约束 1）。"""
        if not self.use_film:
            return torch.zeros((), device=tokens.device)
        return self.film.deviation_penalty(self.encode_language(tokens))

    @torch.no_grad()
    def film_stats(self, tokens: torch.Tensor) -> dict:
        if not self.use_film:
            return {}
        dg, db = self.film.forward(self.encode_language(tokens))
        return {
            "abs_dgamma_mean": round(float(dg.abs().mean()), 5),
            "abs_dbeta_mean": round(float(db.abs().mean()), 5),
        }


def build(variant: str, **kw) -> FiLMBCNet:
    return FiLMBCNet(variant=variant, **kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
