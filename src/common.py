"""共享常量、seed 划分与 BC 模型定义（P3 / 赛道 2 · BabyAI GoToLocal）。

本模块被 gen_expert / train_bc / eval_closed_loop / render_episodes / make_figures 共用，
是「实验协议」的唯一出处。任何脚本都不得自行改写 seed 分段。

--------------------------------------------------------------------------
关于 expert episode 预算与 seed 划分的必要修正（详见 reports/data_audit.md §0）
--------------------------------------------------------------------------
指导文档 §1.3 规定:

    TRAIN_SEEDS = range(0, 4000)      # 用于生成 5000 条专家演示
    VAL_SEEDS   = range(4000, 4500)   # 用于选 checkpoint
    TEST_SEEDS  = range(4500, 4800)   # 300 个，用于最终评估
    TEST_SEEDS_S7 = range(10000, 10300)

存在两个问题：

1. `range(0, 4000)` 只有 4000 个 seed，而同一 env seed 只能产出**一个**确定性 episode，
   所以最多 4000 条互不重复的专家 episode，拿不到 5000 条（循环抽样会造成 1000 条重复样本）。
2. 按 §1.2 ① 必须用**验证集动作准确率**选 checkpoint，而动作准确率需要 VAL seed 上的专家动作标签，
   这意味着额外要跑 500 条专家 episode。若训练集仍用满 5000 条，专家 episode 总数会变成 5500，
   违反手册「最多 5000 条成功专家 episode」的硬性规定。

本实现的处理（完全合规版）：

    训练专家：0..4000 (4000 条) ∪ 4800..5300 (500 条) = 4500 条
    验证专家：4000..4500 (500 条)                       =  500 条
    --------------------------------------------------------------
    专家 episode 总数                                  = 5000 条（正好等于上限，未超）

VAL / TEST / TEST_S7 三个段与文档逐字一致。"测试 seed 绝不进训练"仍然成立：
4800..5300 与 VAL(4000..4500)、TEST(4500..4800) 全不相交，也未参与任何评估。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# 环境
# --------------------------------------------------------------------------
ENV_ID = "BabyAI-GoToLocalS6N3-v0"
ENV_ID_S7 = "BabyAI-GoToLocalS7N4-v0"
N_ACTIONS = 7
N_DIRS = 4
VIEW_SIZE = 7
N_OBJECT_IDS = 11  # Minigrid OBJECT_TO_IDX: 0..10
MAX_STEPS_S6N3 = 36
MAX_STEPS_S7N4 = 49

# --------------------------------------------------------------------------
# seed 分段（实验协议的唯一定义）
# --------------------------------------------------------------------------
TRAIN_SEEDS_PRIMARY = list(range(0, 4000))
TRAIN_SEEDS_SUPPLEMENT = list(range(4800, 5300))  # 见模块 docstring
TRAIN_SEEDS = TRAIN_SEEDS_PRIMARY + TRAIN_SEEDS_SUPPLEMENT  # 4500 个唯一 seed
VAL_SEEDS = list(range(4000, 4500))  # 500
TEST_SEEDS = list(range(4500, 4800))  # 300
TEST_SEEDS_S7 = list(range(10000, 10300))  # 300

# 专家 episode 预算（手册上限 5000）：训练 4500 + 验证 500 = 5000
N_EXPERT_TRAIN = len(TRAIN_SEEDS)  # 4500
N_EXPERT_VAL = len(VAL_SEEDS)  # 500
N_EXPERT_TOTAL = N_EXPERT_TRAIN + N_EXPERT_VAL  # 5000
EXPERT_ALL_SEEDS = TRAIN_SEEDS + VAL_SEEDS  # 4500 + 500，恰好 5000 个唯一 seed
VAL_SET = frozenset(VAL_SEEDS)


def split_of(seed: int) -> str:
    """给定 seed 返回它属于哪个数据划分。"""
    if seed in VAL_SET:
        return "val"
    if seed in TRAIN_SEED_SET:
        return "train"
    if seed in TEST_SET:
        return "test"
    if seed in TEST_S7_SET:
        return "test_s7"
    return "unknown"


TRAIN_SEED_SET = frozenset(TRAIN_SEEDS)
TEST_SET = frozenset(TEST_SEEDS)
TEST_S7_SET = frozenset(TEST_SEEDS_S7)

# --------------------------------------------------------------------------
# 语言
# --------------------------------------------------------------------------
COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
TYPES = ["ball", "box", "key"]
VOCAB = ["<pad>", "go", "to", "the", "a"] + COLORS + TYPES  # 14
PAD_IDX = 0
MAX_TOKENS = 5


def mission_to_ids(mission: str) -> list[int]:
    """把 mission 字符串转成 token id 序列（未知词直接报错，便于尽早发现词表漏洞）。"""
    ids = []
    for w in mission.split():
        try:
            ids.append(VOCAB.index(w))
        except ValueError as e:  # pragma: no cover - 防护性
            raise KeyError(f"词表未覆盖 token {w!r}（mission={mission!r}）") from e
    return ids


def parse_mission(mission: str) -> dict:
    """`go to <article> <color> <type>` -> 结构化字段。"""
    parts = mission.split()
    return {
        "article": parts[2] if len(parts) >= 5 else None,
        "color": parts[3] if len(parts) >= 5 else None,
        "obj_type": parts[4] if len(parts) >= 5 else None,
    }


# --------------------------------------------------------------------------
# 模型（§6.1 规定架构）
# --------------------------------------------------------------------------
class BCNet(nn.Module):
    """image + mission + direction -> 7 维动作 logits。

    image (7,7,3) uint8：
      - img_mode="embed"  : 每个通道用 nn.Embedding 编码（类别索引，默认，推荐）
      - img_mode="div255" : 转 float 后 /255（对照用）
    mission (str)：固定词表 + 32 维平均词嵌入（只对非 pad 词求均值）
    direction (int 0..3)：one-hot(4)
    """

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        n_actions: int = N_ACTIONS,
        n_dirs: int = N_DIRS,
        img_mode: str = "embed",
        obj_emb_dim: int = 8,
        lang_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_mode = img_mode
        self.lang_dim = lang_dim

        if img_mode == "embed":
            self.obj_emb = nn.Embedding(N_OBJECT_IDS, obj_emb_dim)
            in_ch = 3 * obj_emb_dim
        elif img_mode == "div255":
            in_ch = 3
        else:
            raise ValueError(img_mode)

        self.vision = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 32 * 7 * 7 = 1568
        )
        v_dim = 32 * VIEW_SIZE * VIEW_SIZE

        self.lang_emb = nn.Embedding(vocab_size, lang_dim, padding_idx=PAD_IDX)
        self.trunk = nn.Sequential(
            nn.Linear(v_dim + lang_dim + n_dirs, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B,7,7,3) uint8/long -> (B, in_ch, 7, 7)。

        HWC -> CHW 必须显式 permute；embed 模式下每个通道查表后再把 (B,H,W,D)
        转成 (B,D,H,W) 才能沿通道维拼接。
        """
        x = image.permute(0, 3, 1, 2)  # (B,3,7,7)
        if self.img_mode == "embed":
            feats = [
                self.obj_emb(x[:, c].long()).permute(0, 3, 1, 2) for c in range(3)
            ]
            return torch.cat(feats, dim=1)
        return x.float() / 255.0

    def forward(
        self, image: torch.Tensor, tokens: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        v = self.vision(self.encode_image(image))

        mask = (tokens != PAD_IDX).unsqueeze(-1)  # (B,L,1)
        emb = self.lang_emb(tokens)  # (B,L,lang_dim)
        # 用 mean(emb[valid_tokens])，而不是 mean(emb[all_tokens])
        counts = mask.sum(dim=1).clamp(min=1)
        l = (emb * mask).sum(dim=1) / counts

        d = nn.functional.one_hot(direction.long(), num_classes=N_DIRS).float()
        return self.trunk(torch.cat([v, l, d], dim=-1))


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------
def encode_batch(obs_list, blank_language: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 obs 字典列表编码成 (image, tokens, direction) 三个 tensor。"""
    images, toks, dirs = [], [], []
    pad = [PAD_IDX] * MAX_TOKENS
    for obs in obs_list:
        images.append(torch.as_tensor(obs["image"]))
        if blank_language:
            toks.append(torch.tensor(pad, dtype=torch.long))
        else:
            ids = mission_to_ids(obs["mission"])
            ids = (ids + [PAD_IDX] * MAX_TOKENS)[:MAX_TOKENS]
            toks.append(torch.tensor(ids, dtype=torch.long))
        dirs.append(int(obs["direction"]))
    return (
        torch.stack(images),
        torch.stack(toks),
        torch.tensor(dirs, dtype=torch.long),
    )


def predict_actions(model: BCNet, obs_list) -> list[int]:
    """给定一批 obs 返回贪心动作下标（推理路径，eval 模式 + no_grad）。"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        img, toks, d = encode_batch(obs_list, getattr(model, "blank_language", False))
        return model(img, toks, d).argmax(dim=-1).tolist()


def load_run(run_dir: str | Path) -> tuple[BCNet, dict, dict]:
    """加载一个 run 目录，返回 (model, config, vocab_info)。"""
    import json

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    vocab = json.loads((run_dir / "vocab.json").read_text())
    model = BCNet(
        vocab_size=len(vocab["vocab"]),
        img_mode=cfg["arch"]["img_mode"],
        obj_emb_dim=cfg["arch"]["obj_emb_dim"],
        lang_dim=cfg["arch"]["lang_dim"],
        hidden=cfg["arch"]["hidden"],
        dropout=cfg["arch"]["dropout"],
    )
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.blank_language = bool(cfg["arch"].get("no_language", False))
    model.eval()
    return model, cfg, vocab


def set_all_seeds(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
