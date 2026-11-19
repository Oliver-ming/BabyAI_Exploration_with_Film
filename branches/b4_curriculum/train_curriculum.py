"""支线 4 · 课程学习训练脚本（同预算 4500 训练 + 500 验证，30 epoch，步数匹配随机采样）。

用法:
    OMP_NUM_THREADS=1 python branches/b4_curriculum/train_curriculum.py \
        --variant pace_easy --seed 0 --out runs/b4_pace_easy_seed0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Subset

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b4_curriculum"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    ENV_ID,
    MAX_TOKENS,
    PAD_IDX,
    VOCAB,
    BCNet,
    set_all_seeds,
)
from curriculum_utils import FILM_VARIANTS, VARIANTS, build_curriculum  # noqa: E402
from film_net import FiLMBCNet  # noqa: E402
from train_bc import build_dataset, eval_model, make_ds  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_ep_slices(episodes: list[dict]) -> list[tuple[int, int]]:
    slices = []
    cur = 0
    for ep in episodes:
        nxt = cur + len(ep["actions"])
        slices.append((cur, nxt))
        cur = nxt
    return slices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--film-l2", type=float, default=0.0,
                    help="默认 0.0，与 b2_film_nwd 严格对齐")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lang-dim", type=int, default=32)
    ap.add_argument("--obj-emb-dim", type=int, default=8)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.out:
        args.out = f"runs/b4_{args.variant}_seed{args.seed}"

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    pool_eps, val_eps, schedule_fn, cmeta = build_curriculum(
        args.variant, seed=args.seed, data_dir=args.data_dir
    )
    ep_slices = build_ep_slices(pool_eps)
    tr_full = build_dataset(pool_eps)
    va_full = build_dataset(val_eps)
    n_full_trans = int(len(tr_full[3]))
    assert ep_slices[-1][1] == n_full_trans

    full_ds = make_ds(*tr_full)
    val_loader = DataLoader(make_ds(*va_full), batch_size=512, shuffle=False, num_workers=0)

    use_film = args.variant in FILM_VARIANTS
    if use_film:
        model: nn.Module = FiLMBCNet(
            variant="film",
            vocab_size=len(VOCAB),
            obj_emb_dim=args.obj_emb_dim,
            lang_dim=args.lang_dim,
            hidden=args.hidden,
            dropout=args.dropout,
        )
    else:
        model = BCNet(
            vocab_size=len(VOCAB),
            obj_emb_dim=args.obj_emb_dim,
            lang_dim=args.lang_dim,
            hidden=args.hidden,
            dropout=args.dropout,
            img_mode="embed",
        )

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()

    # 缓存按 stage_name 构造好的索引，避免重复拼接
    cached_loaders: dict[str, tuple[DataLoader, int]] = {}

    for epoch in range(1, args.epochs + 1):
        stage_name, active_ep_indices = schedule_fn(epoch)
        if stage_name not in cached_loaders:
            if len(active_ep_indices) == len(pool_eps):
                loader = DataLoader(
                    full_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    generator=g,
                    num_workers=0,
                )
            else:
                idx_chunks = [
                    np.arange(ep_slices[i][0], ep_slices[i][1], dtype=np.int64)
                    for i in sorted(active_ep_indices)
                ]
                stage_tr_indices = np.concatenate(idx_chunks)
                stage_ds = Subset(full_ds, stage_tr_indices.tolist())
                # 步数匹配：子集也抽取恰好 n_full_trans 条转移/epoch
                sampler = RandomSampler(
                    stage_ds, replacement=True, num_samples=n_full_trans, generator=g
                )
                loader = DataLoader(
                    stage_ds,
                    batch_size=args.batch_size,
                    sampler=sampler,
                    num_workers=0,
                )
            cached_loaders[stage_name] = (loader, len(active_ep_indices))

        train_loader, n_stage_eps = cached_loaders[stage_name]

        model.train()
        run_loss, run_n, run_c = 0.0, 0, 0
        for img, tok, d, act in train_loader:
            logits = model(img, tok, d)
            loss = crit(logits, act)
            if use_film and args.film_l2 > 0.0:
                loss = loss + args.film_l2 * model.film_penalty(tok)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * act.size(0)
            run_n += act.size(0)
            run_c += (logits.argmax(-1) == act).sum().item()

        val_loss, val_acc, _p, _l = eval_model(model, val_loader, crit)
        history.append(
            {
                "epoch": epoch,
                "stage": stage_name,
                "n_stage_episodes": n_stage_eps,
                "epoch_transitions": run_n,
                "train_loss": round(run_loss / run_n, 6),
                "train_acc": round(run_c / run_n, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )
        improved = val_acc > best_acc + 1e-9
        if improved:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(
            f"[b4 {args.variant} s{args.seed}] ep {epoch:2d} ({stage_name}, {n_stage_eps}ep) "
            f"loss={run_loss / run_n:.4f} val_acc={val_acc:.4f}"
            + ("  *" if improved else "")
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    val_loss, val_acc, pred, lab = eval_model(model, val_loader, crit)
    torch.save(model.state_dict(), out / "best_model.pt")
    np.savez(
        out / "predictions_val.npz",
        pred=pred,
        label=lab,
        image=va_full[0],
        tokens=va_full[1],
        direction=va_full[2],
    )
    (out / "vocab.json").write_text(
        json.dumps({"vocab": VOCAB, "pad_idx": PAD_IDX, "max_tokens": MAX_TOKENS}, ensure_ascii=False)
    )

    result_block: dict = {
        "best_val_acc": round(best_acc, 6),
        "best_epoch": best_epoch,
        "train_seconds": round(time.time() - t0, 2),
    }
    if use_film and hasattr(model, "film_stats"):
        result_block["film_stats_at_best"] = model.film_stats(torch.from_numpy(va_full[1]).long())

    cfg = {
        "branch": "b4_curriculum",
        "variant": args.variant,
        "run": str(out),
        "git_commit": git_commit(),
        "env_id": ENV_ID,
        "seed": args.seed,
        "arch": {
            "model_type": "film" if use_film else "bcnet",
            "variant": args.variant,
            "img_mode": "embed",
            "obj_emb_dim": args.obj_emb_dim,
            "lang_dim": args.lang_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "n_params": n_params(model),
            "use_film": use_film,
            "use_concat": True,
        },
        "curriculum": cmeta,
        "optim": {
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "film_l2": args.film_l2 if use_film else 0.0,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "step_matched_sampler": True,
            "early_stop": "none（跑满 30 epoch，仅以 S6N3 val_acc 事后选优）",
        },
        "data": {
            "data_dir": args.data_dir,
            "n_train_episodes": len(pool_eps),
            "n_train_transitions": n_full_trans,
            "n_val_episodes": len(val_eps),
            "n_val_transitions": int(len(va_full[3])),
            "val_env_id": ENV_ID,
        },
        "result": result_block,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    print(
        f"{out}: best val_acc={best_acc:.4f} @ ep{best_epoch} "
        f"({n_params(model)} params, {cfg['result']['train_seconds']}s)"
    )


if __name__ == "__main__":
    main()
