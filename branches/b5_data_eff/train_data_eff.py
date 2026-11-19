"""支线 5 · 数据效率训练脚本（步数匹配 + D4 群对称增广 + 正则消融 + FiLM）。

用法:
    OMP_NUM_THREADS=1 python branches/b5_data_eff/train_data_eff.py \
        --method bc_d4 --frac 0.2 --seed 0 --out runs/b5_bc_d4_f020_seed0
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
from torch.utils.data import DataLoader, RandomSampler

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b5_data_eff"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    ENV_ID,
    MAX_TOKENS,
    N_ACTIONS,
    PAD_IDX,
    VOCAB,
    BCNet,
    set_all_seeds,
)
from data_eff_utils import (  # noqa: E402
    METHODS,
    N_REF_TRANSITIONS,
    augment_arrays,
    frac_tag,
    load_subset_episodes,
)
from film_net import FiLMBCNet  # noqa: E402
from lang_encoders import LangBCNet  # noqa: E402
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


def build_model(mcfg: dict, dropout: float) -> nn.Module:
    arch = mcfg["arch"]
    if arch == "bcnet":
        return BCNet(
            vocab_size=len(VOCAB),
            obj_emb_dim=8,
            lang_dim=32,
            hidden=128,
            dropout=dropout,
            img_mode="embed",
        )
    if arch == "film":
        return FiLMBCNet(
            variant="film",
            vocab_size=len(VOCAB),
            obj_emb_dim=8,
            lang_dim=32,
            hidden=128,
            dropout=dropout,
        )
    if arch == "gru_small":
        return LangBCNet(
            variant="gru_small",
            vocab_size=len(VOCAB),
            obj_emb_dim=8,
            lang_dim=32,
            hidden=128,
            dropout=dropout,
        )
    raise ValueError(f"未知 arch={arch}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", default="data/expert_5000.pkl")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    mcfg = METHODS[args.method]
    ftag = frac_tag(args.frac)
    if not args.out:
        args.out = f"runs/b5_{args.method}_{ftag}_seed{args.seed}"

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    train_eps, val_eps = load_subset_episodes(args.data, args.frac, args.seed)
    tr_raw = build_dataset(train_eps)
    va_raw = build_dataset(val_eps)
    n_raw_tr = int(len(tr_raw[3]))

    tr_aug = augment_arrays(mcfg["aug"], *tr_raw)
    n_aug_tr = int(len(tr_aug[3]))
    train_ds = make_ds(*tr_aug)
    val_loader = DataLoader(make_ds(*va_raw), batch_size=512, shuffle=False, num_workers=0)

    if not mcfg["step_match"]:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0
        )
    elif n_aug_tr == N_REF_TRANSITIONS:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0
        )
    else:
        sampler = RandomSampler(
            train_ds,
            replacement=(n_aug_tr < N_REF_TRANSITIONS),
            num_samples=N_REF_TRANSITIONS,
            generator=g,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0
        )

    model = build_model(mcfg, dropout=mcfg["dropout"])

    class_weight = None
    if mcfg["act_balance"]:
        counts = np.bincount(tr_aug[3], minlength=N_ACTIONS).astype(np.float64)
        w = np.ones(N_ACTIONS, dtype=np.float64)
        seen = counts > 0
        inv_sqrt = np.where(seen, 1.0 / np.sqrt(np.maximum(counts, 1.0)), 0.0)
        if seen.any():
            w[seen] = inv_sqrt[seen] / inv_sqrt[seen].mean()
        class_weight = torch.tensor(w, dtype=torch.float32)

    crit = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=mcfg["label_smoothing"])
    eval_crit = nn.CrossEntropyLoss()  # 验证集用无权重标准交叉熵对齐口径
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=mcfg["weight_decay"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
        if mcfg["cosine_lr"]
        else None
    )

    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, -1, None
    best_train_acc = 0.0
    total_grad_steps = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_n, run_c = 0.0, 0, 0
        cur_lr = float(opt.param_groups[0]["lr"])
        for img, tok, d, act in train_loader:
            logits = model(img, tok, d)
            loss = crit(logits, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_grad_steps += 1
            run_loss += loss.item() * act.size(0)
            run_n += act.size(0)
            run_c += (logits.argmax(-1) == act).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_acc = run_c / max(1, run_n)
        val_loss, val_acc, _p, _l = eval_model(model, val_loader, eval_crit)
        history.append(
            {
                "epoch": epoch,
                "lr": round(cur_lr, 6),
                "epoch_transitions": run_n,
                "train_loss": round(run_loss / max(1, run_n), 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )
        improved = val_acc > best_acc + 1e-9
        if improved:
            best_acc, best_epoch = val_acc, epoch
            best_train_acc = train_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(
            f"[b5 {args.method} {ftag} s{args.seed}] ep {epoch:2d} "
            f"tr_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            + ("  *" if improved else "")
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    val_loss, val_acc, pred, lab = eval_model(model, val_loader, eval_crit)
    torch.save(model.state_dict(), out / "best_model.pt")
    np.savez(
        out / "predictions_val.npz",
        pred=pred,
        label=lab,
        image=va_raw[0],
        tokens=va_raw[1],
        direction=va_raw[2],
    )
    (out / "vocab.json").write_text(
        json.dumps({"vocab": VOCAB, "pad_idx": PAD_IDX, "max_tokens": MAX_TOKENS}, ensure_ascii=False)
    )

    cfg = {
        "branch": "b5_data_eff",
        "method": args.method,
        "frac": args.frac,
        "frac_tag": ftag,
        "run": str(out),
        "git_commit": git_commit(),
        "env_id": ENV_ID,
        "seed": args.seed,
        "arch": {
            "arch_type": mcfg["arch"],
            "variant": mcfg["arch"] if mcfg["arch"] != "film" else "film",
            "obj_emb_dim": 8,
            "lang_dim": 32,
            "hidden": 128,
            "dropout": mcfg["dropout"],
            "n_params": n_params(model),
            "use_film": mcfg["arch"] == "film",
        },
        "method_cfg": mcfg,
        "optim": {
            "optimizer": "Adam",
            "lr": args.lr,
            "cosine_lr": mcfg["cosine_lr"],
            "weight_decay": mcfg["weight_decay"],
            "label_smoothing": mcfg["label_smoothing"],
            "act_balance": mcfg["act_balance"],
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "step_match": mcfg["step_match"],
            "total_grad_steps": total_grad_steps,
            "early_stop": "none（跑满 30 epoch，仅按 val_acc 事后选优）",
        },
        "data": {
            "path": args.data,
            "frac": args.frac,
            "n_train_episodes": len(train_eps),
            "n_raw_transitions": n_raw_tr,
            "n_aug_transitions": n_aug_tr,
            "n_val_episodes": len(val_eps),
            "n_val_transitions": int(len(va_raw[3])),
        },
        "result": {
            "best_val_acc": round(best_acc, 6),
            "train_acc_at_best": round(best_train_acc, 6),
            "generalization_gap": round(best_train_acc - best_acc, 6),
            "best_epoch": best_epoch,
            "train_seconds": round(time.time() - t0, 2),
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    print(
        f"{out}: val_acc={best_acc:.4f} (tr={best_train_acc:.4f}, gap={best_train_acc - best_acc:+.4f}) "
        f"@ ep{best_epoch} ({total_grad_steps} steps, {cfg['result']['train_seconds']}s)"
    )


if __name__ == "__main__":
    main()
