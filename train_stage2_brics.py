# train_stage2_brics.py (FINAL: pair sampling to fight NONE dominance)
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import hashlib
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage2_brics_dataset import Stage2BRICSDataset, NONE_CLASS
from brics_fragment_library import BricsFragVocab
from stage2_brics_model import Stage2BRICSModel


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_pairs(
    logits: torch.Tensor,  # [B,N,N,C]
    y: torch.Tensor,       # [B,N,N]
    mask: torch.Tensor,    # [B,N]
    pos_frac: float,
    max_pairs: int,
    device: torch.device,
):
    """
    Build a sampled training set of (logits2, y2) from valid off-diagonal pairs.
    - pos = y != NONE_CLASS
    - neg = y == NONE_CLASS
    - sample to keep pos fraction around pos_frac
    - cap total sampled pairs to max_pairs (per batch)
    """
    B, N, _, C = logits.shape

    node_ok = (mask > 0.5)                        # [B,N]
    pair_ok = node_ok[:, :, None] & node_ok[:, None, :]  # [B,N,N]
    diag = torch.eye(N, device=device, dtype=torch.bool)[None, :, :]
    pair_ok = pair_ok & (~diag)

    # valid positions
    y_valid = y[pair_ok].long()          # [M]
    logits_valid = logits[pair_ok]       # [M,C]

    if y_valid.numel() == 0:
        return None, None, {"M": 0, "pos": 0, "neg": 0, "keep": 0}

    pos_mask = (y_valid != NONE_CLASS)
    neg_mask = (y_valid == NONE_CLASS)

    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    n_pos = int(pos_idx.numel())
    n_neg = int(neg_idx.numel())
    M = int(y_valid.numel())

    # If no positives, we can still train on some negatives (but this batch is not informative)
    if n_pos == 0:
        keep = min(M, max_pairs)
        sel = torch.randperm(M, device=device)[:keep]
        return logits_valid[sel], y_valid[sel], {"M": M, "pos": 0, "neg": n_neg, "keep": keep}

    # Determine how many total to keep so that pos/(pos+neg) ~ pos_frac
    # keep_pos = min(n_pos, something), keep_neg = keep_pos*(1-pos_frac)/pos_frac
    # Also cap by max_pairs
    keep_pos = n_pos
    keep_neg = int(round(keep_pos * (1.0 - pos_frac) / max(pos_frac, 1e-6)))

    # If there are not enough negatives, just take all negatives
    keep_neg = min(keep_neg, n_neg)

    keep_total = keep_pos + keep_neg
    if keep_total > max_pairs:
        # scale down proportionally, but always keep at least 1 positive
        scale = max_pairs / float(keep_total)
        keep_pos2 = max(1, int(round(keep_pos * scale)))
        keep_neg2 = max(0, max_pairs - keep_pos2)
        keep_neg2 = min(keep_neg2, n_neg)
        keep_pos, keep_neg = keep_pos2, keep_neg2

    # Sample indices
    # positives: random subset if needed
    if keep_pos < n_pos:
        sel_pos = pos_idx[torch.randperm(n_pos, device=device)[:keep_pos]]
    else:
        sel_pos = pos_idx

    if keep_neg > 0:
        sel_neg = neg_idx[torch.randperm(n_neg, device=device)[:keep_neg]]
        sel = torch.cat([sel_pos, sel_neg], dim=0)
    else:
        sel = sel_pos

    # shuffle selected
    sel = sel[torch.randperm(sel.numel(), device=device)]

    return logits_valid[sel], y_valid[sel], {"M": M, "pos": n_pos, "neg": n_neg, "keep": int(sel.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_split.parquet")
    ap.add_argument("--val", default="val_split.parquet")
    ap.add_argument("--vocab", default="brics_vocab.tsv")
    ap.add_argument("--ir_len", type=int, default=1024)
    ap.add_argument("--max_nodes", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="best_stage2_brics.pt")

    # imbalance controls
    ap.add_argument("--none_weight", type=float, default=0.001,
                    help="cross-entropy class weight for NONE_CLASS (smaller => punish predicting NONE)")
    ap.add_argument("--pos_frac", type=float, default=0.35,
                    help="target fraction of positive pairs (y!=NONE) in sampled training pairs")
    ap.add_argument("--max_pairs", type=int, default=200000,
                    help="cap sampled pairs per batch after masking/offdiag (keeps memory stable)")

    # regular training utils
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--clip", type=float, default=2.0)

    # debug
    ap.add_argument("--debug_one_batch", action="store_true",
                    help="print one batch sampling stats then exit")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    vocab = BricsFragVocab(args.vocab)
    vocab_sha1 = sha1_file(args.vocab)

    ds_tr = Stage2BRICSDataset(args.train, args.vocab, ir_len=args.ir_len, max_nodes=args.max_nodes)
    ds_va = Stage2BRICSDataset(args.val, args.vocab, ir_len=args.ir_len, max_nodes=args.max_nodes)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False)

    model = Stage2BRICSModel(x_dim=args.ir_len, vocab_size=len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # class weights: downweight NONE
    w = torch.ones((257,), dtype=torch.float32, device=device)
    w[NONE_CLASS] = float(args.none_weight)

    best = float("inf")
    bad = 0

    for ep in range(1, args.epochs + 1):
        # -------- train --------
        model.train()
        tr_loss = 0.0
        n_tr = 0

        pbar = tqdm(dl_tr, desc=f"Epoch {ep:03d} train", leave=False)
        for step, (xb, frag_ids, mask, y) in enumerate(pbar):
            xb = xb.to(device)
            frag_ids = frag_ids.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(xb, frag_ids, mask)  # [B,N,N,257]

            logits2, y2, info = sample_pairs(
                logits=logits,
                y=y,
                mask=mask,
                pos_frac=float(args.pos_frac),
                max_pairs=int(args.max_pairs),
                device=device,
            )

            if logits2 is None:
                continue

            if args.debug_one_batch and ep == 1 and step == 0:
                print("xb:", tuple(xb.shape), xb.mean().item(), xb.std().item(),
                      xb.min().item(), xb.max().item())
                print("pair stats:", info)
                none_ratio = float((y2 == NONE_CLASS).float().mean().item())
                print("sampled NONE ratio:", none_ratio)
                uniq = int(torch.unique(y2).numel())
                print("sampled unique classes:", uniq)
                raise SystemExit

            loss = F.cross_entropy(logits2, y2, weight=w)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            tr_loss += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)

        tr_loss /= max(1, n_tr)

        # -------- val (NO sampling;真实分布) --------
        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, frag_ids, mask, y in tqdm(dl_va, desc=f"Epoch {ep:03d} val", leave=False):
                xb = xb.to(device)
                frag_ids = frag_ids.to(device)
                mask = mask.to(device)
                y = y.to(device)

                logits = model(xb, frag_ids, mask)  # [B,N,N,257]
                B, N, _, C = logits.shape

                node_ok = (mask > 0.5)
                pair_ok = node_ok[:, :, None] & node_ok[:, None, :]
                diag = torch.eye(N, device=device, dtype=torch.bool)[None, :, :]
                pair_ok = pair_ok & (~diag)

                logits_v = logits[pair_ok]          # [M,C]
                y_v = y[pair_ok].long()             # [M]
                if y_v.numel() == 0:
                    continue

                loss = F.cross_entropy(logits_v, y_v, weight=w)
                va_loss += float(loss.item()) * xb.size(0)
                n_va += xb.size(0)

        va_loss /= max(1, n_va)
        print(f"Epoch {ep:03d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}")

        # -------- early stop + save --------
        if va_loss < best:
            best = va_loss
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "ir_len": args.ir_len,
                    "max_nodes": args.max_nodes,
                    "vocab_path": args.vocab,
                    "vocab_sha1": vocab_sha1,
                    "seed": args.seed,
                    "none_weight": args.none_weight,
                    "pos_frac": args.pos_frac,
                    "max_pairs": args.max_pairs,
                },
                args.out,
            )
            print("  saved", args.out)
            print(f"[SAVE] ep={ep} va_loss={va_loss:.4f} best={best:.4f} -> {args.out}")
        else:
            bad += 1
            print(f"[NO-SAVE] ep={ep} va_loss={va_loss:.4f} best={best:.4f} bad={bad}/{args.patience}")
            if bad >= args.patience:
                print(f"Early stop at epoch {ep} (patience={args.patience})")
                break


if __name__ == "__main__":
    main()