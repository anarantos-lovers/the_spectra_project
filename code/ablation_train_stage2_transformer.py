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
from ablation_transformer_stage2_model import AblationTransformerStage2Model


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
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    pos_frac: float,
    max_pairs: int,
    device: torch.device,
):
    B, N, _, C = logits.shape

    node_ok = (mask > 0.5)
    pair_ok = node_ok[:, :, None] & node_ok[:, None, :]
    diag = torch.eye(N, device=device, dtype=torch.bool)[None, :, :]
    pair_ok = pair_ok & (~diag)

    y_valid = y[pair_ok].long()
    logits_valid = logits[pair_ok]

    if y_valid.numel() == 0:
        return None, None, {"M": 0, "pos": 0, "neg": 0, "keep": 0}

    pos_mask = (y_valid != NONE_CLASS)
    neg_mask = (y_valid == NONE_CLASS)

    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    n_pos = int(pos_idx.numel())
    n_neg = int(neg_idx.numel())
    M = int(y_valid.numel())

    if n_pos == 0:
        keep = min(M, max_pairs)
        sel = torch.randperm(M, device=device)[:keep]
        return logits_valid[sel], y_valid[sel], {"M": M, "pos": 0, "neg": n_neg, "keep": keep}

    keep_pos = n_pos
    keep_neg = int(round(keep_pos * (1.0 - pos_frac) / max(pos_frac, 1e-6)))
    keep_neg = min(keep_neg, n_neg)

    keep_total = keep_pos + keep_neg
    if keep_total > max_pairs:
        scale = max_pairs / float(keep_total)
        keep_pos2 = max(1, int(round(keep_pos * scale)))
        keep_neg2 = max(0, max_pairs - keep_pos2)
        keep_neg2 = min(keep_neg2, n_neg)
        keep_pos, keep_neg = keep_pos2, keep_neg2

    if keep_pos < n_pos:
        sel_pos = pos_idx[torch.randperm(n_pos, device=device)[:keep_pos]]
    else:
        sel_pos = pos_idx

    if keep_neg > 0:
        sel_neg = neg_idx[torch.randperm(n_neg, device=device)[:keep_neg]]
        sel = torch.cat([sel_pos, sel_neg], dim=0)
    else:
        sel = sel_pos

    sel = sel[torch.randperm(sel.numel(), device=device)]
    return logits_valid[sel], y_valid[sel], {"M": M, "pos": n_pos, "neg": n_neg, "keep": int(sel.numel())}


def safe_save_checkpoint(save_obj, out_path: str):
    tmp_out = out_path + ".tmp"
    if os.path.exists(tmp_out):
        try:
            os.remove(tmp_out)
        except Exception:
            pass
    torch.save(save_obj, tmp_out)
    os.replace(tmp_out, out_path)


def main():
    ap = argparse.ArgumentParser()

    # 和你现在补充实验完全对齐
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--ir_len", type=int, default=1024)
    ap.add_argument("--max_nodes", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)

    ap.add_argument("--none_weight", type=float, default=0.0005)
    ap.add_argument("--pos_frac", type=float, default=0.35)
    ap.add_argument("--max_pairs", type=int, default=200000)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--clip", type=float, default=2.0)

    # 结构参数
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--ff_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--debug_one_batch", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    vocab = BricsFragVocab(args.vocab)
    vocab_sha1 = sha1_file(args.vocab)

    ds_tr = Stage2BRICSDataset(args.train, args.vocab, ir_len=args.ir_len, max_nodes=args.max_nodes)
    ds_va = Stage2BRICSDataset(args.val, args.vocab, ir_len=args.ir_len, max_nodes=args.max_nodes)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False)

    model = AblationTransformerStage2Model(
        x_dim=args.ir_len,
        vocab_size=len(vocab),
        max_nodes=args.max_nodes,
        hidden=args.hidden,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    w = torch.ones((257,), dtype=torch.float32, device=device)
    w[NONE_CLASS] = float(args.none_weight)

    best = float("inf")
    bad = 0

    print("[INFO] Transformer ablation aligned setting")
    print("[INFO] hidden/nhead/layers:", args.hidden, args.nhead, args.num_layers)

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0

        pbar = tqdm(dl_tr, desc=f"Epoch {ep:03d} train", leave=False)
        for step, (xb, frag_ids, mask, y) in enumerate(pbar):
            xb = xb.to(device)
            frag_ids = frag_ids.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(xb, frag_ids, mask)

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
                raise SystemExit

            loss = F.cross_entropy(logits2, y2, weight=w)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            tr_loss += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)

        tr_loss /= max(1, n_tr)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, frag_ids, mask, y in tqdm(dl_va, desc=f"Epoch {ep:03d} val", leave=False):
                xb = xb.to(device)
                frag_ids = frag_ids.to(device)
                mask = mask.to(device)
                y = y.to(device)

                logits = model(xb, frag_ids, mask)
                B, N, _, C = logits.shape

                node_ok = (mask > 0.5)
                pair_ok = node_ok[:, :, None] & node_ok[:, None, :]
                diag = torch.eye(N, device=device, dtype=torch.bool)[None, :, :]
                pair_ok = pair_ok & (~diag)

                logits_v = logits[pair_ok]
                y_v = y[pair_ok].long()
                if y_v.numel() == 0:
                    continue

                loss = F.cross_entropy(logits_v, y_v, weight=w)
                va_loss += float(loss.item()) * xb.size(0)
                n_va += xb.size(0)

        va_loss /= max(1, n_va)
        print(f"Epoch {ep:03d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}")

        if va_loss < best:
            best = va_loss
            bad = 0
            save_obj = {
                "model": model.state_dict(),
                "ir_len": args.ir_len,
                "max_nodes": args.max_nodes,
                "vocab_path": args.vocab,
                "vocab_sha1": vocab_sha1,
                "hidden": args.hidden,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "ff_dim": args.ff_dim,
                "dropout": args.dropout,
                "seed": args.seed,
                "none_weight": args.none_weight,
                "pos_frac": args.pos_frac,
                "max_pairs": args.max_pairs,
            }
            try:
                safe_save_checkpoint(save_obj, args.out)
            except Exception as e:
                print(f"[WARN] save failed: {e}")
            else:
                print(f"[SAVE] best -> {args.out} (val={va_loss:.4f})")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY STOP] no improvement for {args.patience} epochs.")
                break


if __name__ == "__main__":
    main()