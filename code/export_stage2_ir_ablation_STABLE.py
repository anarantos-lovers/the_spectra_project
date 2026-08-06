import argparse
import torch
import numpy as np
from tqdm import tqdm

from stage2_brics_dataset import Stage2BRICSDataset
from stage2_brics_model import Stage2BRICSModel


# =========================
# IR ABALTION
# =========================
def ablate_ir(x, mode):
    if mode == "none":
        return x
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "noise":
        return torch.randn_like(x)
    if mode == "shuffle_dim":
        idx = torch.randperm(x.shape[-1])
        return x[idx]
    return x


# =========================
# SAFE VOCAB (CRITICAL FIX)
# =========================
def get_vocab_list(vocab_obj, size):
    # method style
    if hasattr(vocab_obj, "id2smiles"):
        fn = vocab_obj.id2smiles
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, (list, tuple)):
                    return [str(x) for x in out]
                return [str(fn(i)) for i in range(size)]
            except:
                return [str(fn(i)) for i in range(size)]

    if hasattr(vocab_obj, "smiles2id"):
        fn = vocab_obj.smiles2id
        if callable(fn):
            try:
                return list(fn().keys())
            except:
                return []

    return ["UNK"] * size


# =========================
# CKPT LOADER
# =========================
def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


# =========================
# SAFE FORWARD
# =========================
def forward_safe(model, ir, frag_ids, mask):
    if ir.dim() == 1:
        ir = ir.unsqueeze(0)
    if frag_ids.dim() == 1:
        frag_ids = frag_ids.unsqueeze(0)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    return model(xb=ir, frag_ids=frag_ids, mask=mask)


# =========================
# MAIN
# =========================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ir_len", type=int, default=1024)
    parser.add_argument("--max_nodes", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--ir_ablation", default="none")
    args = parser.parse_args()

    device = torch.device(args.device)

    # =========================
    # DATASET
    # =========================
    dataset = Stage2BRICSDataset(
        parquet_path=args.data,
        vocab_tsv=args.vocab,
        ir_len=args.ir_len,
        max_nodes=args.max_nodes
    )

    vocab_size = len(dataset.vocab)

    # =========================
    # MODEL
    # =========================
    model = Stage2BRICSModel(
        x_dim=args.ir_len,
        vocab_size=vocab_size
    ).to(device)

    model.load_state_dict(load_ckpt(args.ckpt))
    model.eval()

    # =========================
    # VOCAB (CRITICAL FIX)
    # =========================
    id2s = get_vocab_list(dataset.vocab, vocab_size)

    # =========================
    # STORAGE
    # =========================
    logits_all = []
    topk_vals_all = []
    topk_idx_all = []
    frag_ids_all = []
    mask_all = []
    y_all = []

    # =========================
    # LOOP
    # =========================
    for i in tqdm(range(len(dataset))):

        ir, frag_ids, mask, y = dataset[i]

        ir = ir.to(device)
        frag_ids = frag_ids.to(device)
        mask = mask.to(device)

        ir = ablate_ir(ir, args.ir_ablation)

        with torch.no_grad():
            logits = forward_safe(model, ir, frag_ids, mask)

            topk_vals, topk_idx = torch.topk(
                logits,
                k=args.topk,
                dim=-1
            )

        logits_all.append(logits.squeeze(0).cpu().numpy())
        topk_vals_all.append(topk_vals.squeeze(0).cpu().numpy())
        topk_idx_all.append(topk_idx.squeeze(0).cpu().numpy())
        frag_ids_all.append(frag_ids.cpu().numpy())
        mask_all.append(mask.cpu().numpy())
        y_all.append(y.numpy())

    # =========================
    # SAVE (FINAL NPZ PROTOCOL)
    # =========================
    np.savez_compressed(
        args.out,
        logits=np.array(logits_all),
        topk_vals=np.array(topk_vals_all),
        topk_idx=np.array(topk_idx_all),
        frag_ids=np.array(frag_ids_all),
        mask=np.array(mask_all),
        y=np.array(y_all),

        # 🔥 CRITICAL FOR DECODE
        vocab_id2s=np.array(id2s, dtype=object)
    )

    print("\n[OK] EXPORT DONE:", args.out)


if __name__ == "__main__":
    main()