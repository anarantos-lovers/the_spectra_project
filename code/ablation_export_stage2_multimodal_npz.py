from __future__ import annotations

import argparse
import hashlib
import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ablation_stage2_multimodal_dataset import AblationStage2MultimodalDataset
from brics_fragment_library import BricsFragVocab
from ablation_stage2_multimodal_model import AblationStage2MultimodalModel


DUMMY16 = "".join([f"[{i}*]" for i in range(1, 17)])


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_id2s_list(vocab_obj, vocab_size: int) -> list[str]:
    for fn_name in ("id2s", "i2s", "id2str"):
        if not hasattr(vocab_obj, fn_name):
            continue
        attr = getattr(vocab_obj, fn_name)

        if callable(attr):
            out: list[str] = []
            ok = True
            for i in range(vocab_size):
                try:
                    s = attr(i)
                except TypeError:
                    ok = False
                    break
                out.append("" if s is None else str(s))
            if ok:
                return out

        cand = attr
        if isinstance(cand, dict):
            id2s_list = [""] * vocab_size
            for k, v in cand.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                if 0 <= idx < vocab_size:
                    id2s_list[idx] = "" if v is None else str(v)
            return id2s_list

        if isinstance(cand, (list, tuple, np.ndarray)):
            cand = list(cand)
            if len(cand) < vocab_size:
                cand = cand + [""] * (vocab_size - len(cand))
            return [("" if x is None else str(x)) for x in cand[:vocab_size]]

    raise RuntimeError("Cannot build id2s list from vocab.")


def apply_dummy16_to_vocab(id2s_list: list[str]) -> list[str]:
    out = []
    invalid_tokens = {"", "<UNK>", "[UNK]", "UNK", "<PAD>", "[PAD]", "PAD", "None", "nan"}

    for s in id2s_list:
        s = "" if s is None else str(s).strip()
        if s in invalid_tokens:
            out.append(DUMMY16)
        else:
            out.append(s)

    if len(out) > 0:
        out[0] = DUMMY16
    if len(out) > 1:
        out[1] = DUMMY16

    return out


def tmp_paths(tmp_dir: str):
    os.makedirs(tmp_dir, exist_ok=True)
    return {
        "topv": os.path.join(tmp_dir, "tmp_topv.dat"),
        "topi": os.path.join(tmp_dir, "tmp_topi.dat"),
        "frag": os.path.join(tmp_dir, "tmp_frag.dat"),
        "mask": os.path.join(tmp_dir, "tmp_mask.dat"),
    }


def cleanup_tmp(tmp_dir: str):
    paths = tmp_paths(tmp_dir)
    for p in paths.values():
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                print("[WARN] cannot remove:", p, "->", e)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--vocab", default=None)
    ap.add_argument("--max_nodes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--tmp_dir", default="tmp_export_ablation")
    ap.add_argument("--cleanup_tmp", action="store_true")
    ap.add_argument("--use_dummy16", action="store_true")

    args = ap.parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("ckpt must be dict with key 'model'")

    vocab_path = ckpt.get("vocab_path") or ckpt.get("vocab")
    if vocab_path is None:
        raise KeyError("ckpt missing vocab_path/vocab")

    ckpt_sha = ckpt.get("vocab_sha1", None)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"ckpt vocab_path not found: {vocab_path}")

    file_sha = sha1_file(vocab_path)

    if ckpt_sha is not None and file_sha != ckpt_sha:
        raise SystemExit("FATAL: ckpt vocab_sha1 != sha1(vocab_path)")

    if args.vocab is not None:
        if not os.path.exists(args.vocab):
            raise FileNotFoundError(f"--vocab not found: {args.vocab}")
        user_sha = sha1_file(args.vocab)
        if ckpt_sha is not None and user_sha != ckpt_sha:
            raise SystemExit("FATAL: user vocab sha1 != ckpt vocab_sha1")
        if user_sha != file_sha:
            raise SystemExit("FATAL: user vocab sha1 != ckpt vocab_path sha1")

    ir_len = int(ckpt.get("ir_len", 1024))
    max_nodes = int(args.max_nodes) if args.max_nodes is not None else int(ckpt.get("max_nodes", 64))

    use_h1 = bool(ckpt.get("use_h1", False))
    use_c13 = bool(ckpt.get("use_c13", False))
    use_hsqc = bool(ckpt.get("use_hsqc", False))
    h1_dim = int(ckpt.get("h1_dim", 128))
    c13_dim = int(ckpt.get("c13_dim", 128))
    hsqc_h_dim = int(ckpt.get("hsqc_h_dim", 16))
    hsqc_c_dim = int(ckpt.get("hsqc_c_dim", 16))

    emb_dim = int(ckpt.get("emb_dim", 128))
    hidden = int(ckpt.get("hidden", 256))
    dropout = float(ckpt.get("dropout", 0.1))

    vocab = BricsFragVocab(vocab_path)
    id2s_list = build_id2s_list(vocab, len(vocab))

    if args.use_dummy16:
        id2s_list = apply_dummy16_to_vocab(id2s_list)

    ds = AblationStage2MultimodalDataset(
        args.data,
        vocab_path,
        ir_len=ir_len,
        max_nodes=max_nodes,
        use_h1=use_h1,
        use_c13=use_c13,
        use_hsqc=use_hsqc,
        h1_dim=h1_dim,
        c13_dim=c13_dim,
        hsqc_h_dim=hsqc_h_dim,
        hsqc_c_dim=hsqc_c_dim,
    )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = AblationStage2MultimodalModel(
        x_dim=ds.x_dim,
        vocab_size=len(vocab),
        emb_dim=emb_dim,
        hidden=hidden,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    total = len(ds)
    xb0, frag0, mask0, _ = next(iter(dl))
    B0, N = frag0.shape
    C = 257
    K = int(args.topk)

    if K <= 0 or K > C:
        raise ValueError(f"--topk must be in [1, {C}]")

    topv_dtype = np.float16 if args.dtype == "float16" else np.float32
    mask_dtype = np.float16

    paths = tmp_paths(args.tmp_dir)
    topv_out = np.memmap(paths["topv"], mode="w+", dtype=topv_dtype, shape=(total, N, N, K))
    topi_out = np.memmap(paths["topi"], mode="w+", dtype=np.uint16, shape=(total, N, N, K))
    frag_ids_out = np.memmap(paths["frag"], mode="w+", dtype=np.int64, shape=(total, N))
    mask_out = np.memmap(paths["mask"], mode="w+", dtype=mask_dtype, shape=(total, N))

    idx = 0
    for xb, frag_ids, mask, _y in tqdm(dl, desc="Export"):
        xb = xb.to(device)
        frag_ids = frag_ids.to(device)
        mask = mask.to(device)

        logits = model(xb, frag_ids, mask)
        topv, topi = torch.topk(logits, k=K, dim=-1)

        if args.dtype == "float16":
            topv = topv.to(torch.float16)

        bsz = logits.shape[0]
        topv_out[idx:idx + bsz] = topv.detach().cpu().numpy()
        topi_out[idx:idx + bsz] = topi.detach().cpu().numpy().astype(np.uint16)
        frag_ids_out[idx:idx + bsz] = frag_ids.detach().cpu().numpy()
        mask_out[idx:idx + bsz] = mask.detach().cpu().numpy().astype(mask_dtype)

        idx += bsz

    topv_out.flush()
    topi_out.flush()
    frag_ids_out.flush()
    mask_out.flush()

    topv_arr = np.array(topv_out)
    topi_arr = np.array(topi_out)
    frag_arr = np.array(frag_ids_out)
    mask_arr = np.array(mask_out)

    np.savez_compressed(
        args.out,
        topv=topv_arr,
        topi=topi_arr,
        frag_ids=frag_arr,
        mask=mask_arr,
        vocab_id2s=np.array(id2s_list, dtype=object),
        vocab_sha1=file_sha,
        ir_len=ir_len,
        max_nodes=max_nodes,
        topk=K,
        dtype=args.dtype,
        use_dummy16=bool(args.use_dummy16),
        use_h1=use_h1,
        use_c13=use_c13,
        use_hsqc=use_hsqc,
        h1_dim=h1_dim,
        c13_dim=c13_dim,
        hsqc_h_dim=hsqc_h_dim,
        hsqc_c_dim=hsqc_c_dim,
        x_dim=ds.x_dim,
    )

    print("Saved:", args.out)

    del topv_arr, topi_arr, frag_arr, mask_arr
    del topv_out, topi_out, frag_ids_out, mask_out
    gc.collect()

    if args.cleanup_tmp:
        cleanup_tmp(args.tmp_dir)


if __name__ == "__main__":
    main()