from __future__ import annotations

import argparse
import gc
import hashlib
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from stage2_brics_dataset import Stage2BRICSDataset
from brics_fragment_library import BricsFragVocab
from stage2_brics_model import Stage2BRICSModel


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
            return ["" if x is None else str(x) for x in cand[:vocab_size]]

    raise RuntimeError("Cannot build id2s list from vocab.")


def normalize_vocab_strings(id2s_list: list[str]) -> list[str]:
    out = []
    for s in id2s_list:
        if s is None:
            out.append("")
        else:
            out.append(str(s).strip())
    return out


def build_s2id(id2s_list: List[str]) -> Dict[str, int]:
    s2id: Dict[str, int] = {}
    for i, s in enumerate(id2s_list):
        if s and s not in s2id:
            s2id[s] = i
    return s2id


def canonical_fragment_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, isomericSmiles=True)


def brics_fragments_from_smiles(smiles: str) -> List[str]:
    """
    用真实 SMILES 做 BRICS 分解，得到 attachment-labeled fragment SMILES。
    这是 oracle fragment experiment 的核心。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        broken = BRICS.BreakBRICSBonds(mol)
        frags = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
        out = [canonical_fragment_smiles(f) for f in frags]
        out = [s for s in out if s]
        # 为了稳定性做排序，避免顺序漂移
        out = sorted(out)
        return out
    except Exception:
        return []


def smiles_col_from_df(df: pd.DataFrame) -> str:
    if "smiles_can" in df.columns:
        return "smiles_can"
    if "smiles" in df.columns:
        return "smiles"
    raise KeyError(f"parquet missing smiles/smiles_can. cols={list(df.columns)[:30]}")


def build_oracle_frag_ids_and_mask(
    smiles_list: List[str],
    s2id: Dict[str, int],
    max_nodes: int,
    unk_id: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    返回:
      frag_ids: [B, N]
      mask:     [B, N]
      stats
    """
    B = len(smiles_list)
    frag_ids = np.zeros((B, max_nodes), dtype=np.int64)
    mask = np.zeros((B, max_nodes), dtype=np.float32)

    stats = {
        "total_samples": B,
        "parse_fail_smiles": 0,
        "empty_brics": 0,
        "truncated_samples": 0,
        "total_frags": 0,
        "unknown_fragments": 0,
        "samples_with_unknown": 0,
    }

    for b, smi in enumerate(smiles_list):
        frags = brics_fragments_from_smiles(smi)
        if not frags:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                stats["parse_fail_smiles"] += 1
                continue
            stats["empty_brics"] += 1
            continue

        stats["total_frags"] += len(frags)

        if len(frags) > max_nodes:
            frags = frags[:max_nodes]
            stats["truncated_samples"] += 1

        unk_hit = False
        for j, fs in enumerate(frags):
            fid = s2id.get(fs, unk_id)
            if fid == unk_id and fs not in s2id:
                stats["unknown_fragments"] += 1
                unk_hit = True
            frag_ids[b, j] = int(fid)
            mask[b, j] = 1.0

        if unk_hit:
            stats["samples_with_unknown"] += 1

    return frag_ids, mask, stats


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

    ap.add_argument("--vocab", default=None, help="optional strict-check against ckpt vocab_sha1")
    ap.add_argument("--max_nodes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--tmp_dir", default="tmp_export_oracle")
    ap.add_argument("--cleanup_tmp", action="store_true")

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

    print("[CKPT]", args.ckpt)
    print("[CKPT vocab_path]", vocab_path)
    print("[CKPT vocab_sha1]", ckpt_sha)
    print("[FILE vocab_sha1]", file_sha)

    if ckpt_sha is not None and file_sha != ckpt_sha:
        raise SystemExit("FATAL: ckpt vocab_sha1 != sha1(vocab_path)")

    if args.vocab is not None:
        if not os.path.exists(args.vocab):
            raise FileNotFoundError(f"--vocab not found: {args.vocab}")
        user_sha = sha1_file(args.vocab)
        print("[USER vocab]", args.vocab)
        print("[USER vocab_sha1]", user_sha)
        if ckpt_sha is not None and user_sha != ckpt_sha:
            raise SystemExit("FATAL: user vocab sha1 != ckpt vocab_sha1")
        if user_sha != file_sha:
            raise SystemExit("FATAL: user vocab sha1 != ckpt vocab_path sha1")

    ir_len = int(ckpt.get("ir_len", 1024))
    max_nodes = int(args.max_nodes) if args.max_nodes is not None else int(ckpt.get("max_nodes", 64))

    vocab = BricsFragVocab(vocab_path)
    id2s_list = build_id2s_list(vocab, len(vocab))
    id2s_list = normalize_vocab_strings(id2s_list)
    s2id = build_s2id(id2s_list)

    print("[VOCAB] size =", len(id2s_list))
    print("[VOCAB] id2s[0] =", id2s_list[0] if len(id2s_list) > 0 else "")
    print("[VOCAB] id2s[1] =", id2s_list[1] if len(id2s_list) > 1 else "")

    model = Stage2BRICSModel(x_dim=ir_len, vocab_size=len(vocab))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    # 这里仍然借用现有 dataset 生成 xb（谱图特征），但 frag_ids / mask 将被 oracle 替换
    ds = Stage2BRICSDataset(args.data, vocab_path, ir_len=ir_len, max_nodes=max_nodes)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    df = pd.read_parquet(args.data)
    smiles_col = smiles_col_from_df(df)
    smiles_all = df[smiles_col].astype(str).tolist()

    total = len(ds)
    if len(smiles_all) != total:
        raise ValueError(f"parquet rows ({len(smiles_all)}) != dataset rows ({total})")

    print("Total samples:", total)
    print("SMILES column:", smiles_col)

    # infer output shape
    xb0, _frag0, _mask0, _ = next(iter(dl))
    xb0 = xb0.to(device)

    dummy_frag = torch.zeros((xb0.shape[0], max_nodes), dtype=torch.long, device=device)
    dummy_mask = torch.zeros((xb0.shape[0], max_nodes), dtype=torch.float32, device=device)
    logits0 = model(xb0, dummy_frag, dummy_mask)
    _, N, _, C = logits0.shape

    K = int(args.topk)
    if K <= 0 or K > C:
        raise ValueError(f"--topk must be in [1, {C}], got {K}")

    topv_dtype = np.float16 if args.dtype == "float16" else np.float32
    mask_dtype = np.float16 if args.dtype == "float16" else np.float32

    paths = tmp_paths(args.tmp_dir)
    print("[TMP]", paths)

    topv_out = np.memmap(paths["topv"], dtype=topv_dtype, mode="w+", shape=(total, N, N, K))
    topi_out = np.memmap(paths["topi"], dtype=np.uint16, mode="w+", shape=(total, N, N, K))
    frag_ids_out = np.memmap(paths["frag"], dtype=np.int64, mode="w+", shape=(total, N))
    mask_out = np.memmap(paths["mask"], dtype=mask_dtype, mode="w+", shape=(total, N))

    idx = 0

    global_stats = {
        "total_samples": 0,
        "parse_fail_smiles": 0,
        "empty_brics": 0,
        "truncated_samples": 0,
        "total_frags": 0,
        "unknown_fragments": 0,
        "samples_with_unknown": 0,
    }

    for batch_id, (xb, _frag_ids, _mask, _y) in enumerate(tqdm(dl, desc="Export Oracle")):
        bsz = xb.shape[0]
        xb = xb.to(device)

        batch_smiles = smiles_all[idx:idx + bsz]
        oracle_frag_ids_np, oracle_mask_np, st = build_oracle_frag_ids_and_mask(
            smiles_list=batch_smiles,
            s2id=s2id,
            max_nodes=max_nodes,
            unk_id=1,
        )

        for k, v in st.items():
            global_stats[k] += v

        oracle_frag_ids = torch.from_numpy(oracle_frag_ids_np).to(device=device, dtype=torch.long)
        oracle_mask = torch.from_numpy(oracle_mask_np).to(device=device, dtype=torch.float32)

        logits = model(xb, oracle_frag_ids, oracle_mask)   # [B,N,N,C]
        topv, topi = torch.topk(logits, k=K, dim=-1)       # [B,N,N,K]

        if args.dtype == "float16":
            topv = topv.to(torch.float16)

        topv_out[idx:idx + bsz] = topv.detach().cpu().numpy()
        topi_out[idx:idx + bsz] = topi.detach().cpu().numpy().astype(np.uint16)
        frag_ids_out[idx:idx + bsz] = oracle_frag_ids_np
        mask_out[idx:idx + bsz] = oracle_mask_np.astype(mask_dtype)

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
        oracle_fragments=True,
        oracle_parse_fail_smiles=int(global_stats["parse_fail_smiles"]),
        oracle_empty_brics=int(global_stats["empty_brics"]),
        oracle_truncated_samples=int(global_stats["truncated_samples"]),
        oracle_unknown_fragments=int(global_stats["unknown_fragments"]),
        oracle_samples_with_unknown=int(global_stats["samples_with_unknown"]),
    )

    print("Saved:", args.out)
    print("Shapes:", topv_arr.shape, topi_arr.shape, frag_arr.shape, mask_arr.shape)

    print("\n===== ORACLE FRAGMENT STATS =====")
    for k, v in global_stats.items():
        print(f"{k}: {v}")
    if global_stats["total_samples"] > 0:
        avg_frags = global_stats["total_frags"] / max(1, global_stats["total_samples"])
        print("avg_frags_per_sample:", avg_frags)

    del topv_arr, topi_arr, frag_arr, mask_arr
    del topv_out, topi_out, frag_ids_out, mask_out
    gc.collect()

    if args.cleanup_tmp:
        cleanup_tmp(args.tmp_dir)
        print("[CLEANUP] tmp files removed in", args.tmp_dir)


if __name__ == "__main__":
    main()