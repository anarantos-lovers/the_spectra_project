# build_brics_vocab_from_train.py
from __future__ import annotations
import argparse
from collections import Counter
from typing import Iterable, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS


def canon_smiles(s: str) -> Optional[str]:
    """canonicalize a smiles; keep BRICS dummies if present."""
    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        # 不做 dummy 删除：fragment vocab 需要保留 BRICS dummy
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def brics_frags(smiles: str) -> Iterable[str]:
    """
    用 RDKit BRICS 分解得到 fragments。
    输出是带 dummy 的 fragment SMILES（用于你的 Stage2 片段序列）。
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    try:
        frs = BRICS.BRICSDecompose(m, returnMols=False)
        # frs 是 set[str]
        return list(frs)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train parquet (e.g. train_scaffold_0.parquet)")
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--out_tsv", required=True, help="output vocab tsv: frag_smiles\\tcount")
    ap.add_argument("--min_count", type=int, default=1, help="drop fragments with count < min_count")
    ap.add_argument("--max_vocab", type=int, default=0, help="0 means no limit; else keep top K by count")
    ap.add_argument("--dedup_molecule", action="store_true",
                    help="每个分子内部对 fragment 去重（更保守，避免某些分子重复计数）")
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    if args.smiles_col not in df.columns:
        raise KeyError(f"Column '{args.smiles_col}' not found. Available: {list(df.columns)[:20]}")

    cnt = Counter()
    n_total = 0
    n_ok = 0

    for s in df[args.smiles_col].tolist():
        n_total += 1
        cs = canon_smiles(s)
        if not cs:
            continue
        n_ok += 1
        frs = brics_frags(cs)
        if args.dedup_molecule:
            frs = sorted(set(frs))
        for f in frs:
            cf = canon_smiles(f)
            if cf:
                cnt[cf] += 1

    # filter
    items = [(k, v) for k, v in cnt.items() if v >= int(args.min_count)]
    # sort by count desc, then smiles
    items.sort(key=lambda x: (-x[1], x[0]))

    if args.max_vocab and args.max_vocab > 0:
        items = items[: int(args.max_vocab)]

    with open(args.out_tsv, "w", encoding="utf-8") as f:
        f.write("# frag_smiles\tcount\n")
        for smi, c in items:
            f.write(f"{smi}\t{c}\n")

    print("===== BUILD BRICS VOCAB DONE =====")
    print("train:", args.train)
    print("molecules total:", n_total, "parsed:", n_ok)
    print("unique frags (after filter):", len(items))
    print("saved:", args.out_tsv)


if __name__ == "__main__":
    main()