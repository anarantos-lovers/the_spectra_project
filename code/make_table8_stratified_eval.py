import argparse
import pandas as pd
import numpy as np


def normalize_smiles(s):
    if pd.isna(s):
        return ""
    return str(s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_data", required=True, help="full parquet with scaffold, e.g. dataset_with_scaffold_63k.parquet")
    ap.add_argument("--mask_csv", required=True, help="mask_sum csv generated from full_data")
    ap.add_argument("--test_parquet", required=True, help="test parquet, e.g. test_raw_63k_random0_0.parquet")
    ap.add_argument("--pred_csv", required=True, help="prediction csv for the same test set")
    ap.add_argument("--true_col", default="true", help="ground-truth smiles column in pred csv")
    ap.add_argument("--pred_cols", nargs="+", default=["pred1", "pred2", "pred3", "pred4", "pred5"],
                    help="prediction columns in pred csv")
    ap.add_argument("--out_csv", default="table8_stratified_eval.csv", help="output csv path")
    args = ap.parse_args()

    # 1) read files
    full_df = pd.read_parquet(args.full_data)
    mask_df = pd.read_csv(args.mask_csv)
    test_df = pd.read_parquet(args.test_parquet)
    pred_df = pd.read_csv(args.pred_csv)

    print("full_df rows =", len(full_df))
    print("mask_df rows =", len(mask_df))
    print("test_df rows =", len(test_df))
    print("pred_df rows =", len(pred_df))

    # 2) basic checks
    if "smiles" not in full_df.columns:
        raise ValueError("full_data must contain 'smiles' column")
    if "smiles" not in test_df.columns:
        raise ValueError("test_parquet must contain 'smiles' column")
    if "mask_sum" not in mask_df.columns:
        raise ValueError("mask_csv must contain 'mask_sum' column")

    # 3) align full_data with mask_csv by row order
    if len(full_df) != len(mask_df):
        raise ValueError(f"Row count mismatch: full_data={len(full_df)} vs mask_csv={len(mask_df)}")

    full_df = full_df.copy()
    full_df["mask_sum"] = pd.to_numeric(mask_df["mask_sum"], errors="coerce")

    # 4) attach mask_sum to test set by smiles
    # assume unique enough for this split; if duplicates exist, keep first
    map_df = full_df[["smiles", "mask_sum"]].copy()
    map_df["smiles_norm"] = map_df["smiles"].map(normalize_smiles)
    map_df = map_df.drop_duplicates(subset=["smiles_norm"], keep="first")

    test_df = test_df.copy()
    test_df["smiles_norm"] = test_df["smiles"].map(normalize_smiles)

    test_df = test_df.merge(
        map_df[["smiles_norm", "mask_sum"]],
        on="smiles_norm",
        how="left"
    )

    matched = test_df["mask_sum"].notna().sum()
    print("matched test rows with mask_sum =", int(matched), "/", len(test_df))

    # 5) align pred with test by row order
    n = min(len(test_df), len(pred_df))
    if len(test_df) != len(pred_df):
        print(f"[WARN] row mismatch: test_df={len(test_df)} vs pred_df={len(pred_df)}; using first {n} rows")

    test_df = test_df.iloc[:n].reset_index(drop=True)
    pred_df = pred_df.iloc[:n].reset_index(drop=True)

    # 6) compute Top1/Top5 exact match
    true_smiles = pred_df[args.true_col].map(normalize_smiles)

    pred1 = pred_df[args.pred_cols[0]].map(normalize_smiles)
    top1_hit = (pred1 == true_smiles).astype(int)

    top5_hit = []
    for i in range(n):
        t = true_smiles.iloc[i]
        preds = [normalize_smiles(pred_df.loc[i, c]) for c in args.pred_cols if c in pred_df.columns]
        top5_hit.append(int(t in preds))
    top5_hit = pd.Series(top5_hit)

    # 7) tanimoto-like metric
    # if pred csv already has best_tanimoto or tanimoto, use it; else NaN
    tani_col = None
    for c in ["best_tanimoto", "avg_best_tanimoto", "tanimoto", "best_tani"]:
        if c in pred_df.columns:
            tani_col = c
            break

    if tani_col is not None:
        tani = pd.to_numeric(pred_df[tani_col], errors="coerce")
    else:
        tani = pd.Series([np.nan] * n)
        print("[WARN] no tanimoto column found in pred_csv; Tanimoto will be NaN")

    eval_df = pd.DataFrame({
        "mask_sum": pd.to_numeric(test_df["mask_sum"], errors="coerce"),
        "top1": top1_hit,
        "top5": top5_hit,
        "tanimoto": tani
    })

    # 8) group definitions
    def group_name(x):
        if pd.isna(x):
            return "other"
        x = int(x)
        if x == 2:
            return "mask_sum=2"
        elif x >= 3:
            return "mask_sum>=3"
        else:
            return "other"

    eval_df["group"] = eval_df["mask_sum"].map(group_name)

    # 9) summarize
    rows = []
    for g in ["mask_sum=2", "mask_sum>=3", "other"]:
        sub = eval_df[eval_df["group"] == g]
        if len(sub) == 0:
            rows.append([g, 0, 0.0, 0.0, 0.0])
        else:
            rows.append([
                g,
                len(sub),
                sub["top1"].mean(),
                sub["top5"].mean(),
                sub["tanimoto"].mean(skipna=True)
            ])

    rows.append([
        "all",
        len(eval_df),
        eval_df["top1"].mean(),
        eval_df["top5"].mean(),
        eval_df["tanimoto"].mean(skipna=True)
    ])

    out_df = pd.DataFrame(rows, columns=["group", "n", "Top1", "Top5", "Tanimoto"])
    print("\n===== TABLE 8 STYLE SUMMARY =====")
    print(out_df.to_string(index=False))

    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("\nSaved to:", args.out_csv)


if __name__ == "__main__":
    main()