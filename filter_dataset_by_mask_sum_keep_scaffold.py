import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet with scaffold column")
    ap.add_argument("--mask_csv", required=True, help="csv with row_id, mask_sum")
    ap.add_argument("--min_mask_sum", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    ms = pd.read_csv(args.mask_csv)

    if len(df) != len(ms):
        raise ValueError(f"row mismatch: data={len(df)} mask_csv={len(ms)}")

    if "scaffold" not in df.columns:
        raise KeyError(f"'scaffold' not found in {args.data}. columns={df.columns.tolist()}")

    df2 = df.copy()
    df2["mask_sum"] = ms["mask_sum"].values
    df2 = df2[df2["mask_sum"] >= args.min_mask_sum].reset_index(drop=True)

    df2.to_parquet(args.out, index=False)

    print("saved:", args.out)
    print("orig rows:", len(df))
    print("kept rows:", len(df2))
    print("columns:", df2.columns.tolist())
    print("unique scaffolds:", df2["scaffold"].astype(str).nunique())

if __name__ == "__main__":
    main()