import argparse
import pandas as pd
import numpy as np


def infer_mask_sum(row):
    """
    尽量兼容几种常见情况：
    1) 已经有 mask_sum 列
    2) 有 mask 列，且是 list / np.ndarray / 可迭代对象
    3) 有 frag_mask 列
    """
    if "mask_sum" in row.index and pd.notna(row["mask_sum"]):
        try:
            return int(row["mask_sum"])
        except Exception:
            pass

    for col in ["mask", "frag_mask"]:
        if col in row.index:
            v = row[col]
            if v is None:
                continue
            try:
                arr = np.array(v, dtype=float)
                if arr.ndim >= 1:
                    return int((arr > 0.5).sum())
            except Exception:
                pass

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet file path")
    ap.add_argument("--out_csv", default="", help="optional output csv for full distribution")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    print("rows =", len(df))
    print("columns =", list(df.columns))

    if "mask_sum" in df.columns:
        ms = pd.to_numeric(df["mask_sum"], errors="coerce")
    else:
        ms = df.apply(infer_mask_sum, axis=1)
        ms = pd.to_numeric(ms, errors="coerce")

    valid = ms.notna().sum()
    print("valid mask_sum rows =", int(valid))

    ms = ms.dropna().astype(int)

    if len(ms) == 0:
        print("No valid mask_sum could be inferred.")
        return

    print("\n===== BASIC STATS =====")
    print("min =", int(ms.min()))
    print("max =", int(ms.max()))
    print("mean =", float(ms.mean()))
    print("median =", float(ms.median()))

    print("\n===== KEY BUCKETS =====")
    n = len(ms)
    n2 = int((ms == 2).sum())
    n3 = int((ms == 3).sum())
    n4p = int((ms >= 4).sum())
    n_le2 = int((ms <= 2).sum())
    n_le3 = int((ms <= 3).sum())

    print(f"mask_sum == 2 : {n2}  ({n2/n:.4%})")
    print(f"mask_sum == 3 : {n3}  ({n3/n:.4%})")
    print(f"mask_sum >= 4 : {n4p}  ({n4p/n:.4%})")
    print(f"mask_sum <= 2 : {n_le2} ({n_le2/n:.4%})")
    print(f"mask_sum <= 3 : {n_le3} ({n_le3/n:.4%})")

    print("\n===== FULL DISTRIBUTION =====")
    dist = ms.value_counts().sort_index()
    dist_df = pd.DataFrame({
        "mask_sum": dist.index,
        "count": dist.values,
    })
    dist_df["ratio"] = dist_df["count"] / n
    print(dist_df.to_string(index=False))

    if args.out_csv:
        dist_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print("\nSaved distribution csv to:", args.out_csv)


if __name__ == "__main__":
    main()