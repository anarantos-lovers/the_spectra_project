import pandas as pd
import numpy as np
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="输入的 parquet 文件路径")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--n_splits", type=int, default=10, help="折数")
    ap.add_argument("--outdir", default=".", help="输出目录（会自动创建）")
    args = ap.parse_args()

    # 创建输出目录（如果不存在）
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.data)
    scaffolds = df["scaffold"].unique()

    rng = np.random.RandomState(args.seed)
    rng.shuffle(scaffolds)

    folds = np.array_split(scaffolds, args.n_splits)

    for i in range(args.n_splits):
        test_scaf = set(folds[i])
        test_mask = df["scaffold"].isin(test_scaf)

        df_test = df[test_mask].reset_index(drop=True)
        df_trainval = df[~test_mask].reset_index(drop=True)

        # 从 trainval 中随机切 10% 作为验证集
        n = len(df_trainval)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_val = int(0.1 * n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)
        df_train = df_trainval.iloc[train_idx].reset_index(drop=True)

        # 保存文件到 outdir，文件名格式：train_scaffold_{i}.parquet 等
        train_path = os.path.join(args.outdir, f"train_scaffold_{i}.parquet")
        val_path   = os.path.join(args.outdir, f"val_scaffold_{i}.parquet")
        test_path  = os.path.join(args.outdir, f"test_scaffold_{i}.parquet")

        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)
        df_test.to_parquet(test_path, index=False)

        print(f"Fold {i}: train={len(df_train)} val={len(df_val)} test={len(df_test)}")
        print(f"  保存至: {train_path}, {val_path}, {test_path}")

if __name__ == "__main__":
    main()