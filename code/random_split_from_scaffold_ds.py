import argparse, pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", default="random60k")
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--val_ratio", type=float, default=0.10)  # ratio of total
    args = ap.parse_args()

    df = pd.read_parquet(args.data).reset_index(drop=True)

    # 先切 test
    df_tr, df_te = train_test_split(df, test_size=args.test_ratio, random_state=args.seed, shuffle=True)

    # 再从 train 里切 val
    val_size_in_train = args.val_ratio / (1.0 - args.test_ratio)
    df_tr, df_va = train_test_split(df_tr, test_size=val_size_in_train, random_state=args.seed, shuffle=True)

    df_tr.to_parquet(f"train_{args.out_prefix}_{args.seed}.parquet", index=False)
    df_va.to_parquet(f"val_{args.out_prefix}_{args.seed}.parquet", index=False)
    df_te.to_parquet(f"test_{args.out_prefix}_{args.seed}.parquet", index=False)

    print("saved:",
          len(df_tr), len(df_va), len(df_te),
          "empty_scaffold_train=", (df_tr["scaffold"]=="").mean())

if __name__ == "__main__":
    main()