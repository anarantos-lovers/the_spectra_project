import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.family'] = 'Arial'      # 全局字体
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 25          # 标题字体大小

plt.rcParams['axes.titlepad'] = 10           # 标题与图的间距
plt.rcParams['axes.labelsize'] = 20     # 坐标轴标签字号
plt.rcParams['xtick.labelsize'] = 15     # x轴刻度字号
plt.rcParams['ytick.labelsize'] = 15     # y轴刻度字号
# =========================
# 基础工具
# =========================
def safe_mol(smiles):
    try:
        if pd.isna(smiles):
            return None
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def canon_smiles(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def get_murcko_scaffold(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return None
        s = Chem.MolToSmiles(scaf, canonical=True)
        if s == "":
            return None
        return s
    except Exception:
        return None


def ensure_scaffold_column(df, smiles_col="smiles", scaffold_col="scaffold"):
    """
    如果 parquet 里已有 scaffold，直接用；
    如果没有，就现场从 smiles 生成。
    """
    if scaffold_col in df.columns:
        out = df.copy()
        out[scaffold_col] = out[scaffold_col].astype(str)
        out.loc[out[scaffold_col].isin(["", "None", "nan"]), scaffold_col] = np.nan
        return out

    if smiles_col not in df.columns:
        raise KeyError(f"Neither '{scaffold_col}' nor '{smiles_col}' exists in test parquet.")

    out = df.copy()
    out[scaffold_col] = out[smiles_col].astype(str).apply(get_murcko_scaffold)
    return out


def guess_true_col(df_pred):
    candidates = ["true", "gt", "label", "target", "smiles", "true_smiles"]
    for c in candidates:
        if c in df_pred.columns:
            return c
    raise KeyError(
        f"Cannot find true SMILES column in prediction CSV. Available columns: {list(df_pred.columns)}"
    )


def guess_pred_cols(df_pred):
    pred_cols = [c for c in df_pred.columns if c.lower().startswith("pred")]
    if len(pred_cols) == 0:
        raise KeyError(
            f"Cannot find prediction columns like pred1/pred2/... in CSV. Available columns: {list(df_pred.columns)}"
        )

    # 尽量按 pred1, pred2, pred3... 排序
    def pred_key(x):
        s = x.lower().replace("pred", "")
        try:
            return int(s)
        except Exception:
            return 999999

    pred_cols = sorted(pred_cols, key=pred_key)
    return pred_cols


# =========================
# 主分析逻辑
# =========================
def main():
    ap = argparse.ArgumentParser(description="Analyze failure patterns by Murcko scaffold.")
    ap.add_argument("--pred_csv", required=True, help="decoded prediction csv, e.g. pred_random0_xxx.csv")
    ap.add_argument("--test_parquet", required=True, help="corresponding test parquet")
    ap.add_argument("--outdir", required=True, help="output folder")

    ap.add_argument("--smiles_col", default="smiles", help="SMILES column in test parquet if scaffold is missing")
    ap.add_argument("--scaffold_col", default="scaffold", help="scaffold column name in test parquet")
    ap.add_argument("--topk", type=int, default=5, help="how many prediction columns to use for Top-k success")
    ap.add_argument("--min_scaffold_size", type=int, default=5, help="minimum scaffold size to include in rate plots")
    ap.add_argument("--topn_plot", type=int, default=20, help="top N scaffolds for plotting")
    ap.add_argument("--export_case_per_scaffold", type=int, default=5, help="how many typical failed cases to export per scaffold")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------- 读数据 ----------
    df_pred = pd.read_csv(args.pred_csv)
    df_test = pd.read_parquet(args.test_parquet)

    # ---------- scaffold ----------
    df_test = ensure_scaffold_column(
        df_test,
        smiles_col=args.smiles_col,
        scaffold_col=args.scaffold_col
    )

    # ---------- 预测列 ----------
    true_col = guess_true_col(df_pred)
    pred_cols = guess_pred_cols(df_pred)
    pred_cols = pred_cols[:args.topk]

    # ---------- 按行对齐 ----------
    # 默认假设 pred_csv 与 test_parquet 的样本顺序一致
    n = min(len(df_pred), len(df_test))
    if len(df_pred) != len(df_test):
        print(f"[WARN] row count mismatch: pred_csv={len(df_pred)} vs test_parquet={len(df_test)}")
        print(f"[WARN] will align by row order using first {n} rows")

    df_pred = df_pred.iloc[:n].reset_index(drop=True).copy()
    df_test = df_test.iloc[:n].reset_index(drop=True).copy()

    # ---------- 标准化 smiles ----------
    df_pred["true_c"] = df_pred[true_col].astype(str).apply(canon_smiles)
    for c in pred_cols:
        df_pred[c + "_c"] = df_pred[c].fillna("").astype(str).apply(canon_smiles)

    # ---------- success / failure ----------
    first_pred = pred_cols[0] + "_c"
    df_pred["top1_success"] = (df_pred[first_pred] == df_pred["true_c"]).astype(int)

    predk_cols_c = [c + "_c" for c in pred_cols]
    df_pred["topk_success"] = df_pred.apply(
        lambda r: int(r["true_c"] in [r[c] for c in predk_cols_c]),
        axis=1
    )

    df_pred["top1_failure"] = 1 - df_pred["top1_success"]
    df_pred["topk_failure"] = 1 - df_pred["topk_success"]

    # ---------- 合并 scaffold ----------
    df = pd.concat(
        [
            df_pred,
            df_test[[args.scaffold_col]].rename(columns={args.scaffold_col: "scaffold"})
        ],
        axis=1
    )

    # 对缺失 scaffold 做兜底
    df["scaffold"] = df["scaffold"].fillna("NO_SCAFFOLD").astype(str)

    # ---------- scaffold summary ----------
    rows = []
    grouped = df.groupby("scaffold", dropna=False)

    for scaf, sub in grouped:
        n_total = len(sub)
        top1_succ = int(sub["top1_success"].sum())
        top1_fail = int(sub["top1_failure"].sum())
        topk_succ = int(sub["topk_success"].sum())
        topk_fail = int(sub["topk_failure"].sum())

        rows.append({
            "scaffold": scaf,
            "n_total": n_total,
            "top1_success_n": top1_succ,
            "top1_failure_n": top1_fail,
            "top1_success_rate": top1_succ / n_total if n_total > 0 else np.nan,
            "top1_failure_rate": top1_fail / n_total if n_total > 0 else np.nan,
            f"top{args.topk}_success_n": topk_succ,
            f"top{args.topk}_failure_n": topk_fail,
            f"top{args.topk}_success_rate": topk_succ / n_total if n_total > 0 else np.nan,
            f"top{args.topk}_failure_rate": topk_fail / n_total if n_total > 0 else np.nan,
        })

    summary = pd.DataFrame(rows)

    # scaffold size rank
    summary["size_rank"] = summary["n_total"].rank(method="dense", ascending=False).astype(int)

    # 保存完整统计
    summary = summary.sort_values(["n_total", "top1_failure_rate"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(
        os.path.join(args.outdir, "scaffold_summary_full.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ---------- 导出常见 scaffold ----------
    summary_head = summary.head(50).copy()
    summary_head.to_csv(
        os.path.join(args.outdir, "scaffold_summary_top50_by_size.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ---------- 失败数最多 ----------
    plot_fail_count = summary.sort_values("top1_failure_n", ascending=False).head(args.topn_plot).copy()
    plot_fail_count = plot_fail_count.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(plot_fail_count["scaffold"], plot_fail_count["top1_failure_n"])
    plt.xlabel("Top-1 failure count")
    plt.ylabel("Scaffold")
    plt.title("Scaffolds with the most Top-1 failures")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "scaffold_top1_failure_count.png"), dpi=300)
    plt.close()

    # ---------- 失败率最高 ----------
    plot_fail_rate = summary[summary["n_total"] >= args.min_scaffold_size].copy()
    plot_fail_rate = plot_fail_rate.sort_values("top1_failure_rate", ascending=False).head(args.topn_plot)
    plot_fail_rate = plot_fail_rate.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(plot_fail_rate["scaffold"], plot_fail_rate["top1_failure_rate"])
    plt.xlabel("Top-1 failure rate")
    plt.ylabel("Scaffold")
    plt.title(f"Scaffolds with the highest Top-1 failure rates (n ≥ {args.min_scaffold_size})")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "scaffold_top1_failure_rate.png"), dpi=300)
    plt.close()

    # ---------- Top-k 失败率 ----------
    topk_fail_rate_col = f"top{args.topk}_failure_rate"
    plot_topk_fail_rate = summary[summary["n_total"] >= args.min_scaffold_size].copy()
    plot_topk_fail_rate = plot_topk_fail_rate.sort_values(topk_fail_rate_col, ascending=False).head(args.topn_plot)
    plot_topk_fail_rate = plot_topk_fail_rate.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(plot_topk_fail_rate["scaffold"], plot_topk_fail_rate[topk_fail_rate_col])
    plt.xlabel(f"Top-{args.topk} failure rate")
    plt.ylabel("Scaffold")
    plt.title(f"Scaffolds with the highest Top-{args.topk} failure rates (n ≥ {args.min_scaffold_size})")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"scaffold_top{args.topk}_failure_rate.png"), dpi=300)
    plt.close()

    # ---------- 成功/失败堆叠图（按 scaffold size 排名前 N） ----------
    plot_stacked = summary.sort_values("n_total", ascending=False).head(args.topn_plot).copy()
    plot_stacked = plot_stacked.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(plot_stacked["scaffold"], plot_stacked["top1_success_n"], label="Top-1 success")
    plt.barh(
        plot_stacked["scaffold"],
        plot_stacked["top1_failure_n"],
        left=plot_stacked["top1_success_n"],
        label="Top-1 failure"
    )
    plt.xlabel("Number of samples")
    plt.ylabel("Scaffold")
    plt.title("Top-1 success/failure composition in the most frequent scaffolds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "scaffold_top1_success_failure_stacked.png"), dpi=300)
    plt.close()

    # ---------- 典型失败案例导出 ----------
    # 对失败数较多的 scaffold 导出若干案例
    major_scafs = summary.sort_values("top1_failure_n", ascending=False).head(args.topn_plot)["scaffold"].tolist()

    case_rows = []
    for scaf in major_scafs:
        sub = df[(df["scaffold"] == scaf) & (df["top1_failure"] == 1)].copy()
        if len(sub) == 0:
            continue
        sub = sub.head(args.export_case_per_scaffold)

        for _, r in sub.iterrows():
            row = {
                "scaffold": scaf,
                "true": r["true_c"],
                "top1_pred": r[first_pred],
                "top1_success": r["top1_success"],
                f"top{args.topk}_success": r["topk_success"],
            }
            for c in pred_cols:
                row[c] = r[c + "_c"]
            case_rows.append(row)

    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(
        os.path.join(args.outdir, "typical_failed_cases_by_scaffold.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ---------- scaffold 频数分布 ----------
    size_dist = summary["n_total"].value_counts().sort_index().reset_index()
    size_dist.columns = ["scaffold_size", "num_scaffolds"]
    size_dist.to_csv(
        os.path.join(args.outdir, "scaffold_size_distribution.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== DONE =====")
    print("Prediction CSV:", args.pred_csv)
    print("Test parquet   :", args.test_parquet)
    print("Output dir     :", args.outdir)
    print("True column    :", true_col)
    print("Pred columns   :", pred_cols)
    print("Total samples  :", len(df))
    print("Unique scaffolds:", df["scaffold"].nunique())


if __name__ == "__main__":
    main()