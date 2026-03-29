import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from tqdm import tqdm

def get_mask_sum_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        frags = BRICS.BRICSDecompose(mol, returnMols=False)
        if frags is None:
            return None
        return len(frags)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    print("rows =", len(df))
    print("columns =", list(df.columns))

    mask_sums = []
    for smi in tqdm(df[args.smiles_col].astype(str).tolist()):
        mask_sums.append(get_mask_sum_from_smiles(smi))

    out_df = pd.DataFrame({
        "row_id": list(range(len(df))),
        "mask_sum": mask_sums
    })

    valid = out_df["mask_sum"].notna().sum()
    print("valid mask_sum rows =", valid)

    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()