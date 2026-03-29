import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

tqdm.pandas()

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return None
        smi = Chem.MolToSmiles(scaf)
        if smi == "":
            return None
        return smi
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description="Add Murcko scaffold column to a molecule dataset.")
    parser.add_argument("--data", required=True, help="Input parquet file path (e.g., dataset.parquet)")
    parser.add_argument("--out", required=True, help="Output parquet file path (e.g., dataset_with_scaffold.parquet)")
    args = parser.parse_args()

    INPUT = args.data
    OUTPUT = args.out

    df = pd.read_parquet(INPUT)
    print("Original size:", len(df))

    df["scaffold"] = df["smiles"].progress_apply(get_scaffold)

    # 严格去掉 scaffold 为 None 的
    df = df[df["scaffold"].notnull()].reset_index(drop=True)

    print("After removing invalid scaffold:", len(df))
    print("Unique scaffolds:", df["scaffold"].nunique())

    df.to_parquet(OUTPUT, index=False)
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()