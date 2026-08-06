import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from stage2_brics_dataset import Stage2BRICSDataset


# =========================
# extract IR + SMILES
# =========================
def extract_ir_and_smiles(dataset):
    """
    Extract flattened raw IR spectra and molecule-level SMILES labels.

    This baseline evaluates whether raw IR spectral similarity alone can
    recover the correct molecule by nearest-neighbor retrieval.
    """

    ir_list = []
    smi_list = []

    for i in tqdm(range(len(dataset)), desc="Extract IR + SMILES"):
        ir, frag_ids, mask, y = dataset[i]

        ir_list.append(ir.numpy().reshape(-1))

        # molecule-level label, not pairwise label y
        smi = str(dataset.df.iloc[i]["smiles"])
        smi_list.append(smi)

    return np.asarray(ir_list), np.asarray(smi_list)


# =========================
# IR-NN retrieval
# =========================
def ir_nn_retrieval(train_ir, train_smi, test_ir, k=5, metric="euclidean"):
    """
    Retrieve top-k nearest training spectra in raw IR space.
    """

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(train_ir)

    distances, indices = nn.kneighbors(test_ir)

    preds_topk = train_smi[indices]

    return preds_topk, distances


# =========================
# top-k accuracy
# =========================
def topk_accuracy(preds_topk, gt_smi, k=1):
    """
    Molecule-level Top-k exact-match accuracy.
    """

    correct = 0

    for i in range(len(gt_smi)):
        if gt_smi[i] in preds_topk[i, :k]:
            correct += 1

    return correct / len(gt_smi)


# =========================
# main
# =========================
if __name__ == "__main__":

    print("Loading dataset...")

    train = Stage2BRICSDataset(
        parquet_path="train_mask3_63k_random0_0.parquet",
        vocab_tsv="vocab_global_63k.tsv"
    )

    test = Stage2BRICSDataset(
        parquet_path="test_mask3_63k_random0_0.parquet",
        vocab_tsv="vocab_global_63k.tsv"
    )

    print("Extracting IR + SMILES...")

    train_ir, train_smi = extract_ir_and_smiles(train)
    test_ir, test_smi = extract_ir_and_smiles(test)

    print("Running IR-NN retrieval...")

    preds_topk, distances = ir_nn_retrieval(
        train_ir=train_ir,
        train_smi=train_smi,
        test_ir=test_ir,
        k=5,
        metric="euclidean"
    )

    top1 = topk_accuracy(preds_topk, test_smi, k=1)
    top5 = topk_accuracy(preds_topk, test_smi, k=5)

    validity = 1.0
    empty = 0.0

    print("\n===== IR-NN BASELINE =====")
    print(f"Top-1:   {top1:.4f}")
    print(f"Top-5:   {top5:.4f}")
    print(f"Validity:{validity:.4f}")
    print(f"Empty:   {empty:.4f}")

    df = pd.DataFrame([{
        "method": "IR-NN",
        "retrieval_space": "raw_IR",
        "metric": "euclidean",
        "top1": top1,
        "top5": top5,
        "validity": validity,
        "empty": empty
    }])

    df.to_csv("irnn_baseline_summary.csv", index=False)

    # optional: save nearest-neighbor prediction details
    detail_df = pd.DataFrame({
        "test_smiles": test_smi,
        "pred_top1_smiles": preds_topk[:, 0],
        "correct_top1": preds_topk[:, 0] == test_smi,
        "nearest_distance": distances[:, 0]
    })
    detail_df.to_csv("irnn_baseline_details.csv", index=False)

    print("\nSaved: irnn_baseline_summary.csv")
    print("Saved: irnn_baseline_details.csv")