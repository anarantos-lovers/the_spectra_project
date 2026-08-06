import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from stage2_brics_dataset import Stage2BRICSDataset
from stage2_brics_model import Stage2BRICSModel


# =========================
# collect seen fragment ids from train set
# =========================
def collect_seen_fragment_ids(dataset):
    """
    Collect fragment ids that appear in the training set.
    This avoids mixing string-based vocabulary keys with integer fragment ids.
    """

    seen = set()
    pad_id = dataset.vocab.pad_id

    for i in tqdm(range(len(dataset)), desc="Collect train fragment ids"):
        ir, frag_ids, mask, y = dataset[i]

        for f in frag_ids.numpy().tolist():
            f = int(f)
            if f != pad_id:
                seen.add(f)

    return seen


# =========================
# classify OOD level
# =========================
def classify_fragment_ood(frag_ids, seen_train_ids, pad_id):
    """
    Classify a test molecule according to whether its fragment ids
    were observed in the training set.

    fully_seen:
        all valid fragments are observed in train.
    partial_unseen:
        some but not all valid fragments are unseen.
    fully_unseen:
        all valid fragments are unseen.
    """

    valid = [int(f) for f in frag_ids if int(f) != pad_id]

    if len(valid) == 0:
        return "empty"

    unseen = [f for f in valid if f not in seen_train_ids]

    if len(unseen) == 0:
        return "fully_seen"
    elif len(unseen) == len(valid):
        return "fully_unseen"
    else:
        return "partial_unseen"


# =========================
# optional: masked pairwise accuracy
# =========================
def pairwise_accuracy(pred, y, mask):
    """
    Compute pairwise connection-label accuracy on valid fragment pairs only.

    pred: [1, N, N] or [N, N]
    y:    [1, N, N] or [N, N]
    mask: [1, N] or [N]

    This avoids padded positions dominating the accuracy.
    """

    if pred.dim() == 3:
        pred = pred[0]
    if y.dim() == 3:
        y = y[0]
    if mask.dim() == 2:
        mask = mask[0]

    valid_pair_mask = (mask[:, None] > 0) & (mask[None, :] > 0)

    # remove diagonal self-pairs
    n = valid_pair_mask.shape[0]
    diag = torch.eye(n, dtype=torch.bool, device=valid_pair_mask.device)
    valid_pair_mask = valid_pair_mask & (~diag)

    if valid_pair_mask.sum().item() == 0:
        return np.nan

    acc = (pred[valid_pair_mask] == y[valid_pair_mask]).float().mean().item()

    return acc


# =========================
# evaluate model
# =========================
def evaluate(model, dataset, seen_train_ids, pad_id, device="cuda"):
    """
    Evaluate pairwise connection-label accuracy across fragment-OOD categories.
    """

    stats = {
        "fully_seen": [],
        "partial_unseen": [],
        "fully_unseen": []
    }

    counts = {k: 0 for k in stats}

    model.eval()

    with torch.no_grad():

        for ir, frag_ids, mask, y in tqdm(dataset, desc="Running OOD analysis"):

            cat = classify_fragment_ood(
                frag_ids=frag_ids.numpy(),
                seen_train_ids=seen_train_ids,
                pad_id=pad_id
            )

            if cat not in stats:
                continue

            ir = ir.unsqueeze(0).to(device)
            frag_ids = frag_ids.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            logits = model(ir, frag_ids, mask)
            pred = logits.argmax(-1)

            acc = pairwise_accuracy(pred, y, mask)

            if not np.isnan(acc):
                stats[cat].append(acc)
                counts[cat] += 1

    return stats, counts


# =========================
# summarize
# =========================
def summarize(stats, counts):
    rows = []

    for k in ["fully_seen", "partial_unseen", "fully_unseen"]:
        values = stats[k]

        if len(values) == 0:
            rows.append({
                "category": k,
                "count": 0,
                "pairwise_acc_mean": np.nan,
                "pairwise_acc_std": np.nan
            })
        else:
            rows.append({
                "category": k,
                "count": counts[k],
                "pairwise_acc_mean": float(np.mean(values)),
                "pairwise_acc_std": float(np.std(values))
            })

    return pd.DataFrame(rows)


# =========================
# main
# =========================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading datasets...")

    train_dataset = Stage2BRICSDataset(
        parquet_path="train_mask3_63k_random0_0.parquet",
        vocab_tsv="vocab_global_63k.tsv"
    )

    test_dataset = Stage2BRICSDataset(
        parquet_path="test_mask3_63k_random0_0.parquet",
        vocab_tsv="vocab_global_63k.tsv"
    )

    pad_id = train_dataset.vocab.pad_id

    print("Collecting seen fragment ids from training set...")

    seen_train_ids = collect_seen_fragment_ids(train_dataset)

    print(f"Seen train fragment ids: {len(seen_train_ids)}")

    print("Loading model...")

    model = Stage2BRICSModel(
        x_dim=1024,
        vocab_size=len(train_dataset.vocab)
    ).to(device)

    # =========================
    # IMPORTANT:
    # Load trained checkpoint here if available.
    # Without this, results reflect a randomly initialized model.
    # =========================
    # ckpt_path = "your_trained_model.pt"
    # state = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(state)

    print("Running fragment-vocabulary OOD analysis...")

    stats, counts = evaluate(
        model=model,
        dataset=test_dataset,
        seen_train_ids=seen_train_ids,
        pad_id=pad_id,
        device=device
    )

    df = summarize(stats, counts)

    print("\n===== FRAGMENT-VOCABULARY OOD BREAKDOWN =====")
    print(df)

    df.to_csv("fragment_vocab_ood_pairwise_summary.csv", index=False)

    print("\nSaved: fragment_vocab_ood_pairwise_summary.csv")