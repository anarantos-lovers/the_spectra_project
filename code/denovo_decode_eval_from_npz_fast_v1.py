from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

NONE_CLASS = 256

# ========= caches =========
_FRAG_MOL_CACHE: Dict[str, Optional[Chem.Mol]] = {}
_CANON_CACHE: Dict[str, str] = {}
_POST_CACHE: Dict[Tuple[str, bool], str] = {}
_TANI_CACHE: Dict[Tuple[str, str], float] = {}


def mol_from_smiles_cached(s: str) -> Optional[Chem.Mol]:
    if s not in _FRAG_MOL_CACHE:
        _FRAG_MOL_CACHE[s] = Chem.MolFromSmiles(s)
    return _FRAG_MOL_CACHE[s]


def canon(smiles: str) -> str:
    if smiles in _CANON_CACHE:
        return _CANON_CACHE[smiles]
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        _CANON_CACHE[smiles] = ""
    else:
        _CANON_CACHE[smiles] = Chem.MolToSmiles(m, isomericSmiles=True)
    return _CANON_CACHE[smiles]


def largest_component_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    parts = smiles.split(".")
    if not parts:
        return ""
    parts = [p for p in parts if p]
    if not parts:
        return ""
    best = ""
    best_atoms = -1
    for p in parts:
        m = mol_from_smiles_cached(p)
        if m is None:
            continue
        n = m.GetNumAtoms()
        if n > best_atoms:
            best_atoms = n
            best = Chem.MolToSmiles(m, isomericSmiles=True)
    return best


def strip_dummies(smiles: str) -> str:
    m = mol_from_smiles_cached(smiles)
    if m is None:
        return ""
    rw = Chem.RWMol(m)
    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummies, reverse=True):
        try:
            rw.RemoveAtom(int(idx))
        except Exception:
            pass
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return ""
    return Chem.MolToSmiles(m2, isomericSmiles=True)


def postprocess_cached(smiles_with_dummies: str, largest_component: bool) -> str:
    key = (smiles_with_dummies, largest_component)
    if key in _POST_CACHE:
        return _POST_CACHE[key]

    s = strip_dummies(smiles_with_dummies)
    if not s:
        _POST_CACHE[key] = ""
        return ""

    if largest_component:
        s = largest_component_smiles(s)

    s = canon(s)
    _POST_CACHE[key] = s
    return s


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    if smiles_a > smiles_b:
        key = (smiles_b, smiles_a)
    else:
        key = (smiles_a, smiles_b)

    if key in _TANI_CACHE:
        return _TANI_CACHE[key]

    ma = mol_from_smiles_cached(smiles_a)
    mb = mol_from_smiles_cached(smiles_b)
    if ma is None or mb is None:
        _TANI_CACHE[key] = 0.0
        return 0.0

    fa = Chem.RDKFingerprint(ma)
    fb = Chem.RDKFingerprint(mb)
    v = float(DataStructs.TanimotoSimilarity(fa, fb))
    _TANI_CACHE[key] = v
    return v


def get_id2s_from_npz(npz: np.lib.npyio.NpzFile) -> List[str]:
    if "vocab_id2s" in npz.files:
        arr = npz["vocab_id2s"]
    elif "id2s" in npz.files:
        arr = npz["id2s"]
    else:
        raise KeyError("NPZ missing vocab_id2s/id2s")
    lst = arr.tolist()
    if not isinstance(lst, list):
        lst = list(lst)
    return [("" if x is None else str(x)) for x in lst]


def decode_typepair(cls: int) -> Tuple[int, int]:
    ti = (cls % 16) + 1
    tj = (cls // 16) + 1
    return ti, tj


def _find_dummy_atom_idx(m: Chem.Mol, label: int) -> Optional[int]:
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetIsotope() == label:
            return a.GetIdx()
    return None


def _neighbor_of_atom(m: Chem.Mol, atom_idx: int) -> Optional[int]:
    a = m.GetAtomWithIdx(atom_idx)
    neigh = a.GetNeighbors()
    if len(neigh) != 1:
        return None
    return neigh[0].GetIdx()


def has_label(frag: str, label: int) -> bool:
    return f"[{label}*]" in frag


def connect_two_frags_keep_dummies(fa: str, fb: str, ti: int, tj: int) -> List[str]:
    if not fa or not fb:
        return []

    ma = mol_from_smiles_cached(fa)
    mb = mol_from_smiles_cached(fb)
    if ma is None or mb is None:
        return []

    da = _find_dummy_atom_idx(ma, ti)
    db = _find_dummy_atom_idx(mb, tj)
    if da is None or db is None:
        return []

    aa = _neighbor_of_atom(ma, da)
    bb = _neighbor_of_atom(mb, db)
    if aa is None or bb is None:
        return []

    combo = Chem.CombineMols(ma, mb)
    rw = Chem.RWMol(combo)

    off = ma.GetNumAtoms()
    bb2 = bb + off
    db2 = db + off

    try:
        rw.AddBond(int(aa), int(bb2), Chem.BondType.SINGLE)
    except Exception:
        return []

    for idx in sorted([da, db2], reverse=True):
        try:
            rw.RemoveAtom(int(idx))
        except Exception:
            pass

    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return []

    s = Chem.MolToSmiles(m2, isomericSmiles=True)
    return [s] if s else []


@dataclass
class BeamMol:
    smiles_with_dummies: str
    score: float


@dataclass
class EdgePick:
    i: int
    j: int
    cls: int
    score: float


def top_edges_from_logits(
    logits: np.ndarray,
    node_idx: List[int],
    top_edge_m: int,
    top_type_r: int,
) -> List[EdgePick]:
    picks: List[EdgePick] = []
    for ii in node_idx:
        for jj in node_idx:
            if ii == jj:
                continue
            vec = logits[ii, jj]
            order = np.argsort(vec[:NONE_CLASS])[::-1][:top_type_r]
            for cls in order:
                picks.append(EdgePick(ii, jj, int(cls), float(vec[int(cls)])))

    picks.sort(key=lambda x: x.score, reverse=True)
    return picks[:top_edge_m]


def top_edges_from_sparse(
    topv: np.ndarray,
    topi: np.ndarray,
    node_idx: List[int],
    top_edge_m: int,
    top_type_r: int,
) -> List[EdgePick]:
    picks: List[EdgePick] = []
    for ii in node_idx:
        for jj in node_idx:
            if ii == jj:
                continue

            cls_arr = topi[ii, jj]
            val_arr = topv[ii, jj]

            cand: List[Tuple[int, float]] = []
            for cls, val in zip(cls_arr.tolist(), val_arr.tolist()):
                cls = int(cls)
                if cls >= NONE_CLASS:
                    continue
                cand.append((cls, float(val)))

            cand.sort(key=lambda x: x[1], reverse=True)
            cand = cand[:top_type_r]

            for cls, score in cand:
                picks.append(EdgePick(ii, jj, cls, score))

    picks.sort(key=lambda x: x.score, reverse=True)
    return picks[:top_edge_m]


def _finalize_beam_to_smiles(beam: List[BeamMol], topk: int, largest_component: bool) -> List[str]:
    beam = sorted(beam, key=lambda x: x.score, reverse=True)
    outs: List[str] = []
    seen = set()
    for b in beam:
        s = postprocess_cached(b.smiles_with_dummies, largest_component=largest_component)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        outs.append(s)
        if len(outs) >= topk:
            break
    return outs


def decode_one(
    frag_ids: np.ndarray,
    mask: np.ndarray,
    id2s: List[str],
    beam_size: int,
    top_edge_m: int,
    top_type_r: int,
    topk_out: int,
    strict_frag: bool,
    largest_component: bool,
    logits: Optional[np.ndarray] = None,
    topv: Optional[np.ndarray] = None,
    topi: Optional[np.ndarray] = None,
) -> List[str]:
    node_ok = mask > 0.5
    idxs = np.where(node_ok)[0].tolist()
    if len(idxs) == 0:
        return []

    frags: Dict[int, str] = {}
    valid_idxs: List[int] = []

    INVALID_FRAG_TOKENS = {"", "<UNK>", "[UNK]", "UNK", "<PAD>", "[PAD]", "PAD", "None", "nan"}
    DUMMY16 = "".join([f"[{i}*]" for i in range(1, 17)])

    for i in idxs:
        fid = int(frag_ids[i])
        s = str(id2s[fid]) if (0 <= fid < len(id2s)) else ""
        s = s.strip()

        if s in INVALID_FRAG_TOKENS:
            continue

        if s != DUMMY16 and mol_from_smiles_cached(s) is None:
            continue

        frags[i] = s
        valid_idxs.append(i)

    idxs = valid_idxs
    if len(idxs) == 0:
        return []

    if logits is not None:
        edges = top_edges_from_logits(logits, idxs, top_edge_m=top_edge_m, top_type_r=top_type_r)
    else:
        if topv is None or topi is None:
            return []
        edges = top_edges_from_sparse(topv, topi, idxs, top_edge_m=top_edge_m, top_type_r=top_type_r)

    if not edges:
        return []

    if len(idxs) == 2:
        a, b = idxs[0], idxs[1]
        cand_edges = [e for e in edges if (e.i == a and e.j == b) or (e.i == b and e.j == a)]
        cand_edges.sort(key=lambda x: x.score, reverse=True)

        beam: List[BeamMol] = []
        seen_raw = set()

        for e in cand_edges[: max(beam_size * 24, 512)]:
            ti, tj = decode_typepair(e.cls)

            outs = []
            if (not strict_frag) or (has_label(frags[e.i], ti) and has_label(frags[e.j], tj)):
                outs = connect_two_frags_keep_dummies(frags[e.i], frags[e.j], ti, tj)

            if not outs:
                if (not strict_frag) or (has_label(frags[e.j], tj) and has_label(frags[e.i], ti)):
                    outs = connect_two_frags_keep_dummies(frags[e.j], frags[e.i], tj, ti)

            for s in outs:
                if not s:
                    continue
                if s in seen_raw:
                    continue
                seen_raw.add(s)
                beam.append(BeamMol(s, e.score))

            if len(beam) >= beam_size * 6:
                break

        return _finalize_beam_to_smiles(beam, topk=topk_out, largest_component=largest_component)

    beam_states: List[Tuple[BeamMol, Tuple[int, ...]]] = []
    for e in edges[: max(beam_size * 8, 128)]:
        ti, tj = decode_typepair(e.cls)
        if strict_frag and (not has_label(frags[e.i], ti) or not has_label(frags[e.j], tj)):
            continue
        outs = connect_two_frags_keep_dummies(frags[e.i], frags[e.j], ti, tj)
        if not outs:
            outs = connect_two_frags_keep_dummies(frags[e.j], frags[e.i], tj, ti)
        if not outs:
            continue
        used = {e.i, e.j}
        rem = tuple([x for x in idxs if x not in used])
        for s in outs:
            beam_states.append((BeamMol(s, e.score), rem))
        if len(beam_states) >= beam_size:
            break

    if not beam_states:
        return []

    for _ in range(len(idxs) - 2):
        new_states: List[Tuple[BeamMol, Tuple[int, ...]]] = []
        beam_states.sort(key=lambda x: x[0].score, reverse=True)
        beam_states = beam_states[:beam_size]

        for cur, rem in beam_states:
            if not rem:
                new_states.append((cur, rem))
                continue

            used_nodes = set([x for x in idxs if x not in rem])
            local: List[Tuple[float, int, int, int]] = []
            for e in edges:
                if e.i in used_nodes and e.j in rem:
                    local.append((cur.score + e.score, e.i, e.j, e.cls))
                if e.j in used_nodes and e.i in rem:
                    local.append((cur.score + e.score, e.j, e.i, e.cls))
            if not local:
                new_states.append((cur, rem))
                continue

            local.sort(key=lambda x: x[0], reverse=True)
            taken = 0
            for sc, _i_used, j_new, cls in local[: max(beam_size * 4, 128)]:
                ti, tj = decode_typepair(cls)
                frag_new = frags[j_new]

                outs = connect_two_frags_keep_dummies(cur.smiles_with_dummies, frag_new, ti, tj)
                if not outs:
                    outs = connect_two_frags_keep_dummies(frag_new, cur.smiles_with_dummies, tj, ti)
                if not outs:
                    continue

                for s2 in outs:
                    new_rem = tuple([x for x in rem if x != j_new])
                    new_states.append((BeamMol(s2, sc), new_rem))
                taken += 1
                if taken >= beam_size:
                    break

        if not new_states:
            break
        new_states.sort(key=lambda x: x[0].score, reverse=True)
        beam_states = new_states[:beam_size]

    final_beam = [b for (b, _rem) in beam_states]
    return _finalize_beam_to_smiles(final_beam, topk=topk_out, largest_component=largest_component)


def ensure_parent_dir_for_file(path: str) -> str:
    """
    支持相对路径：
    - pred.csv                -> 保存到当前目录，不报错
    - results/pred.csv        -> 自动创建 results
    - ./results/pred.csv      -> 自动创建 .\\results
    - .\\results\\pred.csv    -> 自动创建对应目录
    返回规范化后的路径（仍可为相对路径）
    """
    path = os.path.normpath(path)

    parent = os.path.dirname(path)
    if parent and parent not in (".", ""):
        os.makedirs(parent, exist_ok=True)

    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--beam_size", type=int, default=128)
    ap.add_argument("--topk_out", type=int, default=5)
    ap.add_argument("--top_edge_m", type=int, default=2048)
    ap.add_argument("--top_type_r", type=int, default=24)
    ap.add_argument("--strict_frag", action="store_true")
    ap.add_argument("--largest_component", action="store_true")
    ap.add_argument("--out_csv", default="pred_top5.csv")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)

    frag_ids_all = npz["frag_ids"]
    mask_all = npz["mask"]
    id2s = get_id2s_from_npz(npz)

    has_dense = ("logits" in npz.files)
    has_sparse = ("topv" in npz.files and "topi" in npz.files)

    if (not has_dense) and (not has_sparse):
        raise KeyError(f"npz must contain either logits or (topv, topi). keys={npz.files}")

    if has_dense:
        logits_all = npz["logits"]
        n_total = int(logits_all.shape[0])
    else:
        topv_all = npz["topv"]
        topi_all = npz["topi"]
        n_total = int(topv_all.shape[0])

    df = pd.read_parquet(args.test)
    if "smiles_can" in df.columns:
        true_smiles = df["smiles_can"].astype(str).tolist()
    elif "smiles" in df.columns:
        true_smiles = df["smiles"].astype(str).tolist()
    else:
        raise KeyError(f"test parquet missing smiles/smiles_can. cols={list(df.columns)[:30]}")

    if len(true_smiles) != n_total:
        raise ValueError(f"test parquet rows ({len(true_smiles)}) != npz samples ({n_total})")

    top1 = 0
    top5 = 0
    tani_sum = 0.0
    empty_pred = 0
    valid_count = 0

    rows: List[Dict[str, Any]] = []

    for k in tqdm(range(n_total), desc="DeNovo(decode fast v1)"):
        frag_ids = frag_ids_all[k]
        mask = mask_all[k]
        true_c = canon(true_smiles[k])

        if has_dense:
            preds = decode_one(
                logits=logits_all[k],
                frag_ids=frag_ids,
                mask=mask,
                id2s=id2s,
                beam_size=args.beam_size,
                top_edge_m=args.top_edge_m,
                top_type_r=args.top_type_r,
                topk_out=args.topk_out,
                strict_frag=args.strict_frag,
                largest_component=args.largest_component,
            )
        else:
            preds = decode_one(
                topv=topv_all[k],
                topi=topi_all[k],
                frag_ids=frag_ids,
                mask=mask,
                id2s=id2s,
                beam_size=args.beam_size,
                top_edge_m=args.top_edge_m,
                top_type_r=args.top_type_r,
                topk_out=args.topk_out,
                strict_frag=args.strict_frag,
                largest_component=args.largest_component,
            )

        valid_preds: List[str] = []
        seen = set()
        for p in preds:
            cp = p if p in _CANON_CACHE.values() else canon(p)
            if not cp:
                continue
            if cp in seen:
                continue
            seen.add(cp)
            valid_preds.append(cp)
        valid_preds = valid_preds[: args.topk_out]

        if not valid_preds:
            empty_pred += 1
            valid_preds = ["C"] * args.topk_out
        else:
            valid_count += 1

        if valid_preds[0] == true_c:
            top1 += 1
        if true_c in valid_preds[:5]:
            top5 += 1

        best_t = 0.0
        for p in valid_preds[:5]:
            best_t = max(best_t, tanimoto(p, true_c))
        tani_sum += best_t

        rows.append({
            "true": true_c,
            "pred1": valid_preds[0],
            "pred2": valid_preds[1] if len(valid_preds) > 1 else "",
            "pred3": valid_preds[2] if len(valid_preds) > 2 else "",
            "pred4": valid_preds[3] if len(valid_preds) > 3 else "",
            "pred5": valid_preds[4] if len(valid_preds) > 4 else "",
        })

    validity_no_fallback = valid_count / max(1, n_total)

    print("\n===== FINAL DE NOVO (fast v1) =====")
    print("Test:", n_total)
    print("Validity(no-fallback):", validity_no_fallback)
    print("Validity(with-fallback):", 1.0)
    print("Top1:", top1 / max(1, n_total))
    print("Top5:", top5 / max(1, n_total))
    print("Avg best Tanimoto:", tani_sum / max(1, n_total))

    print("\n===== DIAGNOSTICS =====")
    print("Empty prediction:", empty_pred, "rate:", empty_pred / max(1, n_total))

    out_csv = ensure_parent_dir_for_file(args.out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
