from __future__ import annotations
import ast
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import BRICS

from brics_fragment_library import BricsFragVocab

NONE_CLASS = 256
N_CLASSES = 257


def encode_typepair(ti: int, tj: int) -> int:
    return (tj - 1) * 16 + (ti - 1)


def safe_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canon_smiles(s: str):
    m = safe_mol(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


def repair_peak_string(s: str) -> str:
    s = s.strip()
    s = s.replace("\r\n", "\n")
    s = re.sub(r"}\s*\n\s*{", "}, {", s)
    s = re.sub(r"}\s*{", "}, {", s)
    return s


def parse_list_like(x):
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        s2 = repair_peak_string(s)
        try:
            obj = json.loads(s2)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s2)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return []


def to_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def norm_range(v, vmin, vmax):
    if vmax <= vmin:
        return 0.0
    return clip01((float(v) - vmin) / (vmax - vmin))


def parse_ir(x, ir_len: int):
    if isinstance(x, str):
        s = x.strip().replace("\n", " ").replace(",", " ").replace("[", " ").replace("]", " ")
        arr = np.fromstring(s, sep=" ", dtype=np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)

    if arr.size == 0:
        arr = np.zeros((ir_len,), dtype=np.float32)

    if arr.size >= ir_len:
        arr = arr[:ir_len]
    else:
        arr = np.concatenate([arr, np.zeros((ir_len - arr.size,), dtype=np.float32)], axis=0)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


H1_CATS = ["s", "d", "t", "q", "m", "dd", "dt", "br", "other"]


def h1_category_onehot(cat: str):
    cat = str(cat).strip().lower()
    if cat not in H1_CATS:
        cat = "other"
    vec = np.zeros((len(H1_CATS),), dtype=np.float32)
    vec[H1_CATS.index(cat)] = 1.0
    return vec


def summarize_j_values(v):
    vals = parse_list_like(v)
    nums = []
    for x in vals:
        try:
            nums.append(float(x))
        except Exception:
            continue
    if len(nums) == 0:
        return 0.0, 0.0, 0.0
    arr = np.asarray(nums, dtype=np.float32)
    return min(len(arr) / 4.0, 1.0), clip01(arr.mean() / 20.0), clip01(arr.max() / 20.0)


def encode_h1_peak(p: dict):
    centroid = to_float(p.get("centroid", p.get("delta (ppm)", p.get("ppm", 0.0))), 0.0)
    intensity = to_float(p.get("integral", p.get("intensity", p.get("max", p.get("area", 1.0)))), 1.0)
    nH = to_float(p.get("nH", p.get("n_h", intensity)), intensity)
    rmin = to_float(p.get("rangeMin", p.get("range_min", centroid)), centroid)
    rmax = to_float(p.get("rangeMax", p.get("range_max", centroid)), centroid)
    width = max(0.0, rmax - rmin)

    j_count, j_mean, j_max = summarize_j_values(p.get("j_values", []))
    cat_oh = h1_category_onehot(p.get("category", "other"))

    feat = np.concatenate([
        np.array([
            norm_range(centroid, 0.0, 12.0),
            clip01(intensity / 10.0),
            clip01(nH / 6.0),
            clip01(width / 2.0),
            j_count,
            j_mean,
            j_max,
        ], dtype=np.float32),
        cat_oh.astype(np.float32),
    ])
    return feat.astype(np.float32)


def encode_c13_peak(p: dict):
    delta = to_float(p.get("delta (ppm)", p.get("centroid", p.get("ppm", 0.0))), 0.0)
    integral = to_float(p.get("integral", 1.0), 1.0)
    intensity = to_float(p.get("intensity", p.get("max", 1.0)), 1.0)
    width = to_float(p.get("width", 0.0), 0.0)

    feat = np.array([
        norm_range(delta, 0.0, 220.0),
        clip01(integral / 10.0),
        clip01(intensity / 10.0),
        clip01(width / 10.0),
    ], dtype=np.float32)
    return feat


def encode_hsqc_peak(p: dict):
    h_cent = to_float(p.get("1H_centroid", p.get("h_centroid", p.get("1H", p.get("h", 0.0)))), 0.0)
    c_cent = to_float(p.get("13C_centroid", p.get("c_centroid", p.get("13C", p.get("c", 0.0)))), 0.0)

    h_min = to_float(p.get("1H_min", p.get("h_min", h_cent)), h_cent)
    h_max = to_float(p.get("1H_max", p.get("h_max", h_cent)), h_cent)
    c_min = to_float(p.get("13C_min", p.get("c_min", c_cent)), c_cent)
    c_max = to_float(p.get("13C_max", p.get("c_max", c_cent)), c_cent)

    h_width = max(0.0, h_max - h_min)
    c_width = max(0.0, c_max - c_min)
    nH = to_float(p.get("nH", p.get("n_h", 1.0)), 1.0)

    feat = np.array([
        norm_range(h_cent, 0.0, 12.0),
        norm_range(c_cent, 0.0, 220.0),
        clip01(h_width / 2.0),
        clip01(c_width / 20.0),
        clip01(nH / 6.0),
    ], dtype=np.float32)
    return feat


def pack_peak_table(peaks, encode_fn, max_peaks: int, feat_dim: int):
    x = np.zeros((max_peaks, feat_dim), dtype=np.float32)
    m = np.zeros((max_peaks,), dtype=np.float32)
    peaks = parse_list_like(peaks)

    if len(peaks) == 0:
        return x, m

    valid = []
    for p in peaks:
        if isinstance(p, dict):
            valid.append(p)

    n = min(len(valid), max_peaks)
    for i in range(n):
        x[i] = encode_fn(valid[i])
        m[i] = 1.0
    return x, m


def brics_decompose_smiles(m: Chem.Mol):
    try:
        frags = BRICS.BRICSDecompose(m, returnMols=False)
        return sorted(list(frags))
    except Exception:
        return []


def build_atom_to_fragid_by_brics_bonds(m: Chem.Mol):
    brics_bonds = []
    try:
        for (a1, a2), (t1, t2) in BRICS.FindBRICSBonds(m):
            brics_bonds.append((int(a1), int(a2), int(t1), int(t2)))
    except Exception:
        brics_bonds = []

    if not brics_bonds:
        return None, []

    bond_ids = []
    for a1, a2, _t1, _t2 in brics_bonds:
        b = m.GetBondBetweenAtoms(int(a1), int(a2))
        if b is None:
            continue
        bond_ids.append(int(b.GetIdx()))

    if not bond_ids:
        return None, []

    try:
        m2 = Chem.FragmentOnBonds(m, bond_ids, addDummies=False)
    except Exception:
        return None, []

    try:
        frags = Chem.GetMolFrags(m2, asMols=False, sanitizeFrags=False)
    except Exception:
        return None, []

    atom2frag = {}
    for fid, atoms in enumerate(frags):
        for a in atoms:
            atom2frag[int(a)] = int(fid)
    return atom2frag, brics_bonds


def build_y_from_brics(m: Chem.Mol, max_nodes: int):
    y = np.full((max_nodes, max_nodes), NONE_CLASS, dtype=np.int64)
    np.fill_diagonal(y, NONE_CLASS)

    atom2frag, bonds = build_atom_to_fragid_by_brics_bonds(m)
    if atom2frag is None or not bonds:
        return y

    for a1, a2, t1, t2 in bonds:
        if a1 not in atom2frag or a2 not in atom2frag:
            continue
        fi = atom2frag[a1]
        fj = atom2frag[a2]
        if fi == fj or fi >= max_nodes or fj >= max_nodes:
            continue

        y[fi, fj] = encode_typepair(int(t1), int(t2))
        y[fj, fi] = encode_typepair(int(t2), int(t1))
    return y


class SIPeakTableStage2Dataset(Dataset):
    H1_FEAT_DIM = 7 + len(H1_CATS)
    C13_FEAT_DIM = 4
    HSQC_FEAT_DIM = 5

    def __init__(
        self,
        parquet_path: str,
        vocab_tsv: str,
        ir_len: int = 1024,
        max_nodes: int = 64,
        use_h1: bool = True,
        use_c13: bool = True,
        use_hsqc: bool = True,
        max_h1_peaks: int = 32,
        max_c13_peaks: int = 32,
        max_hsqc_peaks: int = 64,
        ir_col: str = "ir_spectra",
        h1_col: str = "h_nmr_peaks",
        c13_col: str = "c_nmr_peaks",
        hsqc_col: str = "hsqc_nmr_peaks",
        smiles_col: str = "smiles",
    ):
        super().__init__()
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.ir_len = int(ir_len)
        self.max_nodes = int(max_nodes)
        self.vocab = BricsFragVocab(vocab_tsv)

        self.use_h1 = bool(use_h1)
        self.use_c13 = bool(use_c13)
        self.use_hsqc = bool(use_hsqc)

        self.max_h1_peaks = int(max_h1_peaks)
        self.max_c13_peaks = int(max_c13_peaks)
        self.max_hsqc_peaks = int(max_hsqc_peaks)

        self.ir_col = ir_col
        self.h1_col = h1_col
        self.c13_col = c13_col
        self.hsqc_col = hsqc_col
        self.smiles_col = smiles_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        smi = str(row[self.smiles_col])

        ir = parse_ir(row.get(self.ir_col, None), self.ir_len)

        if self.use_h1:
            h1_x, h1_m = pack_peak_table(
                row.get(self.h1_col, None),
                encode_h1_peak,
                self.max_h1_peaks,
                self.H1_FEAT_DIM,
            )
        else:
            h1_x = np.zeros((self.max_h1_peaks, self.H1_FEAT_DIM), dtype=np.float32)
            h1_m = np.zeros((self.max_h1_peaks,), dtype=np.float32)

        if self.use_c13:
            c13_x, c13_m = pack_peak_table(
                row.get(self.c13_col, None),
                encode_c13_peak,
                self.max_c13_peaks,
                self.C13_FEAT_DIM,
            )
        else:
            c13_x = np.zeros((self.max_c13_peaks, self.C13_FEAT_DIM), dtype=np.float32)
            c13_m = np.zeros((self.max_c13_peaks,), dtype=np.float32)

        if self.use_hsqc:
            hsqc_x, hsqc_m = pack_peak_table(
                row.get(self.hsqc_col, None),
                encode_hsqc_peak,
                self.max_hsqc_peaks,
                self.HSQC_FEAT_DIM,
            )
        else:
            hsqc_x = np.zeros((self.max_hsqc_peaks, self.HSQC_FEAT_DIM), dtype=np.float32)
            hsqc_m = np.zeros((self.max_hsqc_peaks,), dtype=np.float32)

        m = safe_mol(smi)

        frags_smiles = brics_decompose_smiles(m) if m is not None else []
        frag_ids = np.full((self.max_nodes,), self.vocab.pad_id, dtype=np.int64)
        mask = np.zeros((self.max_nodes,), dtype=np.float32)

        if len(frags_smiles) < 2:
            use_ids = [self.vocab.unk_id, self.vocab.unk_id]
        else:
            use_ids = [self.vocab.smiles2id(s) for s in frags_smiles[: self.max_nodes]]

        n = min(len(use_ids), self.max_nodes)
        frag_ids[:n] = np.asarray(use_ids[:n], dtype=np.int64)
        mask[:n] = 1.0

        if m is not None:
            y = build_y_from_brics(m, self.max_nodes)
        else:
            y = np.full((self.max_nodes, self.max_nodes), NONE_CLASS, dtype=np.int64)
            np.fill_diagonal(y, NONE_CLASS)

        return (
            torch.from_numpy(ir),
            torch.from_numpy(h1_x),
            torch.from_numpy(h1_m),
            torch.from_numpy(c13_x),
            torch.from_numpy(c13_m),
            torch.from_numpy(hsqc_x),
            torch.from_numpy(hsqc_m),
            torch.from_numpy(frag_ids),
            torch.from_numpy(mask),
            torch.from_numpy(y),
        )