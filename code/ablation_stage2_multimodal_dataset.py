from __future__ import annotations
import ast
import json
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


def _safe_to_list(x):
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
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return []


def _to_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _norm_or_zero(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(vec.sum())
    if s > 0:
        vec = vec / s
    return vec.astype(np.float32)


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


def parse_h1_peaks(x, dim=128, ppm_min=0.0, ppm_max=12.0):
    vec = np.zeros((dim,), dtype=np.float32)
    peaks = _safe_to_list(x)
    if len(peaks) == 0:
        return vec

    width = (ppm_max - ppm_min) / max(dim, 1)
    for p in peaks:
        if not isinstance(p, dict):
            continue
        ppm = p.get("centroid", None) or p.get("delta (ppm)", None) or p.get("ppm", None)
        inten = (
            p.get("integral", None)
            or p.get("intensity", None)
            or p.get("max", None)
            or p.get("area", None)
            or 1.0
        )
        ppm = _to_float(ppm, None)
        inten = _to_float(inten, 1.0)
        if ppm is None:
            continue
        if ppm < ppm_min or ppm > ppm_max:
            continue
        idx = int((ppm - ppm_min) / max(width, 1e-8))
        idx = min(max(idx, 0), dim - 1)
        vec[idx] += max(float(inten), 0.0)

    return _norm_or_zero(vec)


def parse_c13_peaks(x, dim=128, ppm_min=0.0, ppm_max=220.0):
    vec = np.zeros((dim,), dtype=np.float32)
    peaks = _safe_to_list(x)
    if len(peaks) == 0:
        return vec

    width = (ppm_max - ppm_min) / max(dim, 1)
    for p in peaks:
        if not isinstance(p, dict):
            continue
        ppm = p.get("delta (ppm)", None) or p.get("centroid", None) or p.get("ppm", None)
        inten = (
            p.get("integral", None)
            or p.get("intensity", None)
            or p.get("max", None)
            or p.get("area", None)
            or 1.0
        )
        ppm = _to_float(ppm, None)
        inten = _to_float(inten, 1.0)
        if ppm is None:
            continue
        if ppm < ppm_min or ppm > ppm_max:
            continue
        idx = int((ppm - ppm_min) / max(width, 1e-8))
        idx = min(max(idx, 0), dim - 1)
        vec[idx] += max(float(inten), 0.0)

    return _norm_or_zero(vec)


def parse_hsqc_peaks(
    x,
    h_dim=16,
    c_dim=16,
    h_ppm_min=0.0,
    h_ppm_max=12.0,
    c_ppm_min=0.0,
    c_ppm_max=220.0,
):
    grid = np.zeros((h_dim, c_dim), dtype=np.float32)
    peaks = _safe_to_list(x)
    if len(peaks) == 0:
        return grid.reshape(-1)

    h_width = (h_ppm_max - h_ppm_min) / max(h_dim, 1)
    c_width = (c_ppm_max - c_ppm_min) / max(c_dim, 1)

    for p in peaks:
        if not isinstance(p, dict):
            continue

        hppm = (
            p.get("1H_centroid", None)
            or p.get("h_centroid", None)
            or p.get("1H", None)
            or p.get("h", None)
        )
        cppm = (
            p.get("13C_centroid", None)
            or p.get("c_centroid", None)
            or p.get("13C", None)
            or p.get("c", None)
        )
        inten = (
            p.get("volume", None)
            or p.get("intensity", None)
            or p.get("max", None)
            or p.get("area", None)
            or 1.0
        )

        hppm = _to_float(hppm, None)
        cppm = _to_float(cppm, None)
        inten = _to_float(inten, 1.0)

        if hppm is None or cppm is None:
            continue
        if not (h_ppm_min <= hppm <= h_ppm_max):
            continue
        if not (c_ppm_min <= cppm <= c_ppm_max):
            continue

        hi = int((hppm - h_ppm_min) / max(h_width, 1e-8))
        ci = int((cppm - c_ppm_min) / max(c_width, 1e-8))
        hi = min(max(hi, 0), h_dim - 1)
        ci = min(max(ci, 0), c_dim - 1)
        grid[hi, ci] += max(float(inten), 0.0)

    return _norm_or_zero(grid.reshape(-1))


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
        if fi == fj:
            continue
        if fi >= max_nodes or fj >= max_nodes:
            continue

        cls_ij = encode_typepair(int(t1), int(t2))
        cls_ji = encode_typepair(int(t2), int(t1))
        y[fi, fj] = cls_ij
        y[fj, fi] = cls_ji

    return y


class AblationStage2MultimodalDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        vocab_tsv: str,
        ir_len: int = 1024,
        max_nodes: int = 64,
        use_h1: bool = False,
        use_c13: bool = False,
        use_hsqc: bool = False,
        h1_dim: int = 128,
        c13_dim: int = 128,
        hsqc_h_dim: int = 16,
        hsqc_c_dim: int = 16,
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

        self.h1_dim = int(h1_dim)
        self.c13_dim = int(c13_dim)
        self.hsqc_h_dim = int(hsqc_h_dim)
        self.hsqc_c_dim = int(hsqc_c_dim)

        self.ir_col = ir_col
        self.h1_col = h1_col
        self.c13_col = c13_col
        self.hsqc_col = hsqc_col
        self.smiles_col = smiles_col

        self.x_dim = self.ir_len
        if self.use_h1:
            self.x_dim += self.h1_dim
        if self.use_c13:
            self.x_dim += self.c13_dim
        if self.use_hsqc:
            self.x_dim += self.hsqc_h_dim * self.hsqc_c_dim

    def __len__(self):
        return len(self.df)

    def _build_x(self, row):
        ir = parse_ir(row.get(self.ir_col, None), self.ir_len)
        parts = [ir]

        if self.use_h1:
            parts.append(parse_h1_peaks(row.get(self.h1_col, None), dim=self.h1_dim))
        if self.use_c13:
            parts.append(parse_c13_peaks(row.get(self.c13_col, None), dim=self.c13_dim))
        if self.use_hsqc:
            parts.append(
                parse_hsqc_peaks(
                    row.get(self.hsqc_col, None),
                    h_dim=self.hsqc_h_dim,
                    c_dim=self.hsqc_c_dim,
                )
            )

        xb = np.concatenate(parts, axis=0).astype(np.float32)
        xb = np.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        return xb

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        smi = str(row[self.smiles_col])
        xb = self._build_x(row)

        m = safe_mol(smi)

        frags_smiles = []
        if m is not None:
            frags_smiles = brics_decompose_smiles(m)

        frag_ids = np.full((self.max_nodes,), self.vocab.pad_id, dtype=np.int64)
        mask = np.zeros((self.max_nodes,), dtype=np.float32)

        if len(frags_smiles) < 2:
            use_ids = [self.vocab.unk_id, self.vocab.unk_id]
        else:
            use_ids = [self.vocab.smiles2id(s) for s in frags_smiles[: self.max_nodes]]

        n = min(len(use_ids), self.max_nodes)
        frag_ids[:n] = np.array(use_ids[:n], dtype=np.int64)
        mask[:n] = 1.0

        if m is not None:
            y = build_y_from_brics(m, self.max_nodes)
        else:
            y = np.full((self.max_nodes, self.max_nodes), NONE_CLASS, dtype=np.int64)
            np.fill_diagonal(y, NONE_CLASS)

        return (
            torch.from_numpy(xb),
            torch.from_numpy(frag_ids),
            torch.from_numpy(mask),
            torch.from_numpy(y),
        )