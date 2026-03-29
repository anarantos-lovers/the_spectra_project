# stage2_brics_dataset.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import BRICS

from brics_fragment_library import BricsFragVocab

NONE_CLASS = 256  # 0..255 typepair, 256 none
N_CLASSES = 257

# ---------- typepair ----------
def encode_typepair(ti: int, tj: int) -> int:
    # ti,tj in 1..16 -> cls 0..255
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

def parse_ir(x, ir_len: int):
    # 不截断“信息源”，但训练必须固定长度 -> 这里固定 ir_len
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

    return arr.astype(np.float32)

def brics_decompose_smiles(m: Chem.Mol):
    """
    返回 fragment smiles 列表（带 dummy [n*]）
    """
    try:
        frags = BRICS.BRICSDecompose(m, returnMols=False)
        frags_smiles = sorted(list(frags))
        return frags_smiles
    except Exception:
        return []

def build_atom_to_fragid_by_brics_bonds(m: Chem.Mol):
    """
    用 BRICS bond 切断，得到 atom_idx -> frag_id 映射
    """
    # 1) 找 BRICS bonds（原子对）
    brics_bonds = []
    try:
        for (a1, a2), (t1, t2) in BRICS.FindBRICSBonds(m):
            brics_bonds.append((int(a1), int(a2), int(t1), int(t2)))
    except Exception:
        brics_bonds = []

    if not brics_bonds:
        return None, []  # 没 BRICS bond

    # 2) 找到这些 bond 在 mol 里的 bond index
    bond_ids = []
    for a1, a2, _t1, _t2 in brics_bonds:
        b = m.GetBondBetweenAtoms(int(a1), int(a2))
        if b is None:
            continue
        bond_ids.append(int(b.GetIdx()))

    if not bond_ids:
        return None, []

    # 3) 在这些 bond 上切断（不加 dummy 也行，只要能得到 fragment 划分）
    try:
        m2 = Chem.FragmentOnBonds(m, bond_ids, addDummies=False)
    except Exception:
        return None, []

    # 4) atom->frag_id
    # frags: tuple(tuple(atom_idx...))
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
    """
    构造 y: [max_nodes, max_nodes]，只给 fragment-level BRICS edge 标注
    """
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

class Stage2BRICSDataset(Dataset):
    """
    输出：
      xb: [ir_len]
      frag_ids: [max_nodes]
      mask: [max_nodes]
      y: [max_nodes,max_nodes]
    """

    def __init__(self, parquet_path: str, vocab_tsv: str, ir_len: int = 1024, max_nodes: int = 24):
        super().__init__()
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.ir_len = int(ir_len)
        self.max_nodes = int(max_nodes)
        self.vocab = BricsFragVocab(vocab_tsv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        smi = str(row["smiles"])
        ir = parse_ir(row["ir_spectra"], self.ir_len)

        m = safe_mol(smi)

        # ---- fragments (for node ids) ----
        frags_smiles = []
        if m is not None:
            frags_smiles = brics_decompose_smiles(m)

        frag_ids = np.full((self.max_nodes,), self.vocab.pad_id, dtype=np.int64)
        mask = np.zeros((self.max_nodes,), dtype=np.float32)

        # 保底：如果 BRICS 分解失败 / 只有 0/1 个 fragment，至少给两个 UNK 节点
        if len(frags_smiles) < 2:
            use_ids = [self.vocab.unk_id, self.vocab.unk_id]
        else:
            use_ids = [self.vocab.smiles2id(s) for s in frags_smiles[: self.max_nodes]]

        n = min(len(use_ids), self.max_nodes)
        frag_ids[:n] = np.array(use_ids[:n], dtype=np.int64)
        mask[:n] = 1.0

        # ---- y labels (real BRICS fragment graph) ----
        if m is not None:
            y = build_y_from_brics(m, self.max_nodes)
        else:
            y = np.full((self.max_nodes, self.max_nodes), NONE_CLASS, dtype=np.int64)
            np.fill_diagonal(y, NONE_CLASS)

        return (
            torch.from_numpy(ir),
            torch.from_numpy(frag_ids),
            torch.from_numpy(mask),
            torch.from_numpy(y),
        )