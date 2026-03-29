# brics_fragment_library.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

@dataclass
class BricsFragVocab:
    """
    TSV 格式：smiles \t count
    - 自动跳过空行 / 注释行
    - 自动在 id=0 放 PAD，在 id=1 放 UNK
    """
    path: str
    smiles_list: List[str]
    counts: List[int]
    s2id: Dict[str, int]

    def __init__(self, tsv_path: str):
        self.path = tsv_path
        smiles: List[str] = [PAD_TOKEN, UNK_TOKEN]
        counts: List[int] = [0, 0]

        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) == 1:
                    s = parts[0].strip()
                    c = 1
                else:
                    s = parts[0].strip()
                    try:
                        c = int(parts[1])
                    except Exception:
                        c = 1
                if not s:
                    continue
                smiles.append(s)
                counts.append(c)

        self.smiles_list = smiles
        self.counts = counts
        self.s2id = {s: i for i, s in enumerate(self.smiles_list)}

    def __len__(self) -> int:
        return len(self.smiles_list)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def smiles2id(self, s: str) -> int:
        if s in self.s2id:
            return int(self.s2id[s])
        return self.unk_id

    # ---- 兼容取 smiles 的各种写法 ----
    def id2smiles(self, i: int) -> str:
        i = int(i)
        if 0 <= i < len(self.smiles_list):
            s = self.smiles_list[i]
            return "" if s in (PAD_TOKEN,) else s
        return ""

    def id2s(self, i: int) -> str:
        return self.id2smiles(i)

    def __getitem__(self, i: int) -> str:
        return self.id2smiles(i)