# stage2_brics_model.py
from __future__ import annotations
import torch
import torch.nn as nn

NONE_CLASS = 256
N_CLASSES = 257  # 0..255 types, 256 none

class Stage2BRICSModel(nn.Module):
    def __init__(self, x_dim: int, vocab_size: int, emb_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.frag_emb = nn.Embedding(vocab_size, emb_dim)
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden + 2 * emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, xb, frag_ids, mask):
        """
        xb: [B, x_dim]
        frag_ids: [B, N] (pad=-1)
        mask: [B, N] 1/0
        returns logits [B, N, N, 257]
        """
        B, N = frag_ids.shape
        h = self.x_proj(xb)  # [B, H]

        # map pad=-1 to 0 safely by clamp, then mask it later
        fid = frag_ids.clamp(min=0)
        e = self.frag_emb(fid)  # [B,N,E]

        # build pair features
        hi = e.unsqueeze(2).expand(B, N, N, e.size(-1))
        hj = e.unsqueeze(1).expand(B, N, N, e.size(-1))
        hx = h[:, None, None, :].expand(B, N, N, h.size(-1))

        feat = torch.cat([hx, hi, hj], dim=-1)
        logits = self.pair_mlp(feat)  # [B,N,N,257]

        # mask invalid rows/cols
        m = (mask[:, :, None] * mask[:, None, :]).unsqueeze(-1)  # [B,N,N,1]
        logits = logits * m + (1.0 - m) * (-1e9)
        return logits