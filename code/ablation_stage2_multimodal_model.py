from __future__ import annotations
import torch
import torch.nn as nn

NONE_CLASS = 256
N_CLASSES = 257


class AblationStage2MultimodalModel(nn.Module):
    def __init__(self, x_dim: int, vocab_size: int, emb_dim: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.frag_emb = nn.Embedding(vocab_size, emb_dim)

        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden + 2 * emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, xb, frag_ids, mask):
        B, N = frag_ids.shape
        h = self.x_proj(xb)

        fid = frag_ids.clamp(min=0)
        e = self.frag_emb(fid)

        hi = e.unsqueeze(2).expand(B, N, N, e.size(-1))
        hj = e.unsqueeze(1).expand(B, N, N, e.size(-1))
        hx = h[:, None, None, :].expand(B, N, N, h.size(-1))

        feat = torch.cat([hx, hi, hj], dim=-1)
        logits = self.pair_mlp(feat)

        m = (mask[:, :, None] * mask[:, None, :]).unsqueeze(-1)
        logits = logits * m + (1.0 - m) * (-1e9)
        return logits