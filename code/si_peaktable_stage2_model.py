from __future__ import annotations
import torch
import torch.nn as nn

NONE_CLASS = 256
N_CLASSES = 257


def masked_mean(x, mask):
    # x: [B,K,D], mask: [B,K]
    w = mask.unsqueeze(-1)  # [B,K,1]
    s = (x * w).sum(dim=1)
    denom = w.sum(dim=1).clamp(min=1.0)
    return s / denom


class PeakTableEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x, mask):
        # x: [B,K,D], mask: [B,K]
        h = self.net(x)
        return masked_mean(h, mask)


class SIPeakTableStage2Model(nn.Module):
    def __init__(
        self,
        ir_dim: int,
        vocab_size: int,
        h1_feat_dim: int,
        c13_feat_dim: int,
        hsqc_feat_dim: int,
        emb_dim: int = 128,
        hidden: int = 256,
        aux_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.frag_emb = nn.Embedding(vocab_size, emb_dim)

        self.ir_proj = nn.Sequential(
            nn.Linear(ir_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

        self.h1_enc = PeakTableEncoder(h1_feat_dim, aux_hidden, aux_hidden, dropout=dropout)
        self.c13_enc = PeakTableEncoder(c13_feat_dim, aux_hidden, aux_hidden, dropout=dropout)
        self.hsqc_enc = PeakTableEncoder(hsqc_feat_dim, aux_hidden, aux_hidden, dropout=dropout)

        fused_dim = hidden + aux_hidden * 3
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden + 2 * emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, ir, h1_x, h1_m, c13_x, c13_m, hsqc_x, hsqc_m, frag_ids, mask):
        B, N = frag_ids.shape

        ir_h = self.ir_proj(ir)
        h1_h = self.h1_enc(h1_x, h1_m)
        c13_h = self.c13_enc(c13_x, c13_m)
        hsqc_h = self.hsqc_enc(hsqc_x, hsqc_m)

        fused = self.fuse(torch.cat([ir_h, h1_h, c13_h, hsqc_h], dim=-1))

        fid = frag_ids.clamp(min=0)
        e = self.frag_emb(fid)

        hi = e.unsqueeze(2).expand(B, N, N, e.size(-1))
        hj = e.unsqueeze(1).expand(B, N, N, e.size(-1))
        hx = fused[:, None, None, :].expand(B, N, N, fused.size(-1))

        feat = torch.cat([hx, hi, hj], dim=-1)
        logits = self.pair_mlp(feat)

        m = (mask[:, :, None] * mask[:, None, :]).unsqueeze(-1)
        logits = logits * m + (1.0 - m) * (-1e9)
        return logits