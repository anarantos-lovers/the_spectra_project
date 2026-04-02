from __future__ import annotations
import torch
import torch.nn as nn

NONE_CLASS = 256
N_CLASSES = 257


class AblationTransformerStage2Model(nn.Module):
    def __init__(
        self,
        x_dim: int,
        vocab_size: int,
        max_nodes: int = 64,
        hidden: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden = hidden

        self.frag_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_nodes, hidden)

        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, xb, frag_ids, mask):
        """
        xb: [B, x_dim]
        frag_ids: [B, N]
        mask: [B, N]
        returns logits [B, N, N, 257]
        """
        B, N = frag_ids.shape
        device = frag_ids.device

        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        h_ir = self.x_proj(xb)                      # [B,H]
        tok = self.frag_emb(frag_ids.clamp(min=0)) # [B,N,H]
        tok = tok + self.pos_emb(pos) + h_ir[:, None, :]

        pad_mask = (mask <= 0.5)                   # True means pad
        ctx = self.encoder(tok, src_key_padding_mask=pad_mask)  # [B,N,H]

        hi = ctx.unsqueeze(2).expand(B, N, N, self.hidden)
        hj = ctx.unsqueeze(1).expand(B, N, N, self.hidden)
        hx = h_ir[:, None, None, :].expand(B, N, N, self.hidden)

        feat = torch.cat([hx, hi, hj], dim=-1)
        logits = self.pair_mlp(feat)

        m = (mask[:, :, None] * mask[:, None, :]).unsqueeze(-1)
        logits = logits * m + (1.0 - m) * (-1e9)
        return logits