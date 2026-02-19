from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_items: int
    max_len: int = 200
    emb_dim: int = 256
    n_heads: int = 8
    n_blocks: int = 4
    dropout: float = 0.2
    ff_mult: int = 4


class LiGRBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_mult: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate1 = nn.Linear(d_model, d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = d_model * ff_mult
        self.ff1 = nn.Linear(d_model, hidden)
        self.ff2 = nn.Linear(hidden // 2, d_model)
        self.gate2 = nn.Linear(d_model, d_model)
        self.drop2 = nn.Dropout(dropout)

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.ones(sz, sz, device=device, dtype=torch.bool).triu(1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=self._causal_mask(h.size(1), h.device), key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop1(attn_out) * torch.sigmoid(self.gate1(h))

        h2 = self.ln2(x)
        ff_pre = self.ff1(h2)
        ff_a, ff_b = ff_pre.chunk(2, dim=-1)
        ff = self.ff2(F.silu(ff_a) * ff_b)
        x = x + self.drop2(ff) * torch.sigmoid(self.gate2(h2))
        return x


class ESASRec(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([LiGRBlock(cfg.emb_dim, cfg.n_heads, cfg.dropout, cfg.ff_mult) for _ in range(cfg.n_blocks)])
        self.ln_out = nn.LayerNorm(cfg.emb_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        if self.item_emb.padding_idx is not None:
            with torch.no_grad():
                self.item_emb.weight[self.item_emb.padding_idx].zero_()

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        b, l = seq.shape
        pos = torch.arange(l, device=seq.device).unsqueeze(0).expand(b, l)
        x = self.item_emb(seq) * math.sqrt(self.cfg.emb_dim) + self.pos_emb(pos)
        x = self.drop(x)
        pad_mask = seq.eq(0)
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln_out(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x

    def full_item_logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item_emb.weight[1:].t()
