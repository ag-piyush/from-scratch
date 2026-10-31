"""Tiny char-level GPT.

B = batch, T = sequence length, C = n_embd
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.attention.attention import CausalSelfAttention

import math

from dataclasses import dataclass

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()                      # exact form (default), not 'tanh'
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):    # x: (B, T, C)
        x = self.c_fc(x)     # (B, T, 4C)
        x = self.gelu(x)     # (B, T, 4C)
        x = self.c_proj(x)   # (B, T, C)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):                   # x: (B, T, C)
        x = x + self.attn(self.ln_1(x))     # (B, T, C)  communicate: tokens read from earlier tokens
        x = x + self.mlp(self.ln_2(x))      # (B, T, C)  compute: each token processes what it gathered
        return x

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 3a: self.transformer = nn.ModuleDict(dict(wte=..., wpe=..., h=..., ln_f=...))
        # 3a: self.lm_head = ..., tied to wte
        # 3b: init

    def forward(self, idx, targets=None):   # idx: (B, T) long
        ...
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        ...