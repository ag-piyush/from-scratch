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
        C = config.n_embd
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, C),   # (V, C)
            wpe = nn.Embedding(config.block_size, C),   # (block_size, C)
            h = nn.ModuleList([Block(C, config.n_head, config.dropout) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(C),
        ))
        self.lm_head = nn.Linear(C, config.vocab_size, bias=False)  # weight: (V, C)
        self.lm_head.weight = self.transformer.wte.weight           # tie: same tensor
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("attn.c_proj.weight", "mlp.c_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):   # idx: (B, T) long
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)                    # (T,)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)   # (B, T, C) + (T, C) -> (B, T, C)
        for block in self.transformer.h:
            x = block(x)                                            # (B, T, C)
        x = self.transformer.ln_f(x)                                # (B, T, C)
        logits = self.lm_head(x)                                    # (B, T, V)

        loss = None
        if targets is not None:
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):   # idx: (B, T) long
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]           # (B, ≤ block_size)
            logits, _ = self(idx_cond)                            # (B, T', V)
            logits = logits[:, -1, :]                             # (B, V)
            probs = F.softmax(logits, dim=-1)                     # (B, V)
            next_id = torch.multinomial(probs, num_samples=1)     # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)                # (B, T + 1)
        return idx
