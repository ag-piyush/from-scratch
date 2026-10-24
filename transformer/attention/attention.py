"""Causal multi-head self-attention, from scratch.

B = batch, T = sequence length, C = d_model, nh = n_heads, hs = C // nh
"""

import math

import torch
import torch.nn as nn

def attention(q, k, v, causal=True):
    "q, k, v: (..., T, hs) -> (..., T, hs)"
    T = q.shape[-2]
    hs = q.shape[-1]

    scores = q@k.transpose(-2,-1)
    scores = scores/hs**0.5

    if causal == True:
        scores = scores.masked_fill(torch.tril(torch.ones(T,T)) == 0, float("-inf"))

    weight = scores.softmax(dim=-1)
    out = weight@v
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x): #x: (B, T, C)
        raise NotImplementedError