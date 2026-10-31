"""Character-level data utilities for the tiny GPT.

B = batch, T = sequence length (block_size), V = vocab size
"""
import torch


class CharCodec:
    def __init__(self, text):
        self.chars = sorted(set(text))      
        self.vocab_size = len(self.chars)  
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for i,c in enumerate(self.chars)}

    def encode(self, s):
        "str -> list[int]"
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        "list[int] -> str"
        return ''.join(self.itos[i] for i in ids)


def get_batch(data, block_size, batch_size, generator=None):
    "data: 1-D LongTensor -> x, y: (B, T) LongTensors, y is x shifted by one"
    ...  # later, after CharCodec passes