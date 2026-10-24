# Causal multi-head attention

**What it is:** causal multi-head self-attention, the core of a decoder-only transformer. `attention(q, k, v)` is scaled dot-product attention written out step by step (scores, scale, causal mask, softmax, weighted sum of values). `CausalSelfAttention` wraps it with GPT-2's layer names (`c_attn`, `c_proj`) so pretrained GPT-2 weights load straight in. Heads are computed in a loop over channel slices for now.

**Checked against:** `torch.nn.functional.scaled_dot_product_attention` with the same weights and causal mask: outputs and gradients, in float32 and float64, for one head and many, and a check that changing future tokens never changes earlier outputs.

```bash
uv run pytest transformer/attention
```

**Result:** max abs diff vs SDPA at GPT-2 small size (C = 768, 12 heads, B = 4, T = 64): 2.4e-7 in float32, 3.9e-16 in float64. Both are rounding error.

**Gotcha:** I read the sequence length as `T = q.shape[0]`. That worked on a 2-D `(T, hs)` test tensor, but with real inputs of shape `(B, nh, T, hs)`, dim 0 is the batch, so the causal mask came out 2×2 against 6×6 scores. Leading dims change between one head and many; only the last two have a fixed meaning. So read from the end: `T = q.shape[-2]`, `hs = q.shape[-1]`.

**References:** Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017); Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [nanoGPT](https://github.com/karpathy/nanoGPT); PyTorch docs for [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).