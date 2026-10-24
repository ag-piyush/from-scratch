"""Tests for causal multi-head self-attention, written before the implementation.

Implement `transformer/attention/attention.py` from a blank file until these pass:

    uv run pytest transformer/attention

Done when (project tracker #1): output matches F.scaled_dot_product_attention on the
same weights, causal mask included.

The contract the tests rely on
------------------------------
attention(q, k, v, causal=True) -> Tensor
    Scaled dot-product attention for one or many heads.
    q, k, v: (..., T, hs), e.g. (B, T, hs) for one head or (B, nh, T, hs) for many.
    Returns (..., T, hs). Write the scores, scaling, mask and softmax yourself; don't
    call F.scaled_dot_product_attention (that's what you're checked against).

CausalSelfAttention(d_model, n_heads)        an nn.Module
    forward(x: (B, T, C)) -> (B, T, C), causal.
    Must have these two layers, named as in GPT-2 so its weights can be loaded later:
        c_attn = nn.Linear(C, 3 * C)   # q, k, v in that order along the last dim
        c_proj = nn.Linear(C, C)       # output projection after the heads are merged
    Head h uses channels h*hs : (h+1)*hs of each of q, k and v, where hs = C // n_heads.
    Extra keyword arguments (dropout, bias, ...) are fine if they have defaults.
    Tests run the module in eval() mode.

Suggested order, following the tests top to bottom: one head (test_attention_*),
then the module (test_module_*).
"""

import pytest
import torch
import torch.nn.functional as F

import transformer.attention.attention as impl

SHAPES = [  # (B, T, C, n_heads)
    (1, 1, 8, 1),  # a single token: the causal mask leaves exactly one key
    (2, 5, 16, 4),
    (3, 17, 32, 8),  # T not a power of two
]

TOL = {
    torch.float64: dict(atol=1e-10, rtol=1e-10),
    torch.float32: dict(atol=1e-5, rtol=1e-5),
}


# ---------------------------------------------------------------------------
# Reference: one head at a time, by slicing channels, through PyTorch's SDPA.
# Deliberately a naive loop, so it shares no reshaping code with a vectorised
# implementation and can catch head-splitting bugs.
# ---------------------------------------------------------------------------
def reference_forward(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, n_heads):
    B, T, C = x.shape
    hs = C // n_heads
    q, k, v = F.linear(x, c_attn_w, c_attn_b).split(C, dim=-1)  # each (B, T, C)
    heads = []
    for h in range(n_heads):
        sl = slice(h * hs, (h + 1) * hs)
        heads.append(
            F.scaled_dot_product_attention(q[..., sl], k[..., sl], v[..., sl], is_causal=True)
        )  # (B, T, hs)
    y = torch.cat(heads, dim=-1)  # (B, T, C)
    return F.linear(y, c_proj_w, c_proj_b)  # (B, T, C)


def make_module(C, n_heads, dtype):
    torch.manual_seed(0)
    m = impl.CausalSelfAttention(C, n_heads).to(dtype).eval()
    # Non-zero biases, so a bias that is dropped or applied twice shows up.
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim == 1:
                p.normal_(std=0.1)
    return m


def reference_for(m, x, n_heads):
    return reference_forward(x, m.c_attn.weight, m.c_attn.bias, m.c_proj.weight, m.c_proj.bias, n_heads)


# ---------------------------------------------------------------------------
# Stage 1: the attention function (one head, then a batch of heads)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("lead", [(2,), (2, 4)], ids=["one-head", "multi-head"])
def test_attention_matches_sdpa(dtype, causal, lead):
    torch.manual_seed(0)
    T, hs = 7, 16
    q, k, v = (torch.randn(*lead, T, hs, dtype=dtype) for _ in range(3))
    out = impl.attention(q, k, v, causal=causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, **TOL[dtype])


def test_attention_cannot_see_the_future():
    """Changing q/k/v at positions after t must not change the output at positions <= t."""
    torch.manual_seed(0)
    B, nh, T, hs, t = 2, 3, 9, 8, 4
    q, k, v = (torch.randn(B, nh, T, hs, dtype=torch.float64) for _ in range(3))
    q2, k2, v2 = (z.clone() for z in (q, k, v))
    for z in (q2, k2, v2):
        z[..., t + 1 :, :] = torch.randn_like(z[..., t + 1 :, :])

    a = impl.attention(q, k, v, causal=True)
    b = impl.attention(q2, k2, v2, causal=True)
    torch.testing.assert_close(a[..., : t + 1, :], b[..., : t + 1, :], atol=1e-12, rtol=0)

    # Sanity check on the test itself: without the mask, the past *does* change.
    a_nc = impl.attention(q, k, v, causal=False)
    b_nc = impl.attention(q2, k2, v2, causal=False)
    assert not torch.allclose(a_nc[..., : t + 1, :], b_nc[..., : t + 1, :])


def test_attention_is_stable_for_large_scores():
    """Scores in the hundreds overflow exp() in float32 unless softmax is computed stably."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    q, k = q * 10, k * 10  # scores q.k / sqrt(8) ~ +-100s; exp() overflows float32 past ~88
    out = impl.attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.isfinite(out).all(), "got inf/NaN: is softmax overflowing?"
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Stage 2: the multi-head module
# ---------------------------------------------------------------------------
def test_module_has_gpt2_layers():
    m = impl.CausalSelfAttention(16, 4)
    assert tuple(m.c_attn.weight.shape) == (3 * 16, 16)
    assert tuple(m.c_proj.weight.shape) == (16, 16)


def test_module_rejects_indivisible_heads():
    with pytest.raises((AssertionError, ValueError)):
        impl.CausalSelfAttention(10, 4)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("B,T,C,nh", SHAPES)
def test_module_matches_reference(dtype, B, T, C, nh):
    m = make_module(C, nh, dtype)
    x = torch.randn(B, T, C, dtype=dtype)
    with torch.no_grad():
        out = m(x)
        ref = reference_for(m, x, nh)
    assert out.shape == (B, T, C)
    torch.testing.assert_close(out, ref, **TOL[dtype])


@pytest.mark.parametrize("B,T,C,nh", SHAPES)
def test_module_gradients_match_reference(B, T, C, nh):
    """Same loss, same weights: gradients w.r.t. the input and every weight must agree."""
    m = make_module(C, nh, torch.float64)
    x = torch.randn(B, T, C, dtype=torch.float64, requires_grad=True)
    g = torch.randn(B, T, C, dtype=torch.float64)  # random upstream gradient
    params = [m.c_attn.weight, m.c_attn.bias, m.c_proj.weight, m.c_proj.bias]

    got = torch.autograd.grad((m(x) * g).sum(), [x, *params])
    want = torch.autograd.grad((reference_for(m, x, nh) * g).sum(), [x, *params])
    for name, a, b in zip(["x", "c_attn.weight", "c_attn.bias", "c_proj.weight", "c_proj.bias"], got, want):
        torch.testing.assert_close(a, b, **TOL[torch.float64], msg=f"gradient of {name}")


def test_module_cannot_see_the_future():
    m = make_module(32, 4, torch.float64)
    T, t = 10, 3
    x = torch.randn(2, T, 32, dtype=torch.float64)
    x2 = x.clone()
    x2[:, t + 1 :] = torch.randn_like(x2[:, t + 1 :])
    with torch.no_grad():
        a, b = m(x), m(x2)
    torch.testing.assert_close(a[:, : t + 1], b[:, : t + 1], atol=1e-12, rtol=0)
    assert not torch.allclose(a[:, t + 1 :], b[:, t + 1 :])
