"""Tests for the tiny char-level GPT, written before the implementation.

Implement the two files below from a blank file until these pass:

    uv run pytest transformer/gpt

    transformer/gpt/data.py    CharCodec, get_batch
    transformer/gpt/gpt.py     GPTConfig, MLP, Block, GPT

Done when (project tracker #2): trains on the M4, loss falls steadily, samples are
recognisably text. The tests below are the CPU-sized gates on the way there; the M4 run
at the bottom of this docstring is the real finish line.

Build in this order (each step ends at a test group below, top to bottom)
-------------------------------------------------------------------------
0. data.py: look at the text first. Vocab size, round-trip a sample, check y is x shifted by one.
1. The numbers to know BEFORE training (tiny Shakespeare, 90/10 char split, V = 65):
       uniform (loss at init)    ln(65)                 = 4.174
       unigram (train / val)     3.309 / 3.347
       bigram, add-one (val)     2.482
       trigram, add-0.1 (val)    2.046
   Your untrained model must sit at ~4.17; "it learned something" means it beats the bigram.
2. MLP, then Block (pre-norm, residual), reusing YOUR CausalSelfAttention from
   transformer/attention/attention.py. Then GPT: wte + wpe -> blocks -> ln_f -> lm_head.
3. Init (GPT-2 scheme), loss at init, overfit one batch. Do NOT touch the real dataset until the
   overfit test passes: a model that cannot memorise 64 tokens has a bug, not a tuning problem.
4. Only then: the M4 run.

The contract the tests rely on
------------------------------
CharCodec(text)                      vocab = sorted(set(text)), so ids are deterministic
    .vocab_size -> int
    .encode(s: str) -> list[int]
    .decode(ids: list[int]) -> str   (ids may also come from tensor.tolist())

get_batch(data, block_size, batch_size, generator=None) -> (x, y)
    data: 1-D LongTensor of token ids. Returns two (batch_size, block_size) LongTensors,
    x = data[i : i+T] and y = data[i+1 : i+T+1] for a random start i per row. Every valid
    window must be reachable, including the very last one. Sample starts with
    torch.randint(..., generator=generator).

GPTConfig(block_size, vocab_size, n_layer, n_head, n_embd, dropout=0.0)    a dataclass

MLP(n_embd)                          an nn.Module
    c_fc = nn.Linear(C, 4C), nn.GELU() (the exact one, not tanh), c_proj = nn.Linear(4C, C)

Block(n_embd, n_head, dropout=0.0)   an nn.Module, forward (B, T, C) -> (B, T, C)
    Pre-norm, GPT-2 layer names:
        x = x + attn(ln_1(x));  x = x + mlp(ln_2(x))
    ln_1, ln_2 = nn.LayerNorm(C); attn = your CausalSelfAttention; mlp = MLP.

GPT(config)                          an nn.Module
    forward(idx: (B, T) long, targets: (B, T) long | None) -> (logits, loss)
        logits (B, T, V); loss is None without targets, else the mean cross-entropy over
        all B*T positions. targets are ALREADY shifted (get_batch does that), so forward
        must not shift again. T > block_size must raise (AssertionError or ValueError).
    generate(idx: (B, T), max_new_tokens: int) -> (B, T + max_new_tokens)
        Samples from softmax(logits[:, -1]) one token at a time, feeding the context back
        in, cropped to the last block_size tokens. Plain multinomial; temperature, top-k,
        top-p and beam search are tracker #4.
    Layer names (GPT-2's, so its weights can be loaded later):
        self.transformer = nn.ModuleDict(dict(wte=..., wpe=..., h=nn.ModuleList(...), ln_f=...))
        self.lm_head = nn.Linear(C, V, bias=False), weight tied to transformer.wte.weight
    Init (GPT-2): every Linear and Embedding weight ~ N(0, 0.02), Linear biases 0, and the
    two residual projections in each block (attn.c_proj, mlp.c_proj) scaled down by
    1/sqrt(2 * n_layer) so the residual stream's variance does not grow with depth.

The M4 run (train.py, not tested in CI)
---------------------------------------
Device: "mps" if torch.backends.mps.is_available() else "cpu". Load tiny Shakespeare
(karpathy/char-rnn, data/tinyshakespeare/input.txt, ~1.1 MB; keep it in a git-ignored data/
folder), 90/10 train/val split by characters, fixed seed, AdamW, evaluate train and val
loss every 100 steps over ~50 batches each (print 4 significant digits), then sample ~500
characters from a "\\n" prompt at the end.
    baby   : n_layer=4, n_head=4, n_embd=128, block_size=64,  batch=32, lr=1e-3, dropout=0
    bigger : n_layer=6, n_head=6, n_embd=384, block_size=256, batch=64, lr=1e-3, dropout=0.2
Run baby first. Loss falling smoothly from ~4.2 to below the bigram's 2.48 within the first
few hundred steps is the sign the whole stack (data, shift, mask, init, optimiser) is right.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.gpt.data as data
import transformer.gpt.gpt as impl

TOL = {
    torch.float64: dict(atol=1e-10, rtol=1e-10),
    torch.float32: dict(atol=1e-5, rtol=1e-5),
}


def make_config(**kw):
    base = dict(block_size=16, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return impl.GPTConfig(**base)


# ---------------------------------------------------------------------------
# Stage 0: data
# ---------------------------------------------------------------------------
def test_codec_round_trip():
    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    codec = data.CharCodec(text)
    assert codec.vocab_size == len(set(text))
    ids = codec.encode(text)
    assert all(isinstance(i, int) for i in ids)
    assert codec.decode(ids) == text
    assert codec.decode(torch.tensor(ids).tolist()) == text


def test_codec_ids_are_sorted_and_deterministic():
    codec = data.CharCodec("cabbage")  # sorted unique chars: a b c e g
    assert codec.vocab_size == 5
    assert codec.encode("bag") == [1, 0, 4]
    assert codec.decode([2, 3]) == "ce"


def test_get_batch_shapes_and_shift():
    d = torch.arange(1000)  # token i has id i, so "y is x shifted by one" means y == x + 1
    x, y = data.get_batch(d, block_size=8, batch_size=5, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (5, 8)
    assert x.dtype == y.dtype == torch.long
    assert torch.equal(y, x + 1)
    assert torch.equal(x, x[:, :1] + torch.arange(8))  # each row is a contiguous window


def test_get_batch_reaches_the_last_window_and_never_overruns():
    T = 8
    d = torch.arange(T + 1)  # exactly one valid window: x = 0..T-1, y = 1..T
    x, y = data.get_batch(d, block_size=T, batch_size=3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(x, torch.arange(T).expand(3, T))
    assert torch.equal(y, torch.arange(1, T + 1).expand(3, T))

    d = torch.arange(50)
    g = torch.Generator().manual_seed(0)
    starts = set()
    for _ in range(200):
        x, y = data.get_batch(d, block_size=T, batch_size=16, generator=g)
        assert y.max() <= 49
        starts.update(x[:, 0].tolist())
    assert max(starts) == 50 - T - 1, "the last valid window is never sampled (off-by-one?)"
    assert min(starts) == 0


def test_get_batch_is_reproducible_with_a_generator():
    d = torch.arange(1000)
    a = data.get_batch(d, 8, 4, generator=torch.Generator().manual_seed(7))
    b = data.get_batch(d, 8, 4, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------------------------------------------------------------------------
# Stage 1: MLP and Block, checked against PyTorch's pre-norm encoder layer
# ---------------------------------------------------------------------------
def test_mlp_layers():
    m = impl.MLP(16)
    assert tuple(m.c_fc.weight.shape) == (64, 16)
    assert tuple(m.c_proj.weight.shape) == (16, 64)
    x = torch.randn(2, 5, 16)
    want = m.c_proj(F.gelu(m.c_fc(x)))  # F.gelu defaults to the exact erf form, not the tanh one
    assert m(x).shape == (2, 5, 16)
    torch.testing.assert_close(m(x), want)


def make_block(C, nh, dtype):
    torch.manual_seed(0)
    b = impl.Block(C, nh).to(dtype)
    # Non-trivial everything, so a dropped bias, a missing LayerNorm affine or a wrong
    # order of operations shows up.
    with torch.no_grad():
        for p in b.parameters():
            p.normal_(std=0.2 if p.ndim == 1 else 0.1)
    return b


def reference_block(b, C, nh, dtype):
    """nn.TransformerEncoderLayer(norm_first=True) is exactly: x + attn(ln1(x)); x + ff(ln2(x))."""
    ref = nn.TransformerEncoderLayer(
        d_model=C, nhead=nh, dim_feedforward=4 * C, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    ).to(dtype)
    with torch.no_grad():
        ref.self_attn.in_proj_weight.copy_(b.attn.c_attn.weight)  # q, k, v stacked, same as c_attn
        ref.self_attn.in_proj_bias.copy_(b.attn.c_attn.bias)
        ref.self_attn.out_proj.weight.copy_(b.attn.c_proj.weight)
        ref.self_attn.out_proj.bias.copy_(b.attn.c_proj.bias)
        ref.linear1.weight.copy_(b.mlp.c_fc.weight)
        ref.linear1.bias.copy_(b.mlp.c_fc.bias)
        ref.linear2.weight.copy_(b.mlp.c_proj.weight)
        ref.linear2.bias.copy_(b.mlp.c_proj.bias)
        ref.norm1.weight.copy_(b.ln_1.weight)
        ref.norm1.bias.copy_(b.ln_1.bias)
        ref.norm2.weight.copy_(b.ln_2.weight)
        ref.norm2.bias.copy_(b.ln_2.bias)
    return ref


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("B,T,C,nh", [(1, 1, 8, 1), (2, 5, 16, 4), (3, 17, 32, 8)])
def test_block_matches_reference(dtype, B, T, C, nh):
    b = make_block(C, nh, dtype).eval()
    ref = reference_block(b, C, nh, dtype).eval()
    x = torch.randn(B, T, C, dtype=dtype)
    mask = nn.Transformer.generate_square_subsequent_mask(T, dtype=dtype)
    out = b(x)
    want = ref(x, src_mask=mask, is_causal=True)
    assert out.shape == (B, T, C)
    torch.testing.assert_close(out, want, **TOL[dtype])


def test_block_is_the_identity_when_the_sublayers_write_nothing():
    """The residual connections: zero the two output projections and x must pass straight through."""
    b = impl.Block(16, 4).eval()
    with torch.no_grad():
        for proj in (b.attn.c_proj, b.mlp.c_proj):
            proj.weight.zero_()
            proj.bias.zero_()
    x = torch.randn(2, 6, 16)
    torch.testing.assert_close(b(x), x)


# ---------------------------------------------------------------------------
# Stage 2: the model's structure
# ---------------------------------------------------------------------------
def test_gpt2_layer_names():
    keys = set(impl.GPT(make_config()).state_dict().keys())
    expected = {
        "transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight",
        "transformer.ln_f.weight", "transformer.ln_f.bias",
    }
    for i in range(2):
        for name in ["ln_1", "ln_2", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
            expected.add(f"transformer.h.{i}.{name}.weight")
            if name.startswith(("attn", "mlp")):
                expected.add(f"transformer.h.{i}.{name}.bias")
        expected.add(f"transformer.h.{i}.ln_1.bias")
        expected.add(f"transformer.h.{i}.ln_2.bias")
    missing = expected - keys
    assert not missing, f"missing parameters: {sorted(missing)}"


def test_lm_head_is_tied_to_the_embedding():
    m = impl.GPT(make_config())
    assert m.lm_head.weight is m.transformer.wte.weight


def test_parameter_count():
    """Counting by hand is the cheapest way to find a stray bias, a missing LayerNorm or an untied head."""
    cfg = make_config()
    V, T, C, L = cfg.vocab_size, cfg.block_size, cfg.n_embd, cfg.n_layer
    per_block = (
        2 * C                      # ln_1
        + 3 * C * C + 3 * C        # attn.c_attn
        + C * C + C                # attn.c_proj
        + 2 * C                    # ln_2
        + 4 * C * C + 4 * C        # mlp.c_fc
        + 4 * C * C + C            # mlp.c_proj
    )
    expected = V * C + T * C + L * per_block + 2 * C  # wte (shared with lm_head) + wpe + blocks + ln_f
    got = sum(p.numel() for p in impl.GPT(cfg).parameters())
    assert got == expected


def test_forward_shapes_and_loss_contract():
    cfg = make_config()
    m = impl.GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 10))
    logits, loss = m(idx)
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert loss is None
    logits, loss = m(idx, idx)
    assert loss.ndim == 0 and torch.isfinite(loss)
    want = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), idx.reshape(-1))
    torch.testing.assert_close(loss, want)


def test_sequence_longer_than_block_size_is_rejected():
    cfg = make_config()
    with pytest.raises((AssertionError, ValueError)):
        impl.GPT(cfg)(torch.zeros(1, cfg.block_size + 1, dtype=torch.long))


def test_gpt_matches_reference():
    """The whole forward pass, composed independently from PyTorch's own pieces:
    logits = lm_head(ln_f(encoder_layers(wte(idx) + wpe(pos)))). Catches a missing or misplaced ln_f,
    positions added twice or not at all, blocks applied in the wrong order, and so on."""
    torch.manual_seed(0)
    cfg = make_config(n_layer=3, n_head=4, n_embd=32, block_size=12)
    m = impl.GPT(cfg).double().eval()
    with torch.no_grad():
        for p in m.parameters():  # non-trivial weights everywhere, LayerNorm affines included
            p.normal_(std=0.2 if p.ndim == 1 else 0.1)
    idx = torch.randint(0, cfg.vocab_size, (3, 12))
    T = idx.shape[1]
    mask = nn.Transformer.generate_square_subsequent_mask(T, dtype=torch.float64)
    with torch.no_grad():
        x = m.transformer.wte.weight[idx] + m.transformer.wpe.weight[:T]  # (B, T, C)
        for blk in m.transformer.h:
            x = reference_block(blk, cfg.n_embd, cfg.n_head, torch.float64).eval()(x, src_mask=mask, is_causal=True)
        x = F.layer_norm(x, (cfg.n_embd,), m.transformer.ln_f.weight, m.transformer.ln_f.bias)
        want = x @ m.transformer.wte.weight.T  # (B, T, V): the head is the tied embedding matrix
        got, _ = m(idx)
    torch.testing.assert_close(got, want, **TOL[torch.float64])


# ---------------------------------------------------------------------------
# Stage 3: init, and the loss you should see before any training
# ---------------------------------------------------------------------------
def test_init_scales():
    cfg = make_config(n_embd=128, n_head=4, n_layer=4, block_size=64)
    torch.manual_seed(0)
    m = impl.GPT(cfg)
    resid = 0.02 / math.sqrt(2 * cfg.n_layer)
    for name, p in m.named_parameters():
        if p.ndim == 2:  # Linear and Embedding weights
            want = resid if name.endswith(("attn.c_proj.weight", "mlp.c_proj.weight")) else 0.02
            assert p.std().item() == pytest.approx(want, rel=0.1), name
            assert abs(p.mean().item()) < 0.1 * want, name
        elif name.endswith(".bias") and "ln_" not in name and "ln_f" not in name:
            assert torch.count_nonzero(p) == 0, f"{name} should start at zero"
        elif "ln_" in name or "ln_f" in name:
            assert torch.all(p == (1.0 if name.endswith("weight") else 0.0)), name


@pytest.mark.parametrize("V", [65, 20])
def test_loss_at_init_is_ln_vocab(V):
    """A model that knows nothing should be exactly as unsure as a uniform guess: loss = ln(V).
    (Tiny GPT-2 init puts a little noise on top, hence the small tolerance.)"""
    torch.manual_seed(0)
    cfg = make_config(vocab_size=V, n_embd=64, n_head=4, block_size=32)
    m = impl.GPT(cfg)
    idx = torch.randint(0, V, (16, 32))
    tgt = torch.randint(0, V, (16, 32))
    _, loss = m(idx, tgt)
    assert loss.item() == pytest.approx(math.log(V), abs=0.1), "loss@init off: init scale or loss wiring"


# ---------------------------------------------------------------------------
# Stage 4: what the model is and is not allowed to see
# ---------------------------------------------------------------------------
def test_model_cannot_see_the_future():
    cfg = make_config(block_size=12)
    m = impl.GPT(cfg).double().eval()
    T, t = 12, 4
    idx = torch.randint(0, cfg.vocab_size, (3, T))
    idx2 = idx.clone()
    idx2[:, t + 1 :] = torch.randint(0, cfg.vocab_size, (3, T - t - 1))
    with torch.no_grad():
        a, _ = m(idx)
        b, _ = m(idx2)
    torch.testing.assert_close(a[:, : t + 1], b[:, : t + 1], atol=1e-10, rtol=0)
    assert not torch.allclose(a[:, t + 1 :], b[:, t + 1 :]), "test is vacuous: the future had no effect at all"


def test_position_embeddings_are_used():
    """Without positions, attention sees a bag of tokens: permuting the PAST must not change the
    prediction at the last position at all. With positions it must."""
    torch.manual_seed(0)
    cfg = make_config(n_layer=1)  # one layer on purpose: in a deeper model the causal mask itself leaks order
    m = impl.GPT(cfg).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim == 2:
                p.normal_(std=0.3)  # GPT-2's tiny init makes every effect ~1e-5; blow it up so it is visible
        idx = torch.randint(0, cfg.vocab_size, (4, 10))
        shuffled = idx.clone()
        shuffled[:, :-1] = idx[:, :-1][:, torch.randperm(9)]
        a, _ = m(idx)
        b, _ = m(shuffled)
    assert (a[:, -1] - b[:, -1]).abs().max() > 1e-3, "permuting the past changed nothing: is wpe added to the input?"


# ---------------------------------------------------------------------------
# Stage 5: it can learn
# ---------------------------------------------------------------------------
def train(model, x, y, steps, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_overfit_one_batch():
    """The gate. A model with this much capacity must memorise 4 x 16 tokens. If it can't, there is
    a bug (targets, mask, residuals, optimiser wiring), not a tuning problem; fix it here."""
    torch.manual_seed(0)
    V = 20
    cfg = make_config(vocab_size=V)
    m = impl.GPT(cfg).train()
    seq = torch.randint(0, V, (4, cfg.block_size + 1))
    x, y = seq[:, :-1], seq[:, 1:]
    losses = train(m, x, y, steps=400, lr=3e-3)
    assert losses[0] == pytest.approx(math.log(V), abs=0.15)
    assert all(math.isfinite(l) for l in losses)
    assert losses[-1] < 0.05, f"final loss {losses[-1]:.3f}: cannot even overfit one batch"


def test_learns_a_repeating_pattern():
    """Not memorisation: a periodic text must be predicted perfectly at positions it never trained on."""
    torch.manual_seed(0)
    codec = data.CharCodec("abcd")
    stream = torch.tensor(codec.encode("abcd" * 200))
    cfg = make_config(vocab_size=codec.vocab_size, block_size=8)
    m = impl.GPT(cfg).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        x, y = data.get_batch(stream, cfg.block_size, 16, generator=g)
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    m.eval()
    x, y = data.get_batch(stream, cfg.block_size, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits, loss = m(x, y)
    assert (logits.argmax(-1) == y).float().mean().item() > 0.99
    assert loss.item() < 0.05


# ---------------------------------------------------------------------------
# Stage 6: sampling
# ---------------------------------------------------------------------------
def test_generate_contract():
    torch.manual_seed(0)
    cfg = make_config(block_size=8)
    m = impl.GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    out = m.generate(idx, max_new_tokens=20)  # 5 + 20 > block_size: the context must be cropped
    assert out.shape == (2, 25)
    assert out.dtype == torch.long
    assert torch.equal(out[:, :5], idx)
    assert out.min() >= 0 and out.max() < cfg.vocab_size


def test_generate_follows_a_sharply_peaked_model():
    """If the model is certain the next token is always 3, sampling must produce 3 every time."""
    cfg = make_config(vocab_size=6, block_size=8)
    m = impl.GPT(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.zero_()  # (tied to wte, which is fine for this test)
        m.lm_head.weight[3].fill_(5.0)
        m.transformer.ln_f.weight.fill_(1.0)
        m.transformer.ln_f.bias.fill_(1.0)  # ln_f(h) is ~1 in every channel -> logit of token 3 = 5 * C
    out = m.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=10)
    assert torch.all(out[:, 2:] == 3)