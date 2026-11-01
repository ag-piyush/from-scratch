# Tiny char-level GPT

**What it is:** a decoder-only transformer (GPT-2 layout: token + position embeddings, pre-norm blocks of causal attention and a GELU MLP, final LayerNorm, tied output head) trained to predict the next character of tiny Shakespeare. Reuses my own `CausalSelfAttention` from `transformer/attention`. Layer names follow GPT-2 so its weights can be loaded later.

**Checked against:** `torch.nn.TransformerEncoderLayer(norm_first=True)` for each block and for the whole forward pass (same weights, causal mask), plus the training gates: loss at init equals ln(65), one batch can be memorised, a periodic text is predicted perfectly, and changing future tokens never changes earlier logits.

```bash
uv run pytest transformer/gpt
```

**Data:** tiny Shakespeare, about 1.1 MB of plain ASCII text (1,115,394 characters, 65 unique), from Karpathy's [char-rnn](https://github.com/karpathy/char-rnn). The tests don't need it; only the training run does. Download it into the git-ignored `data/` folder at the repo root:

```bash
mkdir -p data
curl -L -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -c data/input.txt   # 1115394
```

**Training:** from the repo root (use `-m`, so `transformer.gpt` imports resolve):

```bash
uv run python -m transformer.gpt.train
```

**Result:** baby config (4 layers, 4 heads, `n_embd` 128, 0.81M parameters, block size 64), 3,000 steps of batch 32 with AdamW at lr 1e-3, 72 s on an M4 (MPS). Val loss **1.633** (train 1.454), well below the n-gram baselines on the same 90/10 split:

| Model | Val loss |
|---|---|
| Uniform (loss at init) | 4.174 |
| Unigram | 3.347 |
| Bigram (add-one) | 2.482 |
| Trigram (add-0.1) | 2.046 |
| **This GPT, step 3,000** | **1.633** |

Sample from a `"\n"` prompt after training: speaker names, line breaks and verse-length lines, mostly real words, no meaning beyond a phrase.

```text
HASTINGS:
I much some anst let out,
Badcome and me; more place.


DUKE VINCENTIO:
It was' the pitile and a warse obtainty,
And tear told in where poled it, let's theird a body.

KING RICHARD III:
These a shopher is a within their false;
```

**Gotcha:** with PyTorch's default init, the loss at init is about 83, not ln(65) = 4.17. `nn.Embedding` starts at N(0, 1), the output head is tied to it, and `ln_f` feeds it unit-variance vectors, so each logit is a dot product of two length-128 N(0, 1)-ish vectors, with std around √128 ≈ 11. Softmax then puts almost all its probability on a random token, so the model starts confidently wrong. GPT-2's init (N(0, 0.02), with the residual projections scaled by 1/√(2·n_layer)) brings the logits' std down to about 0.26 and the loss to 4.21, so training starts from "uniformly unsure" instead of spending its first steps undoing false confidence.

**References:** Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019); Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), [nanoGPT](https://github.com/karpathy/nanoGPT) and [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).