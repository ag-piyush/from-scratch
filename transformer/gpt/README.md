# Tiny char-level GPT

**What it is:** a decoder-only transformer (GPT-2 layout: token + position embeddings, pre-norm blocks of causal attention and a GELU MLP, final LayerNorm, tied output head) trained to predict the next character of tiny Shakespeare. Reuses my own `CausalSelfAttention` from `transformer/attention`. Layer names follow GPT-2 so its weights can be loaded later.

**Checked against:** `torch.nn.TransformerEncoderLayer(norm_first=True)` for each block and for the whole forward pass (same weights, causal mask), plus the training gates: loss at init equals ln(65), one batch can be memorised, a periodic text is predicted perfectly, and changing future tokens never changes earlier logits.

```bash
uv run pytest transformer/gpt
```

**Result:** <!-- TODO after the M4 run: val loss vs the baselines (uniform 4.174, unigram 3.347, bigram 2.482, trigram 2.046), step count, wall-clock on the M4, and a loss-curve image or a 300-character sample -->

**Gotcha:** <!-- TODO: one bug you hit or detail that surprised you, and why it happens -->

**References:** Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019); Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), [nanoGPT](https://github.com/karpathy/nanoGPT) and [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).