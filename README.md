# from-scratch

[![tests](https://github.com/ag-piyush/from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/ag-piyush/from-scratch/actions/workflows/tests.yml)

From-scratch building blocks of modern ML: transformers and LLMs, deep and offline RL, post-training and GPU kernels, each checked against a reference.

Every component here is written from a blank file in PyTorch and comes with a test that compares it against a reference implementation or a published result. Each folder has a short README covering what it is, what it's checked against, one result, and one gotcha hit along the way.

## Index

<!-- Add a row when a component lands. Checked against = what its test compares to. -->

| Component | Folder | Checked against | Status |
|---|---|---|---|
| Causal multi-head attention | `transformer/attention` | `F.scaled_dot_product_attention` (causal mask) | 🚧 in progress |

## Layout

```
transformer/     attention, GPT, sampling and decoding, KV cache
tokenization/    BPE
rl/              DQN, PPO, IQL, Decision Transformer
post_training/   GRPO objective, LoRA
gpu_kernels/     Triton kernels
```

Each component lives in `<topic>/<component>/` with its implementation, a `test_*.py`, and a README (template in [`.github/DRILL_README_TEMPLATE.md`](.github/DRILL_README_TEMPLATE.md)).

## Running

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                               # install dependencies
uv run pytest                         # run every test
uv run pytest transformer/attention   # run one component's tests
```

Tests run on CPU in CI for every push. Tests import by path from the repo root, e.g. `from transformer.attention.attention import ...`.

## License

[MIT](LICENSE)
