# from-scratch
 
[![tests](https://github.com/ag-piyush/from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/ag-piyush/from-scratch/actions/workflows/tests.yml)
 
From-scratch building blocks of modern ML: transformers and LLMs, deep and offline RL, post-training and GPU kernels, each checked against a reference.
 
Every component here is written from a blank file in PyTorch or NumPy and comes with a test that compares it against a reference implementation or a published result. Each folder has a short README covering what it is, what it's checked against, one result, and one gotcha hit along the way.
 
## Index
 
<!-- Add a row when a component lands. Checked against = what its test compares to. -->
 
| Component | Folder | Checked against | Status |
|---|---|---|---|
| Causal multi-head attention | `transformer/attention` | `F.scaled_dot_product_attention` (causal mask) | ✅ done |
| Tiny char-level GPT | `transformer/gpt` | `nn.TransformerEncoderLayer` (pre-norm, causal mask), loss at init = ln(V), overfit one batch | ✅ done |
| Tabular policy evaluation (linear solve + Monte Carlo) | `rl/tabular` | Sutton & Barto Figure 3.2, a hand-solved 3-state MDP, the Bellman equation, Monte Carlo vs the linear solve | ✅ done |
| Bellman operators (q from v, T_π, T\*) | `rl/tabular` | Hand-computed values on a 3-state MDP, v_π (Figure 3.2) and v\* (Figure 3.5) as fixed points, the matrix form r_π + γP_π v, a contraction check | ✅ done |
| Iterative policy evaluation and policy iteration | `rl/tabular` | Hand-worked sweep counts on a 3-state MDP, the stopping bound γ/(1 − γ)·tol vs the linear solve, the policy improvement theorem, v\* (Figure 3.5), brute force over every deterministic policy of a small random MDP | ✅ done |
 
## Layout
 
```
transformer/     attention, GPT, sampling and decoding, KV cache
tokenization/    BPE
rl/              tabular policy evaluation and policy iteration (value iteration and TD next), DQN, PPO, IQL, Decision Transformer
post_training/   GRPO objective, LoRA
gpu_kernels/     Triton kernels
```
 
Each component lives in `<topic>/<component>/` with its implementation, a `test_*.py`, and a README (template in [`.github/DRILL_README_TEMPLATE.md`](.github/DRILL_README_TEMPLATE.md)).
 
## Running
 
Requires [uv](https://docs.astral.sh/uv/).
 
```bash
uv sync                                  # install dependencies
uv run pytest                            # run every test
uv run pytest transformer/attention      # run one component's tests
uv run python -m transformer.gpt.train   # train the tiny GPT (needs the dataset; see transformer/gpt/README.md)
```
 
Tests run on CPU in CI for every push. Tests import by path from the repo root, e.g. `from transformer.attention.attention import ...`.
 
## License
 
[MIT](LICENSE)