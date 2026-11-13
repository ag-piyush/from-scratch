# Tabular policy evaluation

**What it is:** exact and sampled policy evaluation for small MDPs stored as arrays (`p[s, a, s']`, `R[s, a]`). `policy_eval_linear` builds the policy's reward vector and transition matrix and solves (I − γ·P_π)·v = r_π. `mc_returns` estimates the same v by averaging returns from rollouts cut after a horizon H, which `horizon_for_tolerance` picks from a worst-case bound on what the cut can cost. Policy iteration and value iteration (#8) and TD(0) (#9) will build on this folder.

**Checked against:** Sutton & Barto's Figure 3.2 (5×5 gridworld, random policy, γ = 0.9), to the book's printed precision and to 1e-4; hand-solved values for a 3-state MDP (stay / go / exit, with and without a leaky exit); the Bellman equation itself for a random policy; and Monte Carlo against the linear solve within 4 standard errors plus the cut's bound.

```bash
uv run pytest rl/tabular
```

**Result:** 56/56 pass. Linear solve vs Figure 3.2: max abs diff 0.050 (the book's rounding). Monte Carlo at the top-left cell: 3.30 ± 0.08 (N = 3000, H = 100) vs exact 3.309. Single returns have std 4.14, so ±0.1 needs ~1,700 episodes, and the cut at H = 100 costs at most 0.003. Prediction vs outcome: I guessed a neighbour of A would be the highest-value cell, thinking the +10 is paid on entering A. It's A itself (8.79), because R[s, a] pays for acting in s. It's below 10 because every action from A jumps to A′ on the bottom wall, where v = −1.35.

**Gotcha:** I dropped γ from `I − γ·P_π`, which solves the γ = 1 case. Every row of P_π sums to 1, so (I − P_π)·**1** = 0 and the matrix is singular. numpy raised `LinAlgError` on the tiny MDP, where the absorbing state's row is exactly zero, but on the gridworld it quietly returned values around −5.9 × 10¹⁴, because rounding hides the zero pivot. The Bellman-residual test caught it: a solve that runs without an error isn't proof it's right.

**References:** Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018), §3.5–3.6 and Example 3.5 ([PDF](http://incompleteideas.net/book/RLbook2020.pdf)); §5.1 for Monte Carlo prediction.