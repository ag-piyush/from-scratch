# Tabular policy evaluation

<!-- Skeleton written before the code. Replace every <...> once the tests pass. Keep it short. -->

**What it is:** exact and sampled policy evaluation for small MDPs stored as arrays (`p[s, a, s']`, `R[s, a]`). `policy_eval_linear` builds the policy's reward vector and transition matrix and solves (I − γ·P_π)·v = r_π. `mc_returns` estimates the same v by averaging returns from rollouts cut after a horizon H, which `horizon_for_tolerance` picks from a worst-case bound on what the cut can cost. Policy iteration and value iteration (#8) and TD(0) (#9) will build on this folder.

**Checked against:** Sutton & Barto's Figure 3.2 (5×5 gridworld, random policy, γ = 0.9), to the book's printed precision and to 1e-4; hand-solved values for a 3-state MDP (stay / go / exit, with and without a leaky exit); the Bellman equation itself for a random policy; and Monte Carlo against the linear solve within 4 standard errors plus the cut's bound.

```bash
uv run pytest rl/tabular
```

**Result:** <e.g. "linear solve: max abs diff vs Figure 3.2 of 0.0xx (book rounding); Monte Carlo at the top-left cell: x.xx ± 0.xx (N = 3000, H = 100) vs exact 3.309". Your predictions from the test docstring vs what came out is worth a line too.>

**Gotcha:** <one bug you hit, or a detail that surprised you, and why it happens>

**References:** Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018), §3.5–3.6 and Example 3.5 ([PDF](http://incompleteideas.net/book/RLbook2020.pdf)); §5.1 for Monte Carlo prediction.
