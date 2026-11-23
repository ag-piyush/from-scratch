# Tabular policy evaluation

**What it is:** exact and sampled policy evaluation for small MDPs stored as arrays (`p[s, a, s']`, `R[s, a]`). `policy_eval_linear` builds the policy's reward vector and transition matrix and solves (I − γ·P_π)·v = r_π. `mc_returns` estimates the same v by averaging returns from rollouts cut after a horizon H, which `horizon_for_tolerance` picks from a worst-case bound on what the cut can cost. Policy iteration and value iteration (#8) and TD(0) (#9) will build on this folder.

**Checked against:** Sutton & Barto's Figure 3.2 (5×5 gridworld, random policy, γ = 0.9), to the book's printed precision and to 1e-4; hand-solved values for a 3-state MDP (stay / go / exit, with and without a leaky exit); the Bellman equation itself for a random policy; and Monte Carlo against the linear solve within 4 standard errors plus the cut's bound.

```bash
uv run pytest rl/tabular
```

**Result:** 56/56 pass. Linear solve vs Figure 3.2: max abs diff 0.050 (the book's rounding). Monte Carlo at the top-left cell: 3.30 ± 0.08 (N = 3000, H = 100) vs exact 3.309. Single returns have std 4.14, so a standard error of 0.1 needs ~1,700 episodes (~6,900 to land within ±0.1 at ~95%, i.e. 2 standard errors), and the cut at H = 100 costs at most 0.003. Prediction vs outcome: I guessed a neighbour of A would be the highest-value cell, thinking the +10 is paid on entering A. It's A itself (8.79), because R[s, a] pays for acting in s. It's below 10 because every action from A jumps to A′ on the bottom wall, where v = −1.35.

**Gotcha:** I dropped γ from `I − γ·P_π`, which solves the γ = 1 case. Every row of P_π sums to 1, so (I − P_π)·**1** = 0 and the matrix is singular. numpy raised `LinAlgError` on the tiny MDP, where the absorbing state's row is exactly zero, but on the gridworld it quietly returned values around −5.9 × 10¹⁴, because rounding hides the zero pivot. The Bellman-residual test caught it: a solve that runs without an error isn't proof it's right.

**References:** Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018), §3.5–3.6 and Example 3.5 ([PDF](http://incompleteideas.net/book/RLbook2020.pdf)); §5.1 for Monte Carlo prediction.

## Bellman operators (#8, M2)

**What it is:** the three Bellman operators as array functions. `q_from_v` does one step of lookahead from any value vector v, `T_pi` averages that lookahead under a policy, and `T_star` takes the best action. Policy evaluation, policy iteration and value iteration (M3–M4) are repeated applications of these.

**Checked against:** hand-computed q, T_π v and T\* v on the 3-state MDP (leaky exit, v = [1, 2, 3]); v_π from `policy_eval_linear` (Figure 3.2) and v\* (Figure 3.5) as fixed points; the matrix form T_π v = r_π + γ·P_π v; one-hot probes on a random MDP with S = A, so swapped axes can't hide behind matching shapes; monotonicity; and a contraction check over 2,000 random pairs per operator and γ.

```bash
uv run pytest rl/tabular/test_operators.py
uv run pytest rl/tabular/test_operators.py -k contraction -s
```

**Result:** 40/40 pass (96/96 for the folder). The contraction check's worst ratio is exactly γ for both operators at every γ: 0.5000, 0.9000, 0.9900. Expected before the run: the bound is γ, because a probability-weighted average of v − w can't exceed its largest entry, and random pairs hit it exactly because of A. Every action from A lands on A′, so (Tv − Tw)(A) = γ·(v − w)(A′) with no averaging, and whenever |v − w| peaks at A′ or B′ the ratio is γ (164 of 2,000 pairs for T_π, close to the 2/25 expected). T\* hits it more often (227 pairs) because it copies a single successor wherever v and w choose the same move, while T_π averages over up to four neighbours: away from A and B its worst ratio is 0.816.

**Gotcha:** my first `q_from_v` looped `p[s] * v[s]` over s, using s for both the current state (the first axis of `p`) and the next state (the index of `v`). That sums down the columns of each P_a, giving vᵀP_a, the v-weighted flow *into* each state, not the lookahead P_a·v, and its shape is (A, S): a broadcast error on the gridworld, but an S = A MDP runs it without complaint. The fix is `R + gamma * (p @ v)`: with a 3-D array on the left, `@` sums over the last axis, which is s′. Second slip, in `T_pi`: `q.transpose(-2, -1)` is PyTorch's swap-two-dims call, but NumPy's `transpose(*axes)` takes a full ordering of the axes, so on a 2-D array it does nothing.

**References:** Sutton & Barto, 2nd ed. (2018), §3.6 (v*, Figure 3.5) and §4.1–4.4 ([PDF](http://incompleteideas.net/book/RLbook2020.pdf)).