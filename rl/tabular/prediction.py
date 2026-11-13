"""Tabular policy evaluation: exact (linear solve) and sampled (Monte Carlo).

Arrays, not classes. S = number of states, A = number of actions.
    p[s, a, s']   transition probabilities     shape (S, A, S)
    R[s, a]       rewards                      shape (S, A)
    pi[s, a]      policy, each row sums to 1   shape (S, A)

Fill in every TODO, top to bottom, and check one function at a time:
    uv run pytest rl/tabular -k r_pi_P_pi
    uv run pytest rl/tabular -k policy_eval_linear
    uv run pytest rl/tabular -k horizon
    uv run pytest rl/tabular -k mc_returns
"""

import numpy as np
import math


def r_pi_P_pi(p, R, pi):
    """Average the actions away under pi, turning the MDP into a plain chain.

    r_pi[s]     = sum over a of  pi[s, a] * R[s, a]          shape (S,)
    P_pi[s, s'] = sum over a of  pi[s, a] * p[s, a, s']      shape (S, S); rows "from", columns "to"
    """
    S, A = R.shape
    r_pi = np.zeros(S)
    P_pi = np.zeros((S, S))
    # TODO: fill r_pi and P_pi from the two formulas above. Plain loops over s and a are fine.
    for s in range(S):
        for a in range(A):
            r_pi[s] += pi[s, a]*R[s, a]

    for s in range(S):
        for a in range(A):
            P_pi[s] += pi[s, a]*p[s, a]
    return r_pi, P_pi


def policy_eval_linear(p, R, pi, gamma):
    """Exact v_pi from (I - gamma * P_pi) v = r_pi. Returns v, shape (S,)."""
    # TODO: get r_pi and P_pi, build the matrix, then np.linalg.solve (don't form the inverse).
    # The identity's size has to come from the inputs, not a hard-coded number.
    r_pi, P_pi = r_pi_P_pi(p, R, pi)
    v_pi = np.linalg.solve((np.identity(r_pi.size) - gamma * P_pi), r_pi)
    return v_pi


def horizon_for_tolerance(gamma, r_max, tol):
    """Smallest integer H >= 0 with  r_max * gamma**H / (1 - gamma) <= tol."""
    # TODO: invert the bound with logs (ln(gamma) < 0 flips the inequality), round up,
    # clamp at 0, and return a Python int.
    target = tol*(1-gamma)/r_max                    # need gamma**H <= target
    H = math.ceil(math.log(target)/math.log(gamma)) # ln(gamma) < 0 flips <= into >=
    return max(0, H)                                # tol loose enough that H = 0 already works


def mc_returns(p, R, pi, gamma, s0, n_episodes, horizon, rng):
    """Discounted returns of `n_episodes` rollouts from s0, each cut after `horizon` steps.

    Each step: a_t ~ pi[s_t],  r_t = R[s_t, a_t],  s_{t+1} ~ p[s_t, a_t].
    G = r_0 + gamma * r_1 + ... + gamma**(horizon - 1) * r_{horizon - 1}.
    All randomness comes from rng (a np.random.Generator); no global seeding.
    Returns a float array, shape (n_episodes,).
    """
    S, A = R.shape
    returns = np.zeros(n_episodes)
    for i in range(n_episodes):
        s = s0
        # TODO: run `horizon` steps from s, adding each discounted reward into returns[i].
        # Sampling: rng.choice(n, p=probs) draws an index in 0..n-1 with those probabilities.
        discount = 1                                # weight on the very first reward
        for t in range(horizon):
            a = rng.choice(A, p=pi[s])              # draw an action from the policy, in state s
            returns[i] += discount * R[s, a]        # collect the discounted reward
            s = rng.choice(S, p=p[s, a])            # draw the next state from the dynamics
            discount *= gamma                       # weight for the next reward
    return returns