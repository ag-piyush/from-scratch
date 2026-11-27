"""Dynamic programming for tabular MDPs: iterative policy evaluation and policy iteration.

Arrays, not classes. S = number of states, A = number of actions.
    p[s, a, s']   transition probabilities     shape (S, A, S)
    R[s, a]       rewards                      shape (S, A)
    pi[s, a]      policy, each row sums to 1   shape (S, A)
    q[s, a]       action values                shape (S, A)
    v[s]          state values                 shape (S,)

A deterministic policy is a one-hot pi: a single 1.0 per row. Its action in s is pi[s].argmax().
Full contract: the docstring of test_dp.py.

Fill in every function, top to bottom, and check one at a time:
    uv run pytest rl/tabular/test_dp.py -k greedy
    uv run pytest rl/tabular/test_dp.py -k eval_iterative
    uv run pytest rl/tabular/test_dp.py -k policy_iteration
"""

import numpy as np

import rl.tabular.operators as ops
import rl.tabular.prediction as pred


def greedy_policy(q, pi_current=None, tol=1e-9):
    """One-hot policy that is greedy in q.

    Keeps pi_current's action in a state unless another action beats it by more than tol.
    Without pi_current, takes the first action attaining the max.
    Returns pi, float, shape (S, A).
    """
    a_star = np.argmax(q, axis = 1)
    S, A = q.shape
    pi = np.zeros((S, A))
    for s in range(S):
        if pi_current is not None:
            current_action = np.argmax(pi_current[s])
            if q[s, current_action] >= q[s, a_star[s]] - tol:
                pi[s, current_action] = 1.0
                continue
        pi[s, a_star[s]] = 1.0
    return pi



def policy_eval_iterative(p, R, pi, gamma, tol, in_place=False, order=None, v0=None,
                          max_sweeps=100_000):
    """Sweep until a sweep's largest change is < tol.

    in_place=False: two-array, one application of T_pi per sweep (order ignored).
    in_place=True: update v[s] one state at a time, in `order` (default 0..S-1).
    Starts from v0 (zeros if None) and leaves it unchanged.
    Returns (v, n_sweeps): v float (S,), n_sweeps a Python int counting the stopping sweep.
    Raises RuntimeError after max_sweeps sweeps without reaching tol.
    """
    S, A = R.shape
    v = np.zeros(S) if v0 is None else np.array(v0, dtype=float)

    for sweep in range(max_sweeps):
        v_old = v.copy()
        if in_place:
            if order is None:
                order = np.arange(S)
            for s in order:
                v[s] = ops.T_pi(p, R, pi, v, gamma)[s]      
        else:
            v = ops.T_pi(p, R, pi, v, gamma)                
        max_change = np.max(np.abs(v - v_old))
        if max_change < tol:
            return v, sweep + 1
    raise RuntimeError(f"no convergence after {max_sweeps} sweeps")   # fix 2

def policy_iteration(p, R, gamma, pi0=None, tol=1e-9, max_iters=1000):
    """Evaluate exactly, improve greedily (with the tie rule), stop when the policy is unchanged.

    Starts from pi0 (one-hot; default action 0 everywhere) and leaves it unchanged.
    Returns (pi, v, n_evals): the final one-hot pi, its value v, and the number of
    evaluations as a Python int, counting the last one.
    Raises RuntimeError after max_iters evaluations without a stable policy.
    """
    S, A = R.shape
    pi = np.eye(A)[np.zeros(S, dtype=int)] if pi0 is None else np.array(pi0, dtype=float)    

    for n in range(1, max_iters + 1):
        v = pred.policy_eval_linear(p, R, pi, gamma)       # evaluate exactly
        q = ops.q_from_v(p, R, v, gamma)                    # one step of lookahead
        pi_new = greedy_policy(q, pi, tol)                  # improve, keeping ties
        if np.array_equal(pi_new, pi):                      # stable: pi is optimal
            return pi, v, n
        pi = pi_new
    raise RuntimeError(f"no stable policy after {max_iters} evaluations")