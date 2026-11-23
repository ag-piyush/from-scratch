"""Bellman operators for tabular MDPs, stored as arrays.

Arrays, not classes. S = number of states, A = number of actions.
    p[s, a, s']   transition probabilities     shape (S, A, S)
    R[s, a]       rewards                      shape (S, A)
    pi[s, a]      policy, each row sums to 1   shape (S, A)
    v[s]          any real vector              shape (S,)

Every function returns a NEW array and leaves p, R, pi and v unchanged.

Fill in every TODO, top to bottom, and check one function at a time:
    uv run pytest rl/tabular/test_operators.py -k q_from_v
    uv run pytest rl/tabular/test_operators.py -k T_pi
    uv run pytest rl/tabular/test_operators.py -k T_star
    uv run pytest rl/tabular/test_operators.py -k contraction -s
"""

import numpy as np


def q_from_v(p, R, v, gamma):
    """One step of lookahead from v. No policy appears.

    q[s, a] = R[s, a] + gamma * sum over s' of  p[s, a, s'] * v[s']      shape (S, A)
    """
    S, A = R.shape
    q = np.zeros((S, A))
    # TODO: fill q from the formula above. Plain loops over s and a are fine.
    q = R + gamma*(p@v)
    return q


def T_pi(p, R, pi, v, gamma):
    """Bellman expectation operator.

    (T_pi v)[s] = sum over a of  pi[s, a] * q[s, a],   q = q_from_v(p, R, v, gamma)    shape (S,)
    """
    # TODO: build on q_from_v; don't redo the lookahead here.
    ...
    q = q_from_v(p, R, v, gamma)
    return (pi*q).sum(axis=1)


def T_star(p, R, v, gamma):
    """Bellman optimality operator.

    (T_star v)[s] = max over a of  q[s, a],   q = q_from_v(p, R, v, gamma)    shape (S,)
    """
    # TODO: build on q_from_v. One number per state.
    ...
    q = q_from_v(p, R, v, gamma)
    return q.max(axis=1)
