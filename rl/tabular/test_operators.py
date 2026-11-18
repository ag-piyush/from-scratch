"""Tests for the Bellman operators (study topic #8, M2), written before the implementation.

Implement this file from a blank file until these pass:

    uv run pytest rl/tabular/test_operators.py

    rl/tabular/operators.py     q_from_v, T_pi, T_star

Done when: all three operators match your hand numbers on the tiny MDP, leave v_pi and v*
unchanged (Figure 3.2 and Figure 3.5), and the contraction check never exceeds gamma.
Next in this folder: greedy_policy, policy_eval_iterative, policy_iteration (M3).

Write these down BEFORE you write any code
------------------------------------------
1. Tiny MDP, leaky exit (B -> A w.p. 0.1), gamma = 0.9, and v = [1, 2, 3] for (A, B, T).
   This v is just a vector. It doesn't have to be the value of any policy, and that's fine:
   an operator takes ANY vector. By hand, compute all six entries of q = q_from_v(v),
   then T_pi(v) for pi_half and T_star(v). The first tests below use exactly these numbers.
2. v* for the tiny MDP in three cases: no leak at gamma = 0.9, no leak at gamma = 0.5, and the
   leaky exit at gamma = 0.9. Which action is optimal in A in each case? T_star must leave
   each one unchanged. Work them out before you scroll down to the test that checks them.
3. The contraction check (the main prediction). On the gridworld, the check draws 2,000
   random pairs v, w, each entry independently from N(0, 10^2), and reports the worst ratio
       ||T v - T w||_inf / ||v - w||_inf
   for T = T_pi (equiprobable random policy) and T = T_star, at gamma = 0.5, 0.9 and 0.99.
   Before running it, write down:
     (a) the largest value the ratio can possibly take, and which step of your M1 proof says so;
     (b) for gamma = 0.9, your guess of the worst ratio random pairs actually hit, for T_pi and
         for T_star separately. Close to the bound or well below it? Which operator gets
         closer, and why might they differ?
     (c) if q_from_v forgot gamma, what would the check print instead?
   Then run it and compare with what you wrote:
       uv run pytest rl/tabular/test_operators.py -k contraction -s
4. After the run: explain the number it printed, in one or two lines. Which state s makes
   |(T v - T w)(s)| largest relative to ||v - w||_inf, and what about that state does it?

The contract the tests rely on
------------------------------
Same arrays as prediction.py. S = number of states, A = number of actions.
    p[s, a, s']   transition probabilities     shape (S, A, S)
    R[s, a]       rewards                      shape (S, A)
    pi[s, a]      policy, each row sums to 1   shape (S, A)
    v[s]          any real vector              shape (S,)

q_from_v(p, R, v, gamma) -> q
    q: (S, A),  q[s, a] = R[s, a] + gamma * sum over s' of p[s, a, s'] * v[s']
    One step of lookahead, then trust v. No policy appears.

T_pi(p, R, pi, v, gamma) -> (S,)
    (T_pi v)[s] = sum over a of pi[s, a] * q[s, a],   with q = q_from_v(p, R, v, gamma)

T_star(p, R, v, gamma) -> (S,)
    (T_star v)[s] = max over a of q[s, a]

All three return new float arrays and leave p, R, pi and v unchanged. Any S and A must work
(the tests use S = 3, A = 2 for the tiny MDP, S = 25, A = 4 for the gridworld, and S = A = 5
for a random MDP, where shapes alone can't tell you if two axes are swapped).

Suggested order, top to bottom: q_from_v, T_pi, T_star, then the contraction check.
    uv run pytest rl/tabular/test_operators.py -k q_from_v
    uv run pytest rl/tabular/test_operators.py -k T_pi
    uv run pytest rl/tabular/test_operators.py -k T_star
"""

import numpy as np
import pytest

import rl.tabular.gridworld as gw
import rl.tabular.operators as ops
import rl.tabular.prediction as pred

# S&B Figure 3.5 (final 2nd edition), middle panel: v* for the gridworld, gamma = 0.9,
# as printed in the book (one decimal).
FIG_3_5 = np.array([
    [22.0, 24.4, 22.0, 19.4, 17.5],
    [19.8, 22.0, 19.8, 17.8, 16.0],
    [17.8, 19.8, 17.8, 16.0, 14.4],
    [16.0, 17.8, 16.0, 14.4, 13.0],
    [14.4, 16.0, 14.4, 13.0, 11.7],
])

# The same v* to more decimals.
V_STAR_GRID = np.array([
    [21.97749, 24.41943, 21.97749, 19.41943, 17.47749],
    [19.77974, 21.97749, 19.77974, 17.80176, 16.02159],
    [17.80176, 19.77974, 17.80176, 16.02159, 14.41943],
    [16.02159, 17.80176, 16.02159, 14.41943, 12.97749],
    [14.41943, 16.02159, 14.41943, 12.97749, 11.67974],
])


def uniform_pi(n_states=25, n_actions=4):
    return np.full((n_states, n_actions), 1.0 / n_actions)


# The tiny MDP again (same as test_tabular.py), so these tests don't depend on that file.
# States: 0 = A, 1 = B, 2 = T (absorbing, reward 0). Two action slots everywhere:
#   A: 0 = stay (+1, back to A), 1 = go (0, to B)
#   B: both slots are exit (+10; to A w.p. leak, to T otherwise)
#   T: both slots stay in T with reward 0
def tiny_mdp(leak=0.0):
    p = np.zeros((3, 2, 3))
    R = np.zeros((3, 2))
    p[0, 0, 0] = 1.0
    R[0, 0] = 1.0
    p[0, 1, 1] = 1.0
    for a in range(2):
        p[1, a, 0] = leak
        p[1, a, 2] = 1.0 - leak
        R[1, a] = 10.0
        p[2, a, 2] = 1.0
    return p, R


def tiny_pi(p_stay):
    return np.array([[p_stay, 1.0 - p_stay], [1.0, 0.0], [1.0, 0.0]])


PI_HALF = tiny_pi(0.5)


def random_mdp(n_states, n_actions, seed):
    """Stochastic transitions and random rewards: nothing special to hide a bug behind."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.normal(0.0, 1.0, size=(n_states, n_actions))
    return p, R


# ---------------------------------------------------------------------------
# 1. q_from_v
# ---------------------------------------------------------------------------
def test_q_from_v_shape_and_dtype():
    p, R = gw.make_gridworld()
    q = ops.q_from_v(p, R, np.zeros(25), 0.9)
    assert q.shape == (25, 4)
    assert np.issubdtype(q.dtype, np.floating)


def test_q_from_v_tiny_leaky_matches_your_hand_computation():
    # Prediction 1.
    p, R = tiny_mdp(leak=0.1)
    q = ops.q_from_v(p, R, np.array([1.0, 2.0, 3.0]), 0.9)
    np.testing.assert_allclose(
        q,
        [[1.9, 1.8],
         [12.52, 12.52],
         [2.7, 2.7]],
        atol=1e-12,
    )


def test_q_from_v_zero_v_gives_R():
    # With nothing to look ahead to, q is just the immediate reward.
    p, R = gw.make_gridworld()
    np.testing.assert_allclose(ops.q_from_v(p, R, np.zeros(25), 0.9), R, atol=1e-12)


def test_q_from_v_gamma_zero_ignores_v():
    p, R = gw.make_gridworld()
    v = np.random.default_rng(0).normal(0.0, 10.0, size=25)
    np.testing.assert_allclose(ops.q_from_v(p, R, v, 0.0), R, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.5, 0.9])
def test_q_from_v_one_hot_probes_pin_the_axes(gamma):
    # S = A = 5, so a swapped axis still has the right shape. Put 1 on state t and 0 elsewhere:
    # q - R must then be gamma times the probability of landing in t, for every (s, a).
    p, R = random_mdp(5, 5, seed=1)
    for t in range(5):
        e_t = np.zeros(5)
        e_t[t] = 1.0
        q = ops.q_from_v(p, R, e_t, gamma)
        np.testing.assert_allclose(
            q - R, gamma * p[:, :, t], atol=1e-12,
            err_msg=f"wrong axis? q[s, a] should use p[s, a, s'] (probe t = {t})",
        )


def test_q_from_v_shifting_v_by_a_constant_shifts_q_by_gamma_times_it():
    # Rows of p sum to 1, so adding c to every v[s'] adds gamma * c to every q[s, a].
    p, R = random_mdp(6, 3, seed=2)
    v = np.random.default_rng(3).normal(size=6)
    c, gamma = 4.0, 0.9
    np.testing.assert_allclose(
        ops.q_from_v(p, R, v + c, gamma), ops.q_from_v(p, R, v, gamma) + gamma * c, atol=1e-12
    )


def test_operators_do_not_modify_their_inputs():
    p, R = random_mdp(4, 3, seed=4)
    pi = np.random.default_rng(5).dirichlet(np.ones(3), size=4)
    v = np.random.default_rng(6).normal(size=4)
    copies = [x.copy() for x in (p, R, pi, v)]
    ops.q_from_v(p, R, v, 0.9)
    ops.T_pi(p, R, pi, v, 0.9)
    ops.T_star(p, R, v, 0.9)
    for original, copy in zip((p, R, pi, v), copies):
        np.testing.assert_array_equal(original, copy)


# ---------------------------------------------------------------------------
# 2. T_pi
# ---------------------------------------------------------------------------
def test_T_pi_tiny_leaky_matches_your_hand_computation():
    # Prediction 1.
    p, R = tiny_mdp(leak=0.1)
    out = ops.T_pi(p, R, PI_HALF, np.array([1.0, 2.0, 3.0]), 0.9)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [1.85, 12.52, 2.7], atol=1e-12)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.9, 0.99])
def test_T_pi_equals_r_pi_plus_gamma_P_pi_v(gamma):
    # The matrix form from #7: T_pi v = r_pi + gamma * P_pi v. A non-uniform policy, so
    # "weight by pi" and "plain average over actions" give different answers.
    p, R = random_mdp(5, 5, seed=7)
    rng = np.random.default_rng(8)
    pi = rng.dirichlet(np.ones(5), size=5)
    v = rng.normal(0.0, 10.0, size=5)
    r_pi, P_pi = pred.r_pi_P_pi(p, R, pi)
    np.testing.assert_allclose(ops.T_pi(p, R, pi, v, gamma), r_pi + gamma * P_pi @ v, atol=1e-10)


@pytest.mark.parametrize(
    "leak, p_stay, gamma",
    [(0.0, 1.0, 0.9), (0.0, 0.0, 0.9), (0.0, 0.5, 0.5), (0.1, 0.5, 0.9), (0.1, 0.8, 0.9)],
)
def test_T_pi_leaves_v_pi_unchanged_tiny(leak, p_stay, gamma):
    p, R = tiny_mdp(leak)
    pi = tiny_pi(p_stay)
    v_pi = pred.policy_eval_linear(p, R, pi, gamma)
    np.testing.assert_allclose(ops.T_pi(p, R, pi, v_pi, gamma), v_pi, atol=1e-10)


def test_T_pi_leaves_figure_3_2_unchanged():
    p, R = gw.make_gridworld()
    v_pi = pred.policy_eval_linear(p, R, uniform_pi(), 0.9)
    np.testing.assert_allclose(ops.T_pi(p, R, uniform_pi(), v_pi, 0.9), v_pi, atol=1e-10)


def test_T_pi_moves_a_vector_that_is_not_v_pi():
    # Being a fixed point means something only if other vectors DO move.
    p, R = gw.make_gridworld()
    v = pred.policy_eval_linear(p, R, uniform_pi(), 0.9) + 1.0
    assert not np.allclose(ops.T_pi(p, R, uniform_pi(), v, 0.9), v, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. T_star
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "leak, gamma, v, expected",
    [
        (0.1, 0.9, [1.0, 2.0, 3.0], [1.9, 12.52, 2.7]),   # prediction 1
        (0.0, 0.9, [0.0, 10.0, 0.0], [9.0, 10.0, 0.0]),   # here "go" wins in A
    ],
)
def test_T_star_tiny_matches_your_hand_computation(leak, gamma, v, expected):
    p, R = tiny_mdp(leak)
    out = ops.T_star(p, R, np.array(v), gamma)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_T_star_zero_v_is_the_best_immediate_reward():
    p, R = gw.make_gridworld()
    np.testing.assert_allclose(ops.T_star(p, R, np.zeros(25), 0.9), R.max(axis=1), atol=1e-12)


@pytest.mark.parametrize(
    "leak, gamma, v_star",
    [
        # Prediction 2. Don't read these before you've worked them out.
        (0.0, 0.9, [10.0, 10.0, 0.0]),
        (0.0, 0.5, [5.0, 10.0, 0.0]),
        (0.1, 0.9, [10.0, 10.9, 0.0]),
    ],
)
def test_T_star_leaves_v_star_unchanged_tiny(leak, gamma, v_star):
    p, R = tiny_mdp(leak)
    v_star = np.array(v_star)
    np.testing.assert_allclose(ops.T_star(p, R, v_star, gamma), v_star, atol=1e-10)


def test_T_star_leaves_figure_3_5_unchanged():
    p, R = gw.make_gridworld()
    v = V_STAR_GRID.ravel()
    np.testing.assert_allclose(ops.T_star(p, R, v, 0.9), v, atol=1e-4)
    # The printed figure is within 0.05 of the true v*, so by the triangle inequality and
    # contraction: |T_star(fig) - fig| <= |T_star(fig) - T_star(v*)| + |v* - fig| <= 0.9*0.05 + 0.05.
    fig = FIG_3_5.ravel()
    np.testing.assert_allclose(ops.T_star(p, R, fig, 0.9), fig, atol=0.05 + 0.9 * 0.05)


def test_T_star_beats_every_T_pi_and_ties_the_greedy_one():
    p, R = random_mdp(5, 5, seed=9)
    rng = np.random.default_rng(10)
    v = rng.normal(0.0, 10.0, size=5)
    best = ops.T_star(p, R, v, 0.9)
    for _ in range(20):
        pi = rng.dirichlet(np.ones(5), size=5)
        assert (best >= ops.T_pi(p, R, pi, v, 0.9) - 1e-12).all()
    # The policy that puts all its weight on argmax_a q[s, a] gets exactly T_star.
    greedy = np.eye(5)[ops.q_from_v(p, R, v, 0.9).argmax(axis=1)]
    np.testing.assert_allclose(ops.T_pi(p, R, greedy, v, 0.9), best, atol=1e-12)


@pytest.mark.parametrize("which", ["T_pi", "T_star"])
def test_operators_are_monotone(which):
    # v <= w everywhere  =>  T v <= T w everywhere.
    p, R = random_mdp(6, 4, seed=11)
    rng = np.random.default_rng(12)
    pi = rng.dirichlet(np.ones(4), size=6)
    apply = {
        "T_pi": lambda u: ops.T_pi(p, R, pi, u, 0.9),
        "T_star": lambda u: ops.T_star(p, R, u, 0.9),
    }[which]
    for _ in range(200):
        v = rng.normal(0.0, 10.0, size=6)
        w = v + np.abs(rng.normal(0.0, 1.0, size=6))
        assert (apply(v) <= apply(w) + 1e-12).all()


# ---------------------------------------------------------------------------
# 4. The contraction check (prediction 3). Run with -s to see the printed ratios.
# ---------------------------------------------------------------------------
N_PAIRS = 2000


def worst_ratio(apply, n_states, rng):
    worst = 0.0
    for _ in range(N_PAIRS):
        v = rng.normal(0.0, 10.0, size=n_states)
        w = rng.normal(0.0, 10.0, size=n_states)
        ratio = np.abs(apply(v) - apply(w)).max() / np.abs(v - w).max()
        worst = max(worst, ratio)
    return worst


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
@pytest.mark.parametrize("which", ["T_pi", "T_star"])
def test_contraction_gridworld(which, gamma):
    p, R = gw.make_gridworld()
    apply = {
        "T_pi": lambda u: ops.T_pi(p, R, uniform_pi(), u, gamma),
        "T_star": lambda u: ops.T_star(p, R, u, gamma),
    }[which]
    worst = worst_ratio(apply, 25, np.random.default_rng(2026))
    print(f"\n{which:>6}  gamma = {gamma:<5}  worst ratio over {N_PAIRS} random pairs = {worst:.4f}")
    assert worst <= gamma + 1e-12, f"{which} stretched a pair by {worst:.4f} > gamma = {gamma}"


@pytest.mark.parametrize("gamma", [0.5, 0.9])
@pytest.mark.parametrize("which", ["T_pi", "T_star"])
def test_contraction_random_mdp(which, gamma):
    # Same check on a stochastic MDP with S = A = 6. Silent: it's here to catch bugs.
    p, R = random_mdp(6, 6, seed=13)
    pi = np.random.default_rng(14).dirichlet(np.ones(6), size=6)
    apply = {
        "T_pi": lambda u: ops.T_pi(p, R, pi, u, gamma),
        "T_star": lambda u: ops.T_star(p, R, u, gamma),
    }[which]
    assert worst_ratio(apply, 6, np.random.default_rng(15)) <= gamma + 1e-12
