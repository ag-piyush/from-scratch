"""Tests for iterative policy evaluation and policy iteration (study topic #8, M3), written before
the implementation.

Implement this file from a blank file until these pass:

    uv run pytest rl/tabular/test_dp.py

    rl/tabular/dp.py     greedy_policy, policy_eval_iterative, policy_iteration

Done when: the sweep counts match what you wrote down (or you can say why they don't), iterative
evaluation lands within its stopping bound of the linear solve, and policy iteration reproduces
Figure 3.5 and beats every deterministic policy on a small random MDP.
Next in this folder: value_iteration and the gambler's problem (M4).

Write these down BEFORE you write any code
------------------------------------------
The tiny MDP from block 10 (A, B, T; stay / go / exit), no leak, gamma = 0.9, v0 = 0, tol = 1e-10.
A "sweep" updates every state once. policy_eval_iterative stops after the first sweep whose
largest change, max over s of |v_new(s) - v_old(s)|, is below tol. It reports how many sweeps it
ran, counting that last one.

1. Sweep counts for pi_go: (a) two-array; (b) in-place, states in index order A, B, T;
   (c) in-place in the order T, B, A. Your block-10 notes say after how many sweeps v is exact.
   Is that the number the function reports?
2. pi_stay at tol = 1e-6: how many sweeps, and how far is V(A) from 10 when it stops? Is that
   gap smaller or bigger than tol, and by roughly what factor? Does in-place save any sweeps
   here? Why or why not?
3. Gridworld, random policy (Figure 3.2), tol = 1e-8.
   (a) Bound the number of two-array sweeps: what is the largest change in sweep 1 from v0 = 0,
       and at least how fast must the change shrink from sweep to sweep?
   (b) Guess the number of in-place sweeps: about the same, a bit fewer, or half?
   (c) When it stops, is the error ||v - v_pi||_inf close to the bound you used in 2, or well
       inside it?
   Then run:  uv run pytest rl/tabular/test_dp.py -k sweep_counts -s
4. Policy iteration. Count policy evaluations, including the last one, which finds the policy
   stable.
   (a) Tiny MDP, gamma = 0.9, from pi_go (your block-10 run).
   (b) Tiny MDP, gamma = 0.5, from pi_stay.
   (c) Tiny MDP, gamma = 0.9, from the optimal policy, but with B (and T) on slot 1 instead of
       slot 0. B's two slots are identical exits. And what would (c) give if greedy_policy
       used plain argmax, with no tie rule?
   (d) Gridworld, gamma = 0.9, from "always north". There are 4^25, about 10^15, deterministic
       policies. Guess the number of evaluations: 2, 5, 20 or 100?
   Then run:  uv run pytest rl/tabular/test_dp.py -k policy_iteration_gridworld -s

The contract the tests rely on
------------------------------
Same arrays as prediction.py and operators.py. S = number of states, A = number of actions.
    p[s, a, s']   transition probabilities     shape (S, A, S)
    R[s, a]       rewards                      shape (S, A)
    pi[s, a]      policy, each row sums to 1   shape (S, A)
    q[s, a]       action values                shape (S, A)
    v[s]          state values                 shape (S,)
A deterministic policy is stored like any other pi, with a single 1.0 in each row ("one-hot"),
so it plugs straight into T_pi and policy_eval_linear. Its action in s is pi[s].argmax().

greedy_policy(q, pi_current=None, tol=1e-9) -> pi
    pi: float (S, A), one-hot. In each state s, let best = max over a of q[s, a].
    - If pi_current (one-hot) is given and its action c = pi_current[s].argmax() has
      q[s, c] >= best - tol, keep c. Only an action better by MORE than tol makes it change.
    - Otherwise take the first action that attains best (what np.argmax returns).

policy_eval_iterative(p, R, pi, gamma, tol, in_place=False, order=None, v0=None,
                      max_sweeps=100_000) -> (v, n_sweeps)
    Start from v0 (zeros if None). Sweep until a sweep's largest change is < tol (tol > 0).
    - Two-array (in_place=False): every new value is computed from the old vector, so a sweep
      is one application of T_pi. `order` is ignored: it can't matter here.
    - In-place (in_place=True): update v[s] one state at a time, visiting states in `order`
      (default 0, 1, ..., S-1). A state updated later in a sweep reads the values already
      updated earlier in that same sweep.
    v: float (S,). n_sweeps: Python int, the number of sweeps run, including the one that
    stopped it. Leave every input unchanged, v0 included. Raise RuntimeError if max_sweeps
    sweeps run without getting below tol.

policy_iteration(p, R, gamma, pi0=None, tol=1e-9, max_iters=1000) -> (pi, v, n_evals)
    Start from pi0 (deterministic; default: action 0 in every state) and repeat:
        evaluate pi exactly with policy_eval_linear;
        improve: pi_new = greedy_policy(q_from_v(p, R, v, gamma), pi, tol);
        stop if pi_new is the same policy as pi.
    pi: the final one-hot policy. v: its value. n_evals: Python int, the number of evaluations
    run, including the last one. Leave pi0 unchanged. Raise RuntimeError if max_iters
    evaluations run without a stable policy.

Suggested order, top to bottom:
    uv run pytest rl/tabular/test_dp.py -k greedy
    uv run pytest rl/tabular/test_dp.py -k eval_iterative
    uv run pytest rl/tabular/test_dp.py -k policy_iteration
"""

import itertools

import numpy as np
import pytest

import rl.tabular.dp as dp
import rl.tabular.gridworld as gw
import rl.tabular.operators as ops
import rl.tabular.prediction as pred

# S&B Figure 3.5 (final 2nd edition), middle panel: v* for the gridworld, gamma = 0.9,
# as printed (one decimal), and the same v* to more decimals.
FIG_3_5 = np.array([
    [22.0, 24.4, 22.0, 19.4, 17.5],
    [19.8, 22.0, 19.8, 17.8, 16.0],
    [17.8, 19.8, 17.8, 16.0, 14.4],
    [16.0, 17.8, 16.0, 14.4, 13.0],
    [14.4, 16.0, 14.4, 13.0, 11.7],
])
V_STAR_GRID = np.array([
    [21.97749, 24.41943, 21.97749, 19.41943, 17.47749],
    [19.77974, 21.97749, 19.77974, 17.80176, 16.02159],
    [17.80176, 19.77974, 17.80176, 16.02159, 14.41943],
    [16.02159, 17.80176, 16.02159, 14.41943, 12.97749],
    [14.41943, 16.02159, 14.41943, 12.97749, 11.67974],
])


def uniform_pi(n_states=25, n_actions=4):
    return np.full((n_states, n_actions), 1.0 / n_actions)


def one_hot(actions, n_actions):
    return np.eye(n_actions)[np.asarray(actions)]


def actions_of(pi):
    return pi.argmax(axis=1)


# The tiny MDP again (same as test_operators.py), so these tests don't depend on that file.
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


PI_STAY = one_hot([0, 0, 0], 2)
PI_GO = one_hot([1, 0, 0], 2)
TOL_TINY = 1e-10


def random_mdp(n_states, n_actions, seed):
    """Stochastic transitions and random rewards: nothing special to hide a bug behind."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.normal(0.0, 1.0, size=(n_states, n_actions))
    return p, R


def random_deterministic_pi(n_states, n_actions, rng):
    return one_hot(rng.integers(n_actions, size=n_states), n_actions)


# ---------------------------------------------------------------------------
# 1. greedy_policy
# ---------------------------------------------------------------------------
def test_greedy_policy_shape_and_one_hot():
    q = np.random.default_rng(0).normal(size=(6, 4))
    pi = dp.greedy_policy(q)
    assert pi.shape == (6, 4)
    assert np.issubdtype(pi.dtype, np.floating)
    assert set(np.unique(pi)) <= {0.0, 1.0}
    np.testing.assert_array_equal(pi.sum(axis=1), np.ones(6))


def test_greedy_policy_picks_the_argmax():
    q = np.array([[1.0, 3.0, 2.0], [0.0, -1.0, -2.0], [5.0, 5.5, 7.0]])
    np.testing.assert_array_equal(actions_of(dp.greedy_policy(q)), [1, 0, 2])


def test_greedy_policy_without_current_breaks_ties_by_first_index():
    q = np.array([[2.0, 2.0, 1.0], [0.0, 3.0, 3.0]])
    np.testing.assert_array_equal(actions_of(dp.greedy_policy(q)), [0, 1])


def test_greedy_policy_keeps_the_current_action_on_a_tie():
    q = np.array([[2.0, 2.0, 1.0], [0.0, 3.0, 3.0]])
    current = one_hot([1, 2], 3)
    np.testing.assert_array_equal(actions_of(dp.greedy_policy(q, current)), [1, 2])


def test_greedy_policy_tol_decides_what_counts_as_strictly_better():
    q = np.array([[1.0, 1.0 + 1e-12]])
    current = one_hot([0], 2)
    # Inside the default tol: a rounding-error "improvement" is not a reason to switch.
    np.testing.assert_array_equal(actions_of(dp.greedy_policy(q, current)), [0])
    # With tol = 0, any strictly larger q wins.
    np.testing.assert_array_equal(actions_of(dp.greedy_policy(q, current, tol=0.0)), [1])
    # A clear improvement always wins.
    q_clear = np.array([[1.0, 1.1]])
    np.testing.assert_array_equal(actions_of(dp.greedy_policy(q_clear, current)), [1])


def test_greedy_policy_switches_to_the_best_action_not_just_a_better_one():
    q = np.array([[5.0, 1.0, 3.0]])
    np.testing.assert_array_equal(actions_of(dp.greedy_policy(q, one_hot([1], 3))), [0])


@pytest.mark.parametrize(
    "gamma, v, pi_current, expected",
    [
        # Block 10, round 1: from v_go = [9, 10, 0], q(A, stay) = 9.1 > q(A, go) = 9.
        (0.9, [9.0, 10.0, 0.0], PI_GO, [0, 0, 0]),
        # gamma = 0.5, from v_stay = [2, 10, 0]: q(A, stay) = 2 < q(A, go) = 5.
        (0.5, [2.0, 10.0, 0.0], PI_STAY, [1, 0, 0]),
    ],
)
def test_greedy_policy_tiny_mdp(gamma, v, pi_current, expected):
    p, R = tiny_mdp()
    q = ops.q_from_v(p, R, np.array(v), gamma)
    np.testing.assert_array_equal(actions_of(dp.greedy_policy(q, pi_current)), expected)


def test_greedy_policy_does_not_modify_its_inputs():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(5, 3))
    current = random_deterministic_pi(5, 3, rng)
    copies = q.copy(), current.copy()
    dp.greedy_policy(q, current)
    np.testing.assert_array_equal(q, copies[0])
    np.testing.assert_array_equal(current, copies[1])


@pytest.mark.parametrize("seed", range(5))
def test_greedy_policy_step_obeys_the_improvement_theorem(seed):
    # The theorem from block 10: pi' greedy in q_pi  =>  v_pi' >= v_pi in every state, and if
    # pi' changed anywhere, some state gets strictly better.
    p, R = random_mdp(6, 4, seed=100 + seed)
    rng = np.random.default_rng(200 + seed)
    for _ in range(10):
        pi = random_deterministic_pi(6, 4, rng)
        v = pred.policy_eval_linear(p, R, pi, 0.9)
        pi_new = dp.greedy_policy(ops.q_from_v(p, R, v, 0.9), pi)
        v_new = pred.policy_eval_linear(p, R, pi_new, 0.9)
        assert (v_new >= v - 1e-10).all()
        if not np.array_equal(pi_new, pi):
            assert (v_new - v).max() > 1e-10


# ---------------------------------------------------------------------------
# 2. policy_eval_iterative
# ---------------------------------------------------------------------------
def test_policy_eval_iterative_returns_a_float_vector_and_an_int():
    p, R = gw.make_gridworld()
    v, n = dp.policy_eval_iterative(p, R, uniform_pi(), 0.9, 1e-6)
    assert v.shape == (25,)
    assert np.issubdtype(v.dtype, np.floating)
    assert isinstance(n, int)


def test_policy_eval_iterative_one_two_array_sweep_is_T_pi():
    # A huge tol stops after one sweep. Two-array: that sweep is exactly T_pi(v0).
    p, R = random_mdp(5, 3, seed=3)
    rng = np.random.default_rng(4)
    pi = rng.dirichlet(np.ones(3), size=5)
    v0 = rng.normal(0.0, 10.0, size=5)
    v, n = dp.policy_eval_iterative(p, R, pi, 0.9, tol=1e9, v0=v0)
    assert n == 1
    np.testing.assert_allclose(v, ops.T_pi(p, R, pi, v0, 0.9), atol=1e-12)


@pytest.mark.parametrize(
    "in_place, order, expected",
    [
        (False, None, [0.0, 10.0, 0.0]),
        (True, None, [0.0, 10.0, 0.0]),       # A is updated before B, so it still reads B = 0
        (True, [2, 1, 0], [9.0, 10.0, 0.0]),  # B first, then A reads the new B
    ],
)
def test_policy_eval_iterative_one_sweep_tiny_go(in_place, order, expected):
    p, R = tiny_mdp()
    v, n = dp.policy_eval_iterative(p, R, PI_GO, 0.9, tol=1e9, in_place=in_place, order=order)
    assert n == 1
    np.testing.assert_allclose(v, expected, atol=1e-12)


@pytest.mark.parametrize(
    "in_place, order, n_expected",
    [
        # Prediction 1. Don't read these before you've written yours.
        (False, None, 3),
        (True, None, 3),
        (True, [2, 1, 0], 2),
        (False, [2, 1, 0], 3),  # two-array ignores order
    ],
)
def test_policy_eval_iterative_tiny_go_sweep_counts(in_place, order, n_expected):
    p, R = tiny_mdp()
    v, n = dp.policy_eval_iterative(p, R, PI_GO, 0.9, TOL_TINY, in_place=in_place, order=order)
    np.testing.assert_allclose(v, [9.0, 10.0, 0.0], atol=1e-12)
    assert n == n_expected, f"ran {n} sweeps, expected {n_expected}"


@pytest.mark.parametrize("in_place", [False, True])
def test_policy_eval_iterative_tiny_stay_stops_short_of_10(in_place):
    # Prediction 2.
    p, R = tiny_mdp()
    tol, gamma = 1e-6, 0.9
    v, n = dp.policy_eval_iterative(p, R, PI_STAY, gamma, tol, in_place=in_place)
    assert n == 133, f"ran {n} sweeps, expected 133"
    gap = 10.0 - v[0]
    assert gap == pytest.approx(10.0 * gamma**133, rel=1e-6)
    # The gap is bigger than tol, but no bigger than gamma / (1 - gamma) * tol.
    assert tol < gap <= gamma / (1.0 - gamma) * tol


def test_policy_eval_iterative_gridworld_sweep_counts():
    # Prediction 3. Run with -s to see the counts and how close each error comes to its bound.
    p, R = gw.make_gridworld()
    gamma, tol = 0.9, 1e-8
    v_exact = pred.policy_eval_linear(p, R, uniform_pi(), gamma)
    bound = gamma / (1.0 - gamma) * tol
    counts = {}
    for in_place in (False, True):
        v, n = dp.policy_eval_iterative(p, R, uniform_pi(), gamma, tol, in_place=in_place)
        err = np.abs(v - v_exact).max()
        name = "in-place" if in_place else "two-array"
        print(f"\n{name:>9}: {n} sweeps, error {err:.2e} = {err / bound:.2f} x the bound {bound:.0e}")
        assert err <= bound + 1e-12
        counts[name] = n
    # Change in sweep 1 is at most 10, and it shrinks by at least gamma per sweep.
    assert counts["two-array"] <= 198
    assert counts["in-place"] < counts["two-array"]


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
@pytest.mark.parametrize("mode", ["two-array", "in-place", "in-place, shuffled"])
def test_policy_eval_iterative_lands_within_its_bound_of_the_linear_solve(mode, gamma):
    p, R = random_mdp(6, 3, seed=5)
    rng = np.random.default_rng(6)
    pi = rng.dirichlet(np.ones(3), size=6)
    order = rng.permutation(6) if mode == "in-place, shuffled" else None
    tol = 1e-8
    v, _ = dp.policy_eval_iterative(p, R, pi, gamma, tol, in_place=(mode != "two-array"), order=order)
    v_exact = pred.policy_eval_linear(p, R, pi, gamma)
    assert np.abs(v - v_exact).max() <= gamma / (1.0 - gamma) * tol + 1e-12


@pytest.mark.parametrize("in_place", [False, True])
def test_policy_eval_iterative_starting_at_v_pi_takes_one_sweep(in_place):
    p, R = gw.make_gridworld()
    v_exact = pred.policy_eval_linear(p, R, uniform_pi(), 0.9)
    v, n = dp.policy_eval_iterative(p, R, uniform_pi(), 0.9, 1e-10, in_place=in_place, v0=v_exact)
    assert n == 1
    np.testing.assert_allclose(v, v_exact, atol=1e-10)


@pytest.mark.parametrize("in_place", [False, True])
def test_policy_eval_iterative_does_not_modify_its_inputs(in_place):
    p, R = random_mdp(4, 3, seed=7)
    rng = np.random.default_rng(8)
    pi = rng.dirichlet(np.ones(3), size=4)
    v0 = rng.normal(size=4)
    copies = [x.copy() for x in (p, R, pi, v0)]
    dp.policy_eval_iterative(p, R, pi, 0.9, 1e-8, in_place=in_place, v0=v0)
    for original, copy in zip((p, R, pi, v0), copies):
        np.testing.assert_array_equal(original, copy)


def test_policy_eval_iterative_raises_when_it_runs_out_of_sweeps():
    p, R = gw.make_gridworld()
    with pytest.raises(RuntimeError):
        dp.policy_eval_iterative(p, R, uniform_pi(), 0.99, 1e-12, max_sweeps=5)


# ---------------------------------------------------------------------------
# 3. policy_iteration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "leak, gamma, pi0, actions_expected, v_star, n_expected",
    [
        # Prediction 4 (a) and (b). Don't read these before you've written yours.
        (0.0, 0.9, PI_GO, [0, 0, 0], [10.0, 10.0, 0.0], 2),
        (0.0, 0.5, PI_STAY, [1, 0, 0], [5.0, 10.0, 0.0], 2),
        (0.1, 0.9, PI_GO, [0, 0, 0], [10.0, 10.9, 0.0], 2),
        (0.0, 0.9, None, [0, 0, 0], [10.0, 10.0, 0.0], 1),  # default start is already optimal
    ],
)
def test_policy_iteration_tiny_mdp(leak, gamma, pi0, actions_expected, v_star, n_expected):
    p, R = tiny_mdp(leak)
    pi, v, n = dp.policy_iteration(p, R, gamma, pi0=pi0)
    np.testing.assert_array_equal(pi, one_hot(actions_expected, 2))
    np.testing.assert_allclose(v, v_star, atol=1e-10)
    assert isinstance(n, int)
    assert n == n_expected, f"ran {n} evaluations, expected {n_expected}"


def test_policy_iteration_keeps_a_tied_optimal_policy():
    # Prediction 4 (c). B and T on slot 1, whose q ties exactly with slot 0.
    p, R = tiny_mdp()
    pi0 = one_hot([0, 1, 1], 2)
    pi, v, n = dp.policy_iteration(p, R, 0.9, pi0=pi0)
    np.testing.assert_array_equal(pi, pi0)
    np.testing.assert_allclose(v, [10.0, 10.0, 0.0], atol=1e-10)
    assert n == 1, f"ran {n} evaluations: the policy changed on a tie"


def test_policy_iteration_gridworld_reproduces_figure_3_5():
    # Prediction 4 (d). Run with -s to see the count.
    p, R = gw.make_gridworld()
    pi, v, n = dp.policy_iteration(p, R, 0.9)
    print(f"\npolicy iteration from 'always north': {n} evaluations")
    assert set(np.unique(pi)) <= {0.0, 1.0}
    np.testing.assert_array_equal(pi.sum(axis=1), np.ones(25))
    np.testing.assert_allclose(v, V_STAR_GRID.ravel(), atol=1e-4)
    np.testing.assert_allclose(v, FIG_3_5.ravel(), atol=0.05)
    np.testing.assert_allclose(ops.T_star(p, R, v, 0.9), v, atol=1e-8)
    assert n == 3, f"ran {n} evaluations, expected 3"


@pytest.mark.parametrize("seed", range(5))
def test_policy_iteration_beats_every_deterministic_policy(seed):
    # S = 4, A = 3: only 3^4 = 81 deterministic policies, so check them all.
    p, R = random_mdp(4, 3, seed=300 + seed)
    gamma = 0.9
    all_values = np.array([
        pred.policy_eval_linear(p, R, one_hot(actions, 3), gamma)
        for actions in itertools.product(range(3), repeat=4)
    ])
    pi, v, n = dp.policy_iteration(p, R, gamma)
    np.testing.assert_allclose(v, pred.policy_eval_linear(p, R, pi, gamma), atol=1e-10)
    assert (v >= all_values - 1e-9).all()
    np.testing.assert_allclose(v, all_values.max(axis=0), atol=1e-9)
    assert 1 <= n <= 81


def test_policy_iteration_does_not_modify_pi0():
    p, R = random_mdp(5, 3, seed=9)
    pi0 = random_deterministic_pi(5, 3, np.random.default_rng(10))
    copy = pi0.copy()
    dp.policy_iteration(p, R, 0.9, pi0=pi0)
    np.testing.assert_array_equal(pi0, copy)
