"""Tests for tabular policy evaluation (study topic #7), written before the implementation.

Implement the two files below from a blank file until these pass:

    uv run pytest rl/tabular

    rl/tabular/gridworld.py     make_gridworld
    rl/tabular/prediction.py    r_pi_P_pi, policy_eval_linear, horizon_for_tolerance, mc_returns

Done when: the linear solve reproduces Sutton & Barto's Figure 3.2, Monte Carlo agrees with
the linear solve within its error bars, and the rollout horizon comes from a bound you derived.
#8 (policy iteration, value iteration) and #9 (TD(0)) will add to this folder.

Write these down BEFORE you write any code
------------------------------------------
1. The tiny MDP from your notes (states A, B, T; stay / go / exit), gamma = 0.9:
       no leak:   V^{pi_stay}(A) = 10     V^{pi_go}(A) = 9     V^{pi_half}(A) = 100/11 ~ 9.09
       leaky exit (B -> A w.p. 0.1), pi_half:   V = [9.81, 10.88, 0]   (A, B, T)
   The tests below use exactly these numbers, so a failure points at code, not at maths.
2. The gridworld (S&B Example 3.5, final 2nd edition): before running the linear solve, guess
   which cell has the highest value under the random policy. Then explain, in one line each,
   why v(A) < 10 even though every action in A pays +10, and why v(B) > 5.
   (Naming clash: S&B's special cells are also called A and B. They have nothing to do with
   the tiny MDP's A and B. In this file they're GRID_A and GRID_B.)
3. Cutting rollouts. The gridworld never ends, so mc_returns stops after `horizon` steps.
   Derive, on paper, the worst-case error from stopping at H when every |reward| <= r_max.
   That bound is what horizon_for_tolerance inverts.
4. Monte Carlo at the top-left cell (s0 = 0): predict the spread (std) of single returns and how
   many episodes you need for the average to land within 0.1 of v(0). Compare after the run.

The contract the tests rely on
------------------------------
Arrays, not classes. S = number of states, A = number of actions.

make_gridworld() -> (p, R)
    p: float array (25, 4, 25), p[s, a, s'] = probability of landing in s' after action a in s.
    R: float array (25, 4),     R[s, a]     = reward for taking action a in s.
    State index s = 5 * row + col, with row 0 at the TOP. Actions: 0 north (row - 1),
    1 south (row + 1), 2 east (col + 1), 3 west (col - 1). Dynamics as in S&B Example 3.5:
    from GRID_A (row 0, col 1) every action moves to (4, 1) with reward +10; from GRID_B
    (row 0, col 3) every action moves to (2, 3) with reward +5; a move off the grid leaves
    you where you are with reward -1; every other move has reward 0.

r_pi_P_pi(p, R, pi) -> (r_pi, P_pi)
    pi: (S, A), pi[s, a] = probability of choosing a in s (rows sum to 1).
    r_pi: (S,),   r_pi[s]     = expected one-step reward in s under pi.
    P_pi: (S, S), P_pi[s, s'] = probability of going from s to s' in one step under pi.
    Rows are "from", columns are "to".

policy_eval_linear(p, R, pi, gamma) -> v
    v: (S,), the exact v_pi from (I - gamma * P_pi) v = r_pi. Use np.linalg.solve; don't
    form the inverse.

horizon_for_tolerance(gamma, r_max, tol) -> int
    The smallest H >= 0 such that, in ANY MDP whose rewards all satisfy |r| <= r_max, the
    expected return of a rollout cut after H steps is guaranteed to be within tol of the
    full (infinite-horizon) value. Point 3 above is the derivation.

mc_returns(p, R, pi, gamma, s0, n_episodes, horizon, rng) -> returns
    returns: float array (n_episodes,). Episode i starts at s_0 = s0 and runs exactly
    `horizon` steps: a_t ~ pi[s_t], r_t = R[s_t, a_t], s_{t+1} ~ p[s_t, a_t]. Its entry is
    G = r_0 + gamma * r_1 + ... + gamma**(horizon - 1) * r_{horizon - 1}.
    Terminal states are absorbing with reward 0, so there's no special end-of-episode case.
    All randomness comes from `rng` (a np.random.Generator); no global seeding.

Suggested order, following the tests top to bottom: gridworld, r_pi_P_pi, policy_eval_linear,
horizon_for_tolerance, mc_returns. Create both files first (prediction.py can start empty), so
the gridworld tests can run before anything else exists:

    uv run pytest rl/tabular -k test_gridworld
"""

import numpy as np
import pytest

import rl.tabular.gridworld as gw
import rl.tabular.prediction as pred

NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
GRID_A, GRID_A_PRIME = 1, 21  # (0, 1) -> (4, 1)
GRID_B, GRID_B_PRIME = 3, 13  # (0, 3) -> (2, 3)

# S&B Figure 3.2 (final 2nd edition): v_pi for the equiprobable random policy, gamma = 0.9,
# as printed in the book (rounded to one decimal).
FIG_3_2 = np.array([
    [3.3, 8.8, 4.4, 5.3, 1.5],
    [1.5, 3.0, 2.3, 1.9, 0.5],
    [0.1, 0.7, 0.7, 0.4, -0.4],
    [-1.0, -0.4, -0.4, -0.6, -1.2],
    [-1.9, -1.3, -1.2, -1.4, -2.0],
])

# The same values to more decimals, so small mistakes can't hide inside the book's rounding.
V_GRID = np.array([
    [3.30900, 8.78929, 4.42762, 5.32237, 1.49218],
    [1.52159, 2.99232, 2.25014, 1.90757, 0.54740],
    [0.05082, 0.73817, 0.67311, 0.35819, -0.40314],
    [-0.97359, -0.43550, -0.35488, -0.58561, -1.18308],
    [-1.85770, -1.34523, -1.22927, -1.42292, -1.97518],
])


def s(row, col):
    return 5 * row + col


def uniform_pi(n_states=25, n_actions=4):
    return np.full((n_states, n_actions), 1.0 / n_actions)


# ---------------------------------------------------------------------------
# The tiny MDP from M1-M3, built here so the tests don't depend on your gridworld.
# States: 0 = A, 1 = B, 2 = T (absorbing, reward 0). Two action slots everywhere:
#   A: 0 = stay (+1, back to A), 1 = go (0, to B)
#   B: both slots are exit (+10; to A w.p. leak, to T otherwise)
#   T: both slots stay in T with reward 0
# ---------------------------------------------------------------------------
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
    """Choose stay in A with probability p_stay; B and T have only one real action."""
    return np.array([[p_stay, 1.0 - p_stay], [1.0, 0.0], [1.0, 0.0]])


PI_STAY, PI_GO, PI_HALF = tiny_pi(1.0), tiny_pi(0.0), tiny_pi(0.5)


# ---------------------------------------------------------------------------
# 1. make_gridworld
# ---------------------------------------------------------------------------
def test_gridworld_shapes():
    p, R = gw.make_gridworld()
    assert p.shape == (25, 4, 25)
    assert R.shape == (25, 4)
    assert np.issubdtype(p.dtype, np.floating) and np.issubdtype(R.dtype, np.floating)


def test_gridworld_p_is_a_distribution_for_every_state_action():
    p, _ = gw.make_gridworld()
    assert (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=2), 1.0, atol=1e-12)


def test_gridworld_is_deterministic():
    p, _ = gw.make_gridworld()
    # Every (s, a) lands in exactly one next state.
    assert ((p == 1.0).sum(axis=2) == 1).all()


@pytest.mark.parametrize("a", [NORTH, SOUTH, EAST, WEST])
def test_gridworld_special_cell_A(a):
    p, R = gw.make_gridworld()
    assert p[GRID_A, a, GRID_A_PRIME] == 1.0
    assert R[GRID_A, a] == 10.0


@pytest.mark.parametrize("a", [NORTH, SOUTH, EAST, WEST])
def test_gridworld_special_cell_B(a):
    p, R = gw.make_gridworld()
    assert p[GRID_B, a, GRID_B_PRIME] == 1.0
    assert R[GRID_B, a] == 5.0


@pytest.mark.parametrize(
    "state, action, next_state, reward",
    [
        # top-left corner: north and west bump the wall
        (s(0, 0), NORTH, s(0, 0), -1.0),
        (s(0, 0), WEST, s(0, 0), -1.0),
        (s(0, 0), SOUTH, s(1, 0), 0.0),
        (s(0, 0), EAST, s(0, 1), 0.0),
        # bottom-right corner: south and east bump the wall
        (s(4, 4), SOUTH, s(4, 4), -1.0),
        (s(4, 4), EAST, s(4, 4), -1.0),
        (s(4, 4), NORTH, s(3, 4), 0.0),
        (s(4, 4), WEST, s(4, 3), 0.0),
        # an ordinary top-edge cell (not A or B)
        (s(0, 2), NORTH, s(0, 2), -1.0),
        # the centre: all four moves work
        (s(2, 2), NORTH, s(1, 2), 0.0),
        (s(2, 2), SOUTH, s(3, 2), 0.0),
        (s(2, 2), EAST, s(2, 3), 0.0),
        (s(2, 2), WEST, s(2, 1), 0.0),
    ],
)
def test_gridworld_moves_and_walls(state, action, next_state, reward):
    p, R = gw.make_gridworld()
    assert p[state, action, next_state] == 1.0, "wrong next state (check the action order and s = 5*row + col)"
    assert R[state, action] == reward


# ---------------------------------------------------------------------------
# 2. r_pi_P_pi
# ---------------------------------------------------------------------------
def test_r_pi_P_pi_tiny_leaky_matches_your_hand_computation():
    # M5 step 1: R_pi = [0.5, 10] and P_pi = [[0.5, 0.5], [0.1, 0]], now with T kept as a state.
    p, R = tiny_mdp(leak=0.1)
    r_pi, P_pi = pred.r_pi_P_pi(p, R, PI_HALF)
    np.testing.assert_allclose(r_pi, [0.5, 10.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        P_pi,
        [[0.5, 0.5, 0.0],
         [0.1, 0.0, 0.9],
         [0.0, 0.0, 1.0]],
        atol=1e-12,
        err_msg="rows must be 'from', columns 'to'",
    )


def test_r_pi_P_pi_shapes_and_rows_sum_to_one_on_gridworld():
    p, R = gw.make_gridworld()
    r_pi, P_pi = pred.r_pi_P_pi(p, R, uniform_pi())
    assert r_pi.shape == (25,)
    assert P_pi.shape == (25, 25)
    np.testing.assert_allclose(P_pi.sum(axis=1), 1.0, atol=1e-12)


def test_r_pi_weights_by_pi_not_a_plain_average():
    # A non-uniform policy, so "average over actions" and "weight by pi" give different answers.
    p, R = gw.make_gridworld()
    pi = np.zeros((25, 4))
    pi[:, NORTH] = 0.7
    pi[:, EAST] = 0.3
    r_pi, P_pi = pred.r_pi_P_pi(p, R, pi)
    # top-left: north bumps the wall (-1) 70% of the time, east moves to GRID_A (0) 30%.
    assert r_pi[s(0, 0)] == pytest.approx(-0.7)
    assert P_pi[s(0, 0), s(0, 0)] == pytest.approx(0.7)
    assert P_pi[s(0, 0), s(0, 1)] == pytest.approx(0.3)
    # GRID_A pays +10 and jumps to GRID_A_PRIME whatever the policy says.
    assert r_pi[GRID_A] == pytest.approx(10.0)
    assert P_pi[GRID_A, GRID_A_PRIME] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. policy_eval_linear
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "leak, pi, gamma, expected",
    [
        # M1 and M3, no leak
        (0.0, PI_STAY, 0.9, [10.0, 10.0, 0.0]),
        (0.0, PI_GO, 0.9, [9.0, 10.0, 0.0]),
        (0.0, PI_HALF, 0.9, [100 / 11, 10.0, 0.0]),
        # M4: at gamma = 0.5 the ranking flips
        (0.0, PI_STAY, 0.5, [2.0, 10.0, 0.0]),
        (0.0, PI_GO, 0.5, [5.0, 10.0, 0.0]),
        # M3: leaky exit
        (0.1, PI_HALF, 0.9, [9.81354269, 10.88321884, 0.0]),
        # a policy you haven't solved by hand: 80/20 stay/go, leaky exit
        (0.1, tiny_pi(0.8), 0.9, [9.85595, 10.88704, 0.0]),
    ],
)
def test_policy_eval_linear_tiny_mdp(leak, pi, gamma, expected):
    p, R = tiny_mdp(leak)
    v = pred.policy_eval_linear(p, R, pi, gamma)
    assert v.shape == (3,)
    np.testing.assert_allclose(v, expected, atol=1e-4)


def test_policy_eval_linear_reproduces_figure_3_2():
    p, R = gw.make_gridworld()
    v = pred.policy_eval_linear(p, R, uniform_pi(), 0.9).reshape(5, 5)
    # The book rounds to one decimal, so every true value is within 0.05 of what's printed.
    np.testing.assert_allclose(v, FIG_3_2, atol=0.05 + 1e-9)


def test_policy_eval_linear_gridworld_full_precision():
    p, R = gw.make_gridworld()
    v = pred.policy_eval_linear(p, R, uniform_pi(), 0.9).reshape(5, 5)
    np.testing.assert_allclose(v, V_GRID, atol=1e-4)


def test_policy_eval_linear_satisfies_the_bellman_equation():
    # Whatever the policy, v must satisfy v = r_pi + gamma * P_pi v.
    p, R = gw.make_gridworld()
    rng = np.random.default_rng(7)
    pi = rng.dirichlet(np.ones(4), size=25)
    r_pi, P_pi = pred.r_pi_P_pi(p, R, pi)
    for gamma in (0.0, 0.5, 0.9, 0.99):
        v = pred.policy_eval_linear(p, R, pi, gamma)
        np.testing.assert_allclose(v, r_pi + gamma * P_pi @ v, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. horizon_for_tolerance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gamma, r_max, tol, expected",
    [
        (0.9, 10.0, 0.01, 88),
        (0.9, 10.0, 1.0, 44),
        (0.5, 1.0, 0.1, 5),
        (0.99, 1.0, 0.01, 917),
        (0.9, 10.0, 1000.0, 0),  # loose enough that no steps are needed at all
    ],
)
def test_horizon_for_tolerance_values(gamma, r_max, tol, expected):
    H = pred.horizon_for_tolerance(gamma, r_max, tol)
    assert isinstance(H, (int, np.integer))
    assert H == expected


@pytest.mark.parametrize(
    "gamma, expected",  # tol = 0.003, 0.05, 0.7 with r_max = 3
    [
        (0.5, (11, 7, 4)),
        (0.8, (39, 26, 14)),
        (0.9, (88, 61, 36)),
        (0.95, (194, 139, 87)),
    ],
)
def test_horizon_for_tolerance_grid(gamma, expected):
    got = tuple(pred.horizon_for_tolerance(gamma, 3.0, tol) for tol in (0.003, 0.05, 0.7))
    assert got == expected, "off by one? check whether H is the smallest horizon that's safe"


# ---------------------------------------------------------------------------
# 5. mc_returns
# ---------------------------------------------------------------------------
def test_mc_returns_shape():
    p, R = tiny_mdp(leak=0.1)
    G = pred.mc_returns(p, R, PI_HALF, 0.9, 0, 50, 20, np.random.default_rng(0))
    assert G.shape == (50,)
    assert np.isfinite(G).all()


def test_mc_returns_horizon_zero_collects_nothing():
    p, R = tiny_mdp(leak=0.1)
    G = pred.mc_returns(p, R, PI_HALF, 0.9, 0, 20, 0, np.random.default_rng(0))
    np.testing.assert_array_equal(G, np.zeros(20))


def test_mc_returns_horizon_one_is_the_first_reward_only():
    # From A under pi_half: stay pays 1, go pays 0, and nothing after step 0 counts.
    p, R = tiny_mdp(leak=0.1)
    G = pred.mc_returns(p, R, PI_HALF, 0.9, 0, 2000, 1, np.random.default_rng(1))
    assert set(np.round(G, 9)) <= {0.0, 1.0}, "horizon=1 should collect exactly one reward"
    assert G.mean() == pytest.approx(0.5, abs=0.05)  # = r_pi(A)


def test_mc_returns_horizon_two_discounts_the_second_reward():
    # Two steps from A: stay,stay = 1 + 0.9*1; stay,go = 1 + 0.9*0; go,exit = 0 + 0.9*10.
    p, R = tiny_mdp(leak=0.1)
    G = pred.mc_returns(p, R, PI_HALF, 0.9, 0, 2000, 2, np.random.default_rng(2))
    assert set(np.round(G, 9)) <= {1.9, 1.0, 9.0}, "check where gamma**t starts and how many rewards you sum"
    assert set(np.round(G, 9)) == {1.9, 1.0, 9.0}  # all three paths should turn up in 2000 tries


def test_mc_returns_uses_only_the_rng_you_pass():
    p, R = tiny_mdp(leak=0.1)
    run = lambda seed: pred.mc_returns(p, R, PI_HALF, 0.9, 0, 200, 30, np.random.default_rng(seed))
    np.testing.assert_array_equal(run(3), run(3))
    assert not np.array_equal(run(3), run(4))


def test_mc_returns_follows_pi_not_a_uniform_choice():
    # "Always north" from the top-left corner bumps the wall every step: every return is
    # -(1 + 0.9 + ... + 0.9**99), whatever the rng does. A uniformly random action would
    # average about +3.3 instead.
    p, R = gw.make_gridworld()
    pi = np.zeros((25, 4))
    pi[:, NORTH] = 1.0
    G = pred.mc_returns(p, R, pi, 0.9, s(0, 0), 20, 100, np.random.default_rng(17))
    expected = -sum(0.9**t for t in range(100))
    np.testing.assert_allclose(G, expected, atol=1e-9)


def test_mc_returns_tiny_leaky_distribution():
    # M5 step 3: ~45% of episodes are "go, exit, end" = exactly 9, and nothing is below 9.
    p, R = tiny_mdp(leak=0.1)
    G = pred.mc_returns(p, R, PI_HALF, 0.9, 0, 4000, 60, np.random.default_rng(5))
    assert G.min() >= 9.0 - 1e-9
    assert np.mean(np.isclose(G, 9.0)) == pytest.approx(0.45, abs=0.03)


@pytest.mark.parametrize("p_stay, v_exact", [(0.5, 9.81354269), (0.8, 9.85595)])
def test_mc_returns_tiny_leaky_mean_matches_linear_solve(p_stay, v_exact):
    # Horizon 60: by then almost every episode is sitting in T, so the cut costs ~nothing here.
    p, R = tiny_mdp(leak=0.1)
    G = pred.mc_returns(p, R, tiny_pi(p_stay), 0.9, 0, 4000, 60, np.random.default_rng(11))
    se = G.std() / np.sqrt(len(G))
    assert abs(G.mean() - v_exact) <= 4 * se, f"MC mean {G.mean():.3f} vs exact {v_exact:.3f}, SE {se:.3f}"


def test_mc_returns_gridworld_matches_linear_solve():
    # Top-left cell, random policy, horizon 100. Your bound from point 3 says what the cut can
    # cost at H = 100 here (r_max = 10, gamma = 0.9); it should come out under 0.003.
    p, R = gw.make_gridworld()
    G = pred.mc_returns(p, R, uniform_pi(), 0.9, s(0, 0), 3000, 100, np.random.default_rng(13))
    se = G.std() / np.sqrt(len(G))
    v_exact = V_GRID[0, 0]
    cut = 0.003
    assert abs(G.mean() - v_exact) <= 4 * se + cut, (
        f"MC mean {G.mean():.3f} vs exact {v_exact:.3f}, SE {se:.3f}"
    )
