"""5x5 gridworld from Sutton & Barto, Example 3.5 (final 2nd edition), stored as arrays.

    p[s, a, s'] = probability of landing in s' after taking action a in s    shape (25, 4, 25)
    R[s, a]     = reward for taking action a in s                            shape (25, 4)

Fill in every TODO. Check with:  uv run pytest rl/tabular -k test_gridworld
"""

import numpy as np

N_ROWS, N_COLS = 5, 5
N_STATES = N_ROWS * N_COLS
N_ACTIONS = 4  # 0 north, 1 south, 2 east, 3 west

# (d_row, d_col) for each action, listed in action order. Row 0 is the TOP of the grid.
# MOVES = ...  # TODO
MOVES = [(-1,0), (1,0), (0,1), (0,-1)]

# Special cells as (row, col). From the cell, EVERY action jumps to its target and pays the bonus.
A_CELL, A_TARGET, A_BONUS = (0, 1), (4, 1), 10.0
B_CELL, B_TARGET, B_BONUS = (0, 3), (2, 3), 5.0
WALL_REWARD = -1.0  # a move off the grid leaves you where you are

# give it a cell’s position, (row, col), and it returns one number, the state index s.
def to_state(row, col):
    """(row, col) -> state index s."""
    # ...  # TODO
    return row*N_COLS + col


def make_gridworld():
    """Return (p, R) for the gridworld. See the module docstring for shapes."""
    p = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS))
    for row in range(N_ROWS):
        for col in range(N_COLS):
            s = to_state(row, col)
            for a in range(N_ACTIONS):
                # TODO: set next_row, next_col and reward for taking action a in (row, col).
                # Three rules: special cell (A or B), wall bump, ordinary move.
                # Which rule must be checked first? (North from A goes off the grid.)
                # ...
                if (row, col) == A_CELL:
                    next_row, next_col = A_TARGET
                    reward = A_BONUS
                elif (row, col) == B_CELL:
                    next_row, next_col = B_TARGET
                    reward = B_BONUS
                else:
                    d_row, d_col = MOVES[a]
                    next_row, next_col = row+d_row, col+d_col    # where the move would take you
                    reward = 0.0
                    if not ((0 <= next_row < N_ROWS) and (0 <= next_col < N_COLS)):     # is that cell off the grid?
                        next_row, next_col = row, col      # stay where you are
                        reward = -1
                p[s, a, to_state(next_row, next_col)] = 1.0
                R[s, a] = reward
    return p, R