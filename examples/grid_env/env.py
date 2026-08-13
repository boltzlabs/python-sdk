"""A 5x5 grid world — the smallest environment a policy can actually learn.

Walk to the goal. Four actions (0 up, 1 right, 2 down, 3 left), -0.01 a step so
dawdling costs something, +1 on arrival, truncated at 50 steps.

It returns Gymnasium's 5-tuple, so it also demonstrates the thing that is easy
to get wrong at this boundary: the wire has one `done` flag, and `serve` records
`terminated` and `truncated` separately in `info` so a value function can still
tell "the episode ended" from "we stopped watching". `BoltzLabsVecEnv` reads them
back out.

Pure Python on purpose — no numpy. The sandbox mounts this directory and an
interpreter, nothing else.
"""

import random

from boltzlabs.env import serve

SIZE = 5
GOAL = (4, 4)
MAX_STEPS = 50
MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

state = {"pos": (0, 0), "t": 0, "rng": random.Random()}


def _obs():
    r, c = state["pos"]
    gr, gc = GOAL
    # Position and the vector to the goal, normalised — a policy can learn from
    # this without knowing the grid size.
    return [r / (SIZE - 1), c / (SIZE - 1), (gr - r) / (SIZE - 1), (gc - c) / (SIZE - 1)]


def reset(seed):
    rng = random.Random(seed)
    state["rng"] = rng
    state["t"] = 0
    # Random start, never on the goal: a fixed start makes a policy that has
    # memorised one path look like a policy that has learned to navigate.
    while True:
        pos = (rng.randrange(SIZE), rng.randrange(SIZE))
        if pos != GOAL:
            state["pos"] = pos
            return _obs()


def step(action):
    dr, dc = MOVES.get(int(action) if action is not None else 0, (0, 0))
    r, c = state["pos"]
    state["pos"] = (min(SIZE - 1, max(0, r + dr)), min(SIZE - 1, max(0, c + dc)))
    state["t"] += 1

    terminated = state["pos"] == GOAL
    truncated = (not terminated) and state["t"] >= MAX_STEPS
    reward = 1.0 if terminated else -0.01

    return _obs(), reward, terminated, truncated, {"t": state["t"], "pos": list(state["pos"])}


serve(reset=reset, step=step)
