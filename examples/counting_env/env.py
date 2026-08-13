"""The smallest environment that is still an environment.

Count to 10. The observation is the counter, the action is how much to add, the
reward is the action, and the episode ends at 10. It exists so that "does my
pool work" can be answered without also debugging a policy.
"""

import random

from boltzlabs.env import serve

state = {"t": 0, "rng": random.Random()}


def reset(seed):
    state["rng"] = random.Random(seed)
    state["t"] = 0
    return {"t": 0, "noise": state["rng"].random()}


def step(action):
    state["t"] += int(action or 0)
    done = state["t"] >= 10
    return (
        {"t": state["t"], "noise": state["rng"].random()},
        float(action or 0),
        done,
        {"steps": state["t"]},
    )


serve(reset=reset, step=step)
