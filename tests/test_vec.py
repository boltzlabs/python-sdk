"""BoltzLabsVecEnv — the Gymnasium adapter, against a real worker.

Gymnasium itself is optional, and these run either way: the adapter subclasses
`gym.vector.VectorEnv` when it is importable and stands alone when it is not,
and the behaviour a trainer depends on is the same in both cases.
"""

import numpy as np
import pytest

from boltzlabs import RLPool
from boltzlabs.vec import BoltzLabsVecEnv

# Ends after two steps, and returns Gymnasium's 5-tuple so termination and
# truncation can be told apart on the far side of a wire that carries one flag.
FIVE_TUPLE_ENV = """
from boltzlabs.env import serve

state = {"t": 0}

def reset(seed):
    state["t"] = 0
    return [0.0, 0.0]

def step(action):
    state["t"] += 1
    terminated = action == "term" and state["t"] >= 2
    truncated = action == "trunc" and state["t"] >= 2
    return [float(state["t"]), float(len(str(action)))], 1.0, terminated, truncated, {}

serve(reset=reset, step=step)
"""


@pytest.fixture
def env_dir(tmp_path):
    (tmp_path / "env.py").write_text(FIVE_TUPLE_ENV)
    return str(tmp_path)


def make_vec(worker, env_dir, n=4, **kw):
    pool = RLPool(env_dir=env_dir, n=n, direct=worker["url"], worker_token=worker["token"])
    return BoltzLabsVecEnv(pool, **kw)


def test_reset_and_step_shapes(worker, env_dir):
    vec = make_vec(worker, env_dir, n=4)
    try:
        obs, infos = vec.reset(seed=0)
        assert isinstance(obs, np.ndarray) and obs.shape == (4, 2)
        assert obs.dtype == np.float32
        assert len(infos) == 4

        obs, rewards, terminations, truncations, infos = vec.step(["x"] * 4)
        assert obs.shape == (4, 2)
        assert rewards.shape == (4,) and rewards.dtype == np.float32
        assert terminations.dtype == np.bool_ and truncations.dtype == np.bool_
        assert not terminations.any() and not truncations.any()
    finally:
        vec.close()


def test_termination_and_truncation_are_split(worker, env_dir):
    vec = make_vec(worker, env_dir, n=4)
    try:
        vec.reset(seed=0)
        actions = ["term", "trunc", "plain", "term"]
        vec.step(actions)  # t == 1, nothing ends yet
        obs, rewards, terminations, truncations, infos = vec.step(actions)

        assert terminations.tolist() == [True, False, False, True]
        assert truncations.tolist() == [False, True, False, False]
        # Same-step autoreset: the terminal observation is preserved, and obs
        # already holds the new episode's first one.
        assert infos[0]["final_observation"] == [2.0, 4.0]
        assert obs[0][0] == 0.0
        # The environment that did not end kept running.
        assert obs[2][0] == 2.0
    finally:
        vec.close()


def test_close_closes_the_pool(worker, env_dir):
    vec = make_vec(worker, env_dir, n=2)
    vec.close()
    assert vec.pool.closed


def test_close_pool_false_leaves_it_open(worker, env_dir):
    pool = RLPool(env_dir=env_dir, n=2, direct=worker["url"], worker_token=worker["token"])
    vec = BoltzLabsVecEnv(pool, close_pool=False)
    vec.close()
    assert not pool.closed
    pool.close()


def test_non_numeric_observations_are_left_alone(worker, tmp_path):
    (tmp_path / "env.py").write_text(
        """
from boltzlabs.env import serve
def reset(seed): return {"a": 1, "b": [1, 2]}
def step(action): return {"a": 2, "b": [3, 4]}, 0.0, False, {}
serve(reset=reset, step=step)
"""
    )
    vec = make_vec(worker, str(tmp_path), n=2)
    try:
        obs, _ = vec.reset()
        # A dict observation cannot become a float32 batch, and guessing at one
        # would hand the caller an array they cannot trust.
        assert obs == [{"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]}]
    finally:
        vec.close()


def test_subclasses_gymnasium_when_available(worker, env_dir):
    gym = pytest.importorskip("gymnasium")
    vec = make_vec(worker, env_dir, n=2)
    try:
        assert isinstance(vec, gym.vector.VectorEnv)
        assert len(vec) == 2
    finally:
        vec.close()
