"""RLPool against a real worker over real HTTP."""

import os

import numpy as np
import pytest

from boltzlabs import RLPool
from boltzlabs._http import Session
from boltzlabs.errors import AuthError, BoltzLabsError, NotFoundError

# An environment that carries its own identity in every observation.
#
# `echo` is the last action it was given, and reset keeps it. That is what makes
# a mis-spliced partial reset visible: if slot 3 comes back holding echo 7, the
# observations were put back in the wrong places, and in a real run the policy
# would silently be scoring environment 7's state as environment 3's.
IDENTITY_ENV = """
from boltzlabs.env import serve

state = {"t": 0, "echo": None, "seed": None}

def reset(seed):
    state["t"] = 0
    state["seed"] = seed
    return {"t": 0, "echo": state["echo"], "seed": seed}

def step(action):
    if isinstance(action, dict) and "sleep" in action:
        import time
        time.sleep(action["sleep"])
        action = action.get("value")
    state["t"] += 1
    state["echo"] = action
    return {"t": state["t"], "echo": action}, float(state["t"]), state["t"] >= 4, {"echo": action}

serve(reset=reset, step=step)
"""


@pytest.fixture
def env_dir(tmp_path):
    (tmp_path / "env.py").write_text(IDENTITY_ENV)
    return str(tmp_path)


def make_pool(worker, env_dir, n=8, **kw):
    kw.setdefault("worker_token", worker["token"])
    kw.setdefault("mode", "boltz")
    return RLPool(env_dir=env_dir, n=n, direct=worker["url"], **kw)


def test_create_reset_step(worker, env_dir):
    with make_pool(worker, env_dir, n=8) as pool:
        assert pool.n == 8 and pool.ready == 8
        assert pool.spawn_ms.get("p50") is not None

        obs = pool.reset(seed=3)
        assert len(obs) == 8
        assert all(o["t"] == 0 and o["seed"] == 3 for o in obs)

        obs, rewards, dones, infos = pool.step(list(range(8)))
        assert [o["echo"] for o in obs] == list(range(8))
        assert rewards.dtype == np.float32 and dones.dtype == np.bool_
        assert rewards.shape == (8,) and dones.shape == (8,)
        assert not dones.any()
        assert [i["echo"] for i in infos] == list(range(8))


def test_partial_reset_splices_into_the_right_slots(worker, env_dir):
    """The failure this guards against is silent, so it is worth being explicit.

    Reset a subset. The worker answers in the order the indices were sent, so
    slot 0 of its reply belongs to environment 2 (say), not to environment 0.
    Getting that mapping wrong corrupts a training run without raising anything.
    """
    with make_pool(worker, env_dir, n=8) as pool:
        pool.reset(seed=1)
        for _ in range(3):
            obs, _, _, _ = pool.step(list(range(8)))
        assert all(o["t"] == 3 for o in obs)

        mask = np.zeros(8, dtype=bool)
        mask[[2, 5, 7]] = True
        obs = pool.reset(where=mask)

        assert len(obs) == 8
        for i in range(8):
            if mask[i]:
                assert obs[i]["t"] == 0, f"env {i} should have been reset"
            else:
                assert obs[i]["t"] == 3, f"env {i} should have been left alone"
            # The identity check: whichever slot it is in, the observation is
            # that environment's own.
            assert obs[i]["echo"] == i


def test_partial_reset_by_index_list(worker, env_dir):
    with make_pool(worker, env_dir, n=4) as pool:
        pool.reset()
        pool.step([0, 1, 2, 3])
        obs = pool.reset(where=[1, 3])
        assert [o["t"] for o in obs] == [1, 0, 1, 0]
        assert [o["echo"] for o in obs] == [0, 1, 2, 3]


def test_empty_mask_makes_no_request(worker, env_dir):
    with make_pool(worker, env_dir, n=4) as pool:
        obs = pool.reset()
        before = pool.reset_timing
        again = pool.reset(where=np.zeros(4, dtype=bool))
        assert again == obs
        assert pool.reset_timing is before  # nothing crossed the wire


def test_dones_drive_the_loop(worker, env_dir):
    with make_pool(worker, env_dir, n=4) as pool:
        pool.reset()
        for _ in range(4):
            obs, rewards, dones, _ = pool.step([1, 1, 1, 1])
        assert dones.all()  # the env ends at t == 4
        obs = pool.reset(where=dones)
        assert all(o["t"] == 0 for o in obs)


def test_numpy_actions(worker, env_dir):
    with make_pool(worker, env_dir, n=4) as pool:
        pool.reset()
        obs, _, _, _ = pool.step(np.arange(4, dtype=np.int64))
        assert [o["echo"] for o in obs] == [0, 1, 2, 3]


def test_action_count_is_checked_before_the_request(worker, env_dir):
    with make_pool(worker, env_dir, n=4) as pool:
        pool.reset()
        with pytest.raises(ValueError) as exc:
            pool.step([1, 2])
        assert "one action per environment (4)" in str(exc.value)


def test_bad_mask_shape(worker, env_dir):
    with make_pool(worker, env_dir, n=4) as pool:
        pool.reset()
        with pytest.raises(ValueError):
            pool.reset(where=np.zeros(3, dtype=bool))


def test_timing_keeps_worker_and_round_trip_apart(worker, env_dir):
    with make_pool(worker, env_dir, n=4) as pool:
        pool.reset()
        pool.step([1] * 4)
        t = pool.timing
        assert t.worker_ms > 0
        # The round trip contains the worker's own time plus everything around
        # it. If these were ever the same number, one of them would be a lie.
        assert t.roundtrip_ms >= t.worker_ms
        assert t.overhead_ms == pytest.approx(t.roundtrip_ms - t.worker_ms)
        assert t.stragglers == 0 and t.serialized is False


def test_straggler_is_reported_not_raised(worker, env_dir):
    """One wedged environment degrades a batch; it must not freeze the run."""
    with make_pool(worker, env_dir, n=4, step_timeout_ms=400) as pool:
        pool.reset()
        actions = [0, {"sleep": 5, "value": 1}, 2, 3]
        obs, rewards, dones, infos = pool.step(actions)

        assert pool.timing.stragglers == 1
        assert dones[1] and rewards[1] == 0.0
        assert infos[1]["boltzlabs_straggler"] is True
        # The other three answered normally.
        assert [obs[i]["echo"] for i in (0, 2, 3)] == [0, 2, 3]


def test_serialize_measurement_is_reported(worker, env_dir):
    with make_pool(worker, env_dir, n=4, serialize_measurement=True) as pool:
        pool.reset()
        pool.step([1] * 4)
        assert pool.timing.serialized is True


def test_status_and_close(worker, env_dir):
    pool = make_pool(worker, env_dir, n=4)
    st = pool.status()
    assert st["pool_id"] == pool.pool_id and st["alive"] == 4

    pool.close()
    assert pool.closed
    pool.close()  # idempotent
    with pytest.raises(BoltzLabsError):
        pool.step([1] * 4)


def test_pool_is_gone_after_close(worker, env_dir):
    pool = make_pool(worker, env_dir, n=2)
    pool_id = pool.pool_id
    pool.close()

    # Ask the worker directly: the pool object is closed, but the question is
    # whether the environments are actually gone from the box.
    session = Session(worker["url"], headers={"X-Worker-Token": worker["token"]})
    with pytest.raises(NotFoundError):
        session.call("GET", f"/worker/rl/pools/{pool_id}")


def test_wrong_token_is_an_auth_error(worker, env_dir):
    with pytest.raises(AuthError):
        make_pool(worker, env_dir, n=1, worker_token="nope")


def test_vendored_serve_runs_the_shipped_examples(worker, examples_dir):
    """The examples import `from boltzlabs.env import serve`, and the sandbox has
    nothing but the uploaded directory — so this passes only if vendoring works."""
    counting = os.path.join(examples_dir, "counting_env")
    with make_pool(worker, counting, n=4) as pool:
        obs = pool.reset(seed=11)
        assert all(o["t"] == 0 for o in obs)
        obs, rewards, dones, _ = pool.step([3, 3, 3, 3])
        assert all(o["t"] == 3 for o in obs)
        assert rewards.tolist() == [3.0] * 4


# --- named environments ----------------------------------------------------
# A pool runs either an environment the platform ships or one you wrote. Both,
# or neither, is refused before anything is packed or sent: the two readings of
# "whose code ran" are equally plausible, and the wrong one is a training run
# against an environment nobody chose.


def test_a_pool_needs_exactly_one_source_of_code():
    from boltzlabs import RLPool

    with pytest.raises(ValueError, match="not both and not neither"):
        RLPool("./some_env", 4, environment="cartpole")

    with pytest.raises(ValueError, match="not both and not neither"):
        RLPool(n=4)


def test_n_is_required():
    from boltzlabs import RLPool

    with pytest.raises(ValueError, match="n is required"):
        RLPool(environment="cartpole")
