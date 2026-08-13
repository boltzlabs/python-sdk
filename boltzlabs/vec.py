"""Gymnasium vectorized-environment adapter over an RLPool.

The point of this file is that a pool of a thousand remote environments should
drop into a trainer that already speaks ``VectorEnv`` — Stable-Baselines3,
CleanRL, anything — without the trainer knowing that the environments are on
another machine.

Two things are worth reading before you trust it.

**Autoreset is same-step**, the classic SB3/gym semantics that Gymnasium calls
``AutoresetMode.SAME_STEP``: an environment that ends at step t is reset inside
that same call, ``obs`` carries the *new* episode's first observation, and the
terminal one is preserved in ``info["final_observation"]``. Next-step autoreset
cannot be expressed over this wire — the pool steps all N environments in one
request, so there is no way to hold one back while the others advance.

**``terminated`` and ``truncated`` come back split** whenever the environment
returned Gymnasium's 5-tuple. ``boltzlabs.env.serve`` collapses them into the
wire's single ``done`` flag but records both in ``info``, and this reads them
back out. An environment that returns a 4-tuple has no truncation information to
recover, so everything it ends is reported as a termination — which is what a
4-tuple means.
"""

import numpy as np

from .pool import RLPool

__all__ = ["BoltzLabsVecEnv"]

try:  # optional: the adapter is useful without it, and gymnasium is a big dep
    import gymnasium as gym

    _Base = gym.vector.VectorEnv
except Exception:  # noqa: BLE001
    gym = None
    _Base = object


def _stack(obs):
    """Best-effort array of the batch's observations.

    Numeric observations become one ``float32`` array, which is what a policy
    wants. Dicts and ragged shapes are left exactly as they arrived — guessing
    at a tensor layout for arbitrary JSON would produce a silently wrong batch,
    and a list the caller has to handle is better than an array they cannot
    trust.
    """
    try:
        arr = np.asarray(obs, dtype=np.float32)
    except (ValueError, TypeError):
        return obs
    if arr.dtype == np.object_:
        return obs
    return arr


class BoltzLabsVecEnv(_Base):
    """Wrap an ``RLPool`` (or create one) as a vectorized environment.

        env = BoltzLabsVecEnv(env_dir="./my_env", n=1000,
                           url="https://boltzlabs.cloud", api_key=...,
                           observation_space=Box(...), action_space=Discrete(4))

    ``observation_space`` and ``action_space`` are the caller's to declare: the
    observation crosses the wire as arbitrary JSON, so this layer genuinely does
    not know what shape it is. Trainers that only call ``reset``/``step`` do not
    need them.
    """

    metadata = {"autoreset_mode": "same-step"}

    def __init__(
        self,
        pool=None,
        *,
        observation_space=None,
        action_space=None,
        stack_obs=True,
        close_pool=True,
        **pool_kwargs,
    ):
        if pool is None:
            pool = RLPool(**pool_kwargs)
        elif pool_kwargs:
            raise TypeError("pass either an existing pool= or the arguments to build one")

        self.pool = pool
        self.num_envs = pool.n
        self._stack = stack_obs
        self._close_pool = close_pool
        self._closed = False

        if gym is not None:
            # Declared as the enum rather than a string, because a trainer that
            # branches on autoreset mode compares against the enum — and one
            # that silently falls through to the next-step branch would put the
            # terminal observation in the wrong step's batch.
            self.metadata = dict(self.metadata)
            self.metadata["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP

        if observation_space is not None:
            self.single_observation_space = observation_space
        if action_space is not None:
            self.single_action_space = action_space
        if gym is not None and observation_space is not None:
            self.observation_space = gym.vector.utils.batch_space(observation_space, pool.n)
        if gym is not None and action_space is not None:
            self.action_space = gym.vector.utils.batch_space(action_space, pool.n)

    # -- VectorEnv API ------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        hard = bool(options.get("hard")) if isinstance(options, dict) else False
        obs = self.pool.reset(seed=seed, hard=hard)
        return self._out(obs), [{} for _ in range(self.num_envs)]

    def step(self, actions):
        obs, rewards, dones, infos = self.pool.step(actions)

        infos = [dict(i) if isinstance(i, dict) else ({} if i is None else {"info": i}) for i in infos]
        truncations = np.array([bool(i.get("truncated", False)) for i in infos], dtype=np.bool_)
        # A truncation is a done that is not a termination. Deriving it this way
        # keeps the two consistent even for an environment that reported only
        # `truncated` and let `done` follow from it.
        terminations = np.asarray(dones, dtype=np.bool_) & ~truncations
        truncations &= np.asarray(dones, dtype=np.bool_)

        if dones.any():
            # Same-step autoreset: hand the terminal observation to the caller in
            # info — a value function needs it to bootstrap — and put the new
            # episode's first observation in its place.
            for i in np.flatnonzero(dones):
                infos[i]["final_observation"] = obs[i]
            obs = self.pool.reset(where=dones)

        return self._out(obs), rewards, terminations, truncations, infos

    def close(self, **_kwargs):
        if self._closed:
            return
        self._closed = True
        if self._close_pool:
            self.pool.close()

    # -- extras -------------------------------------------------------------

    @property
    def timing(self):
        """The last step's timing — worker time and round trip, kept apart."""
        return self.pool.timing

    def __len__(self):
        return self.num_envs

    def __repr__(self):
        return f"<BoltzLabsVecEnv {self.num_envs} envs on {self.pool.pool_id}>"

    def _out(self, obs):
        return _stack(obs) if self._stack else obs
