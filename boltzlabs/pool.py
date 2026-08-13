"""RLPool — N live environments behind one batched step.

    from boltzlabs import RLPool

    pool = RLPool(url="https://boltzlabs.cloud", api_key=...,
                  env_dir="./my_env", n=1000, runtime="python3")
    obs = pool.reset()
    for t in range(max_steps):
        actions = policy(obs)
        obs, rewards, dones, infos = pool.step(actions)   # ONE request
        if dones.any():
            obs = pool.reset(where=dones)
    pool.close()

The whole design follows from that one comment. A trainer produces N actions at
once and cannot proceed until it has N results, so the unit of work is the
batch: one request carries every action and comes back with every result. N
requests per step would put N network round trips inside the inner loop of
training and make any latency number this platform quotes meaningless.
"""

import base64
import time
from dataclasses import dataclass

import numpy as np

from . import _pack, config
from ._http import Session
from .errors import BoltzLabsError

__all__ = ["RLPool", "Timing"]

# The public path, served by the SvelteKit origin and proxied to the control
# plane — which binds loopback and is never addressed directly. The worker's own
# routes exist so a benchmark can measure the worker with no hop in the path,
# and so the SDK can be developed against `make up` alone.
_CP_PREFIX = "/api/rl/pools"
_WORKER_PREFIX = "/worker/rl/pools"


@dataclass(frozen=True)
class Timing:
    """What the last call cost, split by where the time went.

    ``worker_ms`` is the batch measured on the worker itself. ``roundtrip_ms`` is
    this process's wall clock around the whole call, serialisation included.
    They are never added together and never conflated: the gap between them is
    the network and the control plane, and quoting a step latency measured from
    a trainer on another continent as if it were the worker's is a statement
    about the internet rather than about the platform.
    """

    worker_ms: float
    roundtrip_ms: float
    fanout_p50_ms: float
    fanout_p99_ms: float
    fanout_max_ms: float
    stragglers: int
    serialized: bool

    @property
    def overhead_ms(self):
        """Everything that was not the worker: transport, proxy, JSON."""
        return max(0.0, self.roundtrip_ms - self.worker_ms)

    @classmethod
    def from_wire(cls, timing, roundtrip_ms):
        t = timing or {}
        return cls(
            worker_ms=float(t.get("worker_ms", 0.0)),
            roundtrip_ms=float(roundtrip_ms),
            fanout_p50_ms=float(t.get("fanout_p50_ms", 0.0)),
            fanout_p99_ms=float(t.get("fanout_p99_ms", 0.0)),
            fanout_max_ms=float(t.get("fanout_max_ms", 0.0)),
            stragglers=int(t.get("stragglers", 0)),
            serialized=bool(t.get("serialized", False)),
        )

    def __str__(self):
        return (
            f"worker {self.worker_ms:.2f} ms | round trip {self.roundtrip_ms:.2f} ms "
            f"| overhead {self.overhead_ms:.2f} ms | p99 fan-out {self.fanout_p99_ms:.2f} ms"
            + (f" | {self.stragglers} stragglers" if self.stragglers else "")
        )


def _jsonable(value):
    """Coerce actions into something ``json`` will take.

    Policies emit numpy. The wire is JSON. This is the one place that conversion
    happens, so the cost of it shows up in ``roundtrip_ms`` where a caller can
    see it, rather than being hidden inside the transport.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class RLPool:
    """A pool of live environments on an BoltzLabs worker.

    Through the control plane, which is what a customer uses::

        RLPool(url="https://boltzlabs.cloud", api_key="ak_...",
               env_dir="./my_env", n=1000)

    Straight at a worker, for local development and for benchmarks that must not
    have a control-plane hop in the measured path::

        RLPool(direct="http://127.0.0.1:9877", worker_token="xxx",
               env_dir="./my_env", n=64)

    Either form reads its credentials from the environment when they are not
    passed: ``BOLTZLABS_URL`` / ``BOLTZLABS_API_KEY``, or ``BOLTZLABS_WORKER_URL`` /
    ``BOLTZLABS_WORKER_TOKEN``.
    """

    def __init__(
        self,
        env_dir=None,
        n=None,
        *,
        environment=None,
        url=None,
        api_key=None,
        direct=None,
        worker_token=None,
        runtime="python3",
        entrypoint=None,
        name=None,
        memory_mb=None,
        step_timeout_ms=None,
        serialize_measurement=False,
        measure_cpu=None,
        vendor_sdk=True,
        timeout=60.0,
        create_timeout=900.0,
    ):
        # A pool runs either an environment the platform ships or one you wrote.
        # Refused rather than resolved by precedence: the two readings of "whose
        # code ran" are equally plausible, and the wrong one is a training run
        # against an environment nobody chose.
        if bool(env_dir) == bool(environment):
            raise ValueError(
                "pass either env_dir=<your environment> or environment=<a name>, "
                "not both and not neither — see boltzlabs.rl_environments()"
            )
        if n is None:
            raise ValueError("n is required: how many environments to run at once")

        direct = direct or config.get("BOLTZLABS_WORKER_URL")
        if url and direct:
            raise ValueError("pass url= or direct=, not both")

        if direct:
            token = worker_token or config.get("BOLTZLABS_WORKER_TOKEN", "")
            # Empty is legitimate here: a worker started without RL_WORKER_TOKEN
            # accepts everything, which is what `make up` on a laptop does.
            headers = {"X-Worker-Token": token} if token else {}
            self._base, self._prefix, self.via = direct, _WORKER_PREFIX, "worker"
        else:
            # No arguments is the common case: the CLI's saved login, or the
            # environment, or the production origin. See config.py.
            url, key = config.resolve(url, api_key)
            headers = {"Authorization": f"Bearer {key}"}
            self._base, self._prefix, self.via = url, _CP_PREFIX, "platform"

        self._session = Session(self._base, headers=headers, timeout=timeout)
        self._step_timeout = timeout

        body = {"n": int(n), "serialize_measurement": bool(serialize_measurement)}

        if environment:
            # A ready-made environment: nothing to pack, and runtime/entrypoint
            # belong to the catalogue entry rather than to this call. Sending
            # them anyway would be a request the platform refuses.
            self.environment = environment
            self.runtime = None
            self.entrypoint = None
            self.code_bytes = 0
            self.skipped_symlinks = ()
            body["environment"] = environment
        else:
            self.environment = None
            self.runtime = runtime
            self.entrypoint = entrypoint or ("env.js" if runtime == "node" else "env.py")

            code, skipped_links = _pack.pack_env_dir(
                env_dir, entrypoint=self.entrypoint, vendor=vendor_sdk
            )
            self.code_bytes = len(code)
            self.skipped_symlinks = skipped_links

            body["runtime"] = runtime
            body["entrypoint"] = self.entrypoint
            body["code_tar_gz"] = base64.b64encode(code).decode("ascii")
        if name:
            body["name"] = name
        if memory_mb:
            body["memory_mb"] = int(memory_mb)
        if step_timeout_ms:
            body["step_timeout_ms"] = int(step_timeout_ms)
        if measure_cpu is not None:
            body["measure_cpu"] = int(measure_cpu)

        # Creating a thousand sandboxes legitimately takes minutes — a thousand
        # interpreter startups, staggered. This is the one call with a long
        # deadline; every call after it is on the training loop's clock.
        created, _ = self._session.call("POST", self._prefix, body, timeout=create_timeout)

        self.pool_id = created["pool_id"]
        self.n = int(created.get("n", n))
        self.ready = int(created.get("ready", self.n))
        self.spawn_ms = created.get("spawn_ms") or {}
        self.serialize_measurement = bool(serialize_measurement)

        self._obs = [None] * self.n
        self._closed = False
        self._steps = 0
        self._timing = None
        self._reset_timing = None
        self._created_at = time.time()

    # -- properties ---------------------------------------------------------

    @property
    def timing(self):
        """The last ``step``'s timing, or ``None`` before the first one."""
        return self._timing

    @property
    def reset_timing(self):
        """The last ``reset``'s timing. Kept apart from ``timing`` because a
        hard reset is a cold start and mixing the two would flatter the step
        numbers."""
        return self._reset_timing

    @property
    def steps(self):
        return self._steps

    @property
    def closed(self):
        return self._closed

    def __len__(self):
        return self.n

    def __repr__(self):
        state = "closed" if self._closed else f"{self.n} envs"
        return f"<RLPool {self.pool_id} {state} via {self.via}>"

    # -- the loop -----------------------------------------------------------

    def reset(self, seed=None, where=None, hard=False):
        """Start new episodes and return the full observation list.

        ``where`` selects a subset — a boolean mask of length n (the ``dones``
        array from ``step`` is exactly that), or a list of indices. The returned
        list is always length n regardless: the environments that were not reset
        keep the observation they last reported, so the result can go straight
        back into the policy without the caller splicing anything by hand.

        Getting that splice wrong corrupts a training run silently — the policy
        sees environment 7's observation labelled 3 — which is why it happens
        here, once, and is covered by a test.

        ``hard=True`` kills and respawns the sandboxes instead of calling the
        environment's own reset. Slower, and the only version that guarantees
        episode N+1 starts byte-identical to episode N.
        """
        self._check_open()

        indices = None if where is None else self._indices(where)
        if indices is not None and len(indices) == 0:
            # Nothing selected is not an error and must not be a request: a
            # training loop that calls reset(where=dones) unconditionally would
            # otherwise pay a round trip on every step where nothing finished.
            return list(self._obs)

        body = {"hard": bool(hard)}
        if seed is not None:
            body["seed"] = int(seed)
        if indices is not None:
            body["indices"] = indices

        started = time.perf_counter()
        res, _ = self._session.call(
            "POST", f"{self._prefix}/{self.pool_id}/reset", body, timeout=self._reset_timeout(hard)
        )
        roundtrip_ms = (time.perf_counter() - started) * 1000.0

        obs = res.get("obs") or []
        if indices is None:
            if len(obs) != self.n:
                raise BoltzLabsError(f"reset returned {len(obs)} observations, expected {self.n}")
            self._obs = list(obs)
        else:
            if len(obs) != len(indices):
                raise BoltzLabsError(
                    f"reset of {len(indices)} environments returned {len(obs)} observations"
                )
            # The worker answers in the order the indices were sent, so slot i of
            # the reply belongs to environment indices[i] — not to environment i.
            for slot, idx in enumerate(indices):
                self._obs[idx] = obs[slot]

        self._reset_timing = Timing.from_wire(res.get("timing"), roundtrip_ms)
        return list(self._obs)

    def step(self, actions):
        """One action per environment, one request, N results.

        Returns ``(obs, rewards, dones, infos)``:

        * ``obs``   — list of length n, arbitrary JSON, exactly what each
          environment returned
        * ``rewards`` — ``np.float32[n]``
        * ``dones``   — ``np.bool_[n]``
        * ``infos``   — list of length n, arbitrary JSON

        The two arrays are numpy because that is what a trainer's next line
        expects; obs and info stay as they came, because coercing arbitrary JSON
        into an array is a guess about the caller's observation space that this
        layer has no business making.

        An environment that missed its deadline on the worker comes back as
        ``done=True`` with zero reward and ``info["boltzlabs_straggler"]``, and is
        counted in ``pool.timing.stragglers`` — a degraded batch is visible in
        the data rather than inferred from a stall.
        """
        self._check_open()
        actions = self._normalise_actions(actions)

        # The clock starts before serialisation and stops after decoding: it is
        # the wall time the training loop actually pays for a step, not the part
        # of it that happens to be on a socket.
        started = time.perf_counter()
        res, _ = self._session.call(
            "POST",
            f"{self._prefix}/{self.pool_id}/step",
            {"actions": actions},
            timeout=self._step_timeout,
        )
        obs = res["obs"]
        rewards = np.asarray(res["rewards"], dtype=np.float32)
        dones = np.asarray(res["dones"], dtype=np.bool_)
        infos = res["infos"]
        roundtrip_ms = (time.perf_counter() - started) * 1000.0

        if len(obs) != self.n:
            raise BoltzLabsError(f"step returned {len(obs)} observations, expected {self.n}")

        self._obs = list(obs)
        self._steps += 1
        self._timing = Timing.from_wire(res.get("timing"), roundtrip_ms)
        return obs, rewards, dones, infos

    # -- observability ------------------------------------------------------

    def status(self):
        """Live pool status, including the memory number that decides the bill.

        ``pss_bytes_per_env`` is proportional set size: shared pages divided
        among the processes mapping them, so it is the marginal cost of the next
        environment rather than a total that counts one shared libpython N times.
        """
        self._check_open()
        res, _ = self._session.call("GET", f"{self._prefix}/{self.pool_id}")
        return res

    def envs_per_gb(self):
        """Environments per gigabyte at the current size, or ``None`` if the
        worker could not read PSS (no ``smaps_rollup`` — a non-Linux host)."""
        per = self.status().get("pss_bytes_per_env")
        if not per:
            return None
        return 1e9 / float(per)

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        """Destroy the pool. Idempotent, and safe to call from a finaliser."""
        if self._closed:
            return
        self._closed = True
        try:
            self._session.call("DELETE", f"{self._prefix}/{self.pool_id}", timeout=120.0)
        except BoltzLabsError:
            # Already gone, or the worker left the fleet. Either way the
            # environments are not running and raising here would mask whatever
            # exception was already on its way out of a `with` block.
            pass
        finally:
            self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    def __del__(self):
        # A pool that is not closed is N processes still running on a rented box.
        # An interrupted script is exactly when that matters, so the finaliser
        # tries — and swallows everything, because raising during interpreter
        # teardown only produces noise nobody can act on.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- internals ----------------------------------------------------------

    def _check_open(self):
        if self._closed:
            raise BoltzLabsError("this pool is closed")

    def _reset_timeout(self, hard):
        # A hard reset respawns sandboxes; that is a cold start, not a step, and
        # holding it to the step deadline would kill healthy pools.
        return max(self._step_timeout, 300.0) if hard else self._step_timeout

    def _indices(self, where):
        """Boolean mask or index list → the list of indices the worker wants."""
        arr = np.asarray(where)
        if arr.dtype == np.bool_:
            if arr.shape != (self.n,):
                raise ValueError(
                    f"where= is a boolean mask of shape {arr.shape}; this pool has {self.n} "
                    f"environments, so it must be ({self.n},)"
                )
            return [int(i) for i in np.flatnonzero(arr)]
        idx = [int(i) for i in np.asarray(arr).ravel()]
        for i in idx:
            if not 0 <= i < self.n:
                raise ValueError(f"index {i} out of range for a pool of {self.n}")
        return idx

    def _normalise_actions(self, actions):
        if isinstance(actions, np.ndarray):
            count = actions.shape[0] if actions.ndim else 0
        else:
            actions = list(actions)
            count = len(actions)
        if count != self.n:
            raise ValueError(
                f"expected one action per environment ({self.n}), got {count}"
            )
        return _jsonable(actions)
