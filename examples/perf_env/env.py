"""An environment whose reward *is* a measurement.

Every step runs the same fixed kernel and returns `reward = 1 / elapsed`. The
work never changes, so in a quiet machine the reward is a constant. Any spread
in it is the machine, not the environment: L3 contention, memory bandwidth,
frequency scaling, another tenant's step landing on the same core.

That is the whole point. Reward is the training signal, so a reward that moves
with what the neighbours happen to be doing is noise injected directly into the
gradient — and it does not look like a bug. Nothing crashes. The run just learns
more slowly, or learns the wrong thing, and the cause is invisible from inside
the trainer.

`serialize_measurement=True` on the pool takes a single process-wide gate around
each environment's step, so exactly one piece of user code runs at a time;
`measure_cpu=<n>` additionally pins them to one core, which is coherent
precisely *because* the gate means only one is ever running. Rollout stays
concurrent — only the measurement funnels.

`bench/bench.py --only contamination` runs this pool with the gate off and then
on, and plots the reward's standard deviation against N. Two lines, one chart.

Pure Python and no allocation inside the timed region: the kernel walks a
pre-allocated buffer, so what it measures is memory and cache rather than the
allocator or the GC.
"""

import time

from boltzlabs.env import serve

# ~2 MB: comfortably larger than a typical L2 slice and small enough that a
# thousand of these do not swamp the box. The contention it exposes is real
# shared-cache contention rather than swapping.
FOOTPRINT_BYTES = 2 << 20
STRIDE = 64  # one cache line
ROUNDS = 3
EPISODE_STEPS = 50

buf = bytearray(FOOTPRINT_BYTES)
state = {"t": 0}


def _kernel():
    """Fixed work: touch every cache line of the buffer, ROUNDS times.

    Deterministic in both the work done and the value produced — the checksum is
    returned so a run can prove every environment did the same thing rather than
    assume it.
    """
    total = 0
    for _ in range(ROUNDS):
        for i in range(0, FOOTPRINT_BYTES, STRIDE):
            buf[i] = (buf[i] + 1) & 0xFF
            total += buf[i]
    return total


def reset(seed):
    state["t"] = 0
    # Zero the buffer rather than reallocate: a fresh bytearray would hand the
    # first step of each episode a cold, untouched mapping and make it slower
    # than the rest for a reason that has nothing to do with contention.
    for i in range(0, FOOTPRINT_BYTES, 4096):
        buf[i] = 0
    return {"t": 0, "footprint_bytes": FOOTPRINT_BYTES, "rounds": ROUNDS}


def step(action):
    started = time.perf_counter()
    checksum = _kernel()
    elapsed = time.perf_counter() - started

    state["t"] += 1
    # 1/elapsed, so faster is better and the reward is on a scale a plot can read.
    # elapsed is never 0 for this kernel, but the guard costs nothing and a
    # divide-by-zero here would end the episode with an error object instead.
    reward = 1.0 / elapsed if elapsed > 0 else 0.0

    return (
        {"t": state["t"], "elapsed_ms": elapsed * 1000.0},
        reward,
        state["t"] >= EPISODE_STEPS,
        {"elapsed_s": elapsed, "checksum": checksum},
    )


serve(reset=reset, step=step)
