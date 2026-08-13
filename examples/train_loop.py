#!/usr/bin/env python3
"""The loop, end to end. A random policy on a thousand grid worlds.

    # against a worker on your laptop (`cd rlworker && make up`)
    python examples/train_loop.py --direct http://127.0.0.1:9877 --token xxx -n 64

    # against the platform
    BOLTZLABS_API_KEY=ak_... python examples/train_loop.py \
        --url https://boltzlabs.cloud -n 1000

There is no learning in here — a random policy is enough to exercise the shape a
trainer has, which is what this file is for. What it prints is the part that is
actually interesting: worker time and round trip side by side, and environments
per gigabyte.
"""

import argparse
import statistics
import time

import numpy as np

from boltzlabs import RLPool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="control plane, e.g. https://boltzlabs.cloud")
    ap.add_argument("--api-key")
    ap.add_argument("--direct", help="a worker directly, e.g. http://127.0.0.1:9877")
    ap.add_argument("--token", help="worker token, with --direct")
    ap.add_argument("-n", type=int, default=64, help="environments")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--env-dir", default=None, help="default: examples/grid_env")
    args = ap.parse_args()

    env_dir = args.env_dir
    if env_dir is None:
        import os

        env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_env")

    t0 = time.perf_counter()
    pool = RLPool(
        env_dir=env_dir,
        n=args.n,
        url=args.url,
        api_key=args.api_key,
        direct=args.direct,
        worker_token=args.token,
        name="train_loop",
    )
    create_s = time.perf_counter() - t0

    print(f"pool {pool.pool_id}: {pool.ready}/{pool.n} environments in {create_s:.1f}s")
    if pool.spawn_ms:
        print(
            f"  cold start  p50 {pool.spawn_ms.get('p50', 0):.1f} ms   "
            f"p99 {pool.spawn_ms.get('p99', 0):.1f} ms"
        )

    with pool:
        obs = pool.reset(seed=0)
        rng = np.random.default_rng(0)

        worker_ms, roundtrip_ms = [], []
        total_reward = 0.0
        episodes = 0

        loop_started = time.perf_counter()
        for _ in range(args.steps):
            # The policy. Replace this line with a network and the rest of the
            # loop is unchanged — that is the point of the shape.
            actions = rng.integers(0, 4, size=pool.n).tolist()

            obs, rewards, dones, infos = pool.step(actions)

            total_reward += float(rewards.sum())
            episodes += int(dones.sum())
            worker_ms.append(pool.timing.worker_ms)
            roundtrip_ms.append(pool.timing.roundtrip_ms)

            if dones.any():
                # One request, only for the environments that finished. The rest
                # keep running — resetting all N here would throw away every
                # in-flight episode.
                obs = pool.reset(where=dones)

        wall = time.perf_counter() - loop_started

        p = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))]
        print(f"\n{args.steps} steps x {pool.n} environments in {wall:.1f}s")
        print(f"  {args.steps * pool.n / wall:,.0f} environment-steps/sec, {episodes} episodes")
        print(f"  mean reward per step  {total_reward / (args.steps * pool.n):+.4f}")
        print(
            f"  worker      p50 {p(worker_ms, .5):6.2f} ms   p99 {p(worker_ms, .99):6.2f} ms"
        )
        print(
            f"  round trip  p50 {p(roundtrip_ms, .5):6.2f} ms   p99 {p(roundtrip_ms, .99):6.2f} ms"
            f"   (overhead p50 {p(roundtrip_ms, .5) - p(worker_ms, .5):.2f} ms)"
        )

        st = pool.status()
        per_env = st.get("pss_bytes_per_env")
        if per_env:
            print(
                f"  memory      {per_env / 1e6:.1f} MB PSS per environment"
                f"  →  {int(1e9 / per_env):,} environments per GB"
            )
        else:
            print("  memory      not reported (PSS needs Linux smaps_rollup)")

    print("\npool closed")


if __name__ == "__main__":
    main()
