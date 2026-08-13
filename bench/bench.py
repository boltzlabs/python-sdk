#!/usr/bin/env python3
"""Benchmarks for the RL plane, written to be believed.

    python bench/bench.py --direct http://127.0.0.1:9877 --token xxx
    python bench/bench.py --direct ... --token ... --url https://boltzlabs.cloud --api-key ak_...
    python bench/bench.py --direct ... --token ... --only contamination

Five measurements, in the order they matter:

1. **Cold start** — the worker's own `spawn_ms`: process spawn to the
   environment answering its first message. Measured to the answer, not to the
   exec, because a process that has started but not yet imported its libraries
   cannot serve a step.
2. **Step latency** — `worker_ms` and the client's round trip, always both, and
   through the control plane as well as straight at the worker when an API key
   is given. The gap between the two pairs is what the platform costs; quoting
   one number for a step is how latency claims become dishonest.
3. **Throughput** — environment-steps/sec and episodes/sec against N.
4. **Environments per GB** — from PSS. The headline: it is what decides an RL
   customer's bill, it improves as N grows because of page-cache sharing, and
   nobody else publishes it.
5. **Reward contamination** — the spread of a timing-derived reward against N,
   with `serialize_measurement` off and then on. Two lines, one chart.

The sweep runs until creation fails, and records where it broke and why. A curve
with a wall on it reads as credible; one that stops at a round number reads as a
stunt. Nothing here is smoothed, and the first steps of each pool are discarded
as warm-up rather than averaged in — say so in any chart that gets published.
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boltzlabs import RLPool  # noqa: E402
from boltzlabs._http import Session  # noqa: E402
from boltzlabs.errors import BoltzLabsError  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def pct(values, q):
    """Nearest-rank percentile — the same rule the worker uses.

    Nearest-rank rather than interpolated so a reported p99 is a latency that
    was actually observed, not one computed between two that were not.
    """
    if not values:
        return 0.0
    s = sorted(values)
    i = max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))
    return float(s[i])


def summarise(values):
    return {
        "n": len(values),
        "p50": round(pct(values, 0.50), 4),
        "p90": round(pct(values, 0.90), 4),
        "p99": round(pct(values, 0.99), 4),
        "max": round(max(values), 4) if values else 0.0,
        "mean": round(statistics.fmean(values), 4) if values else 0.0,
    }


def sizes_up_to(max_n, start=1):
    n = start
    while n <= max_n:
        yield n
        n *= 2


def log(msg):
    print(msg, flush=True)


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(HERE),
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# the pool under test
# ---------------------------------------------------------------------------


class Target:
    """One endpoint the benchmark can drive: a worker, or a control plane."""

    def __init__(self, label, **kw):
        self.label = label
        self.kw = kw

    def pool(self, env_dir, n, **extra):
        return RLPool(env_dir=env_dir, n=n, **self.kw, **extra)


def worker_capacity(direct, token):
    """The box's own view of itself — CPUs and memory, for the results header.

    Recorded because every number below is relative to it. "3 ms at N=128" means
    nothing without the machine it was measured on.
    """
    try:
        session = Session(direct, headers={"X-Worker-Token": token} if token else {})
        cap, _ = session.call("GET", "/worker/rl/capacity")
        session.close()
        return cap
    except BoltzLabsError:
        return None


def drive(pool, steps, warmup, action=lambda i: 1):
    """Run `steps` steps, return the per-step samples.

    The warm-up steps are dropped rather than averaged in: the first step of a
    pool pays for whatever the environment imports lazily, and folding that into
    a p50 makes a one-off cost look like a per-step one.
    """
    n = pool.n
    worker_ms, roundtrip_ms, rewards, elapsed_s = [], [], [], []
    episodes = 0

    for i in range(warmup + steps):
        actions = [action(j) for j in range(n)]
        obs, rew, dones, infos = pool.step(actions)
        if i < warmup:
            continue
        worker_ms.append(pool.timing.worker_ms)
        roundtrip_ms.append(pool.timing.roundtrip_ms)
        rewards.extend(float(r) for r in rew)
        for info in infos:
            if isinstance(info, dict) and "elapsed_s" in info:
                elapsed_s.append(float(info["elapsed_s"]))
        episodes += int(dones.sum())
        if dones.any():
            pool.reset(where=dones)

    return {
        "worker_ms": worker_ms,
        "roundtrip_ms": roundtrip_ms,
        "rewards": rewards,
        "elapsed_s": elapsed_s,
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


def bench_sweep(args, target, results):
    """Cold start, step latency, throughput and memory — one sweep, four series.

    They share a sweep because they share a pool: creating 1024 environments to
    measure cold start and then destroying them to create 1024 more for
    throughput would double the runtime and measure a different box (a warmer
    page cache the second time).
    """
    env_dir = args.env_dir or os.path.join(EXAMPLES, "counting_env")
    cold, latency, throughput, memory = [], [], [], []
    wall = None

    for n in sizes_up_to(args.max_n):
        log(f"\n── N = {n} " + "─" * 40)
        created_at = time.perf_counter()
        try:
            pool = target.pool(env_dir, n, create_timeout=args.create_timeout)
        except (BoltzLabsError, OSError) as exc:
            # This is the interesting part of the curve, not a failure of the
            # benchmark: record the size that did not fit and why.
            log(f"   create failed at N={n}: {exc}")
            wall = {"n": n, "error": str(exc)[:600]}
            break
        create_s = time.perf_counter() - created_at

        try:
            spawn = pool.spawn_ms or {}
            cold.append(
                {
                    "n": n,
                    "create_wall_s": round(create_s, 3),
                    "spawn_p50_ms": spawn.get("p50"),
                    "spawn_p99_ms": spawn.get("p99"),
                    "spawn_max_ms": spawn.get("max"),
                    "ready": pool.ready,
                }
            )
            log(
                f"   cold start   p50 {spawn.get('p50', 0):8.1f} ms   "
                f"p99 {spawn.get('p99', 0):8.1f} ms   (wall {create_s:.1f}s)"
            )

            pool.reset(seed=0)
            started = time.perf_counter()
            s = drive(pool, args.steps, args.warmup)
            wall_s = time.perf_counter() - started

            latency.append(
                {
                    "n": n,
                    "via": target.label,
                    "worker_ms": summarise(s["worker_ms"]),
                    "roundtrip_ms": summarise(s["roundtrip_ms"]),
                    "overhead_p50_ms": round(
                        pct(s["roundtrip_ms"], 0.5) - pct(s["worker_ms"], 0.5), 4
                    ),
                }
            )
            log(
                f"   step         worker p50 {pct(s['worker_ms'], .5):7.2f} ms   "
                f"round trip p50 {pct(s['roundtrip_ms'], .5):7.2f} ms   "
                f"(overhead {pct(s['roundtrip_ms'], .5) - pct(s['worker_ms'], .5):.2f} ms)"
            )

            measured_steps = args.steps
            throughput.append(
                {
                    "n": n,
                    "steps_per_s": round(measured_steps / wall_s, 2),
                    "env_steps_per_s": round(measured_steps * n / wall_s, 1),
                    "episodes_per_s": round(s["episodes"] / wall_s, 2),
                }
            )
            log(
                f"   throughput   {measured_steps / wall_s:8.1f} steps/s   "
                f"{measured_steps * n / wall_s:12,.0f} env-steps/s   "
                f"{s['episodes'] / wall_s:8.1f} episodes/s"
            )

            st = pool.status()
            per_env = st.get("pss_bytes_per_env") or 0
            memory.append(
                {
                    "n": n,
                    "pss_bytes_per_env": per_env,
                    "pss_bytes_total": st.get("pss_bytes_total"),
                    "rss_bytes_total": st.get("rss_bytes_total"),
                    "processes": st.get("processes"),
                    "envs_per_gb": round(1e9 / per_env, 1) if per_env else None,
                }
            )
            if per_env:
                log(
                    f"   memory       {per_env / 1e6:8.2f} MB PSS/env   "
                    f"{int(1e9 / per_env):,} environments per GB   "
                    f"(RSS total {(st.get('rss_bytes_total') or 0) / 1e6:,.0f} MB)"
                )
            else:
                log("   memory       not reported (PSS needs Linux smaps_rollup)")
        finally:
            pool.close()

    results["cold_start"] = cold
    results.setdefault("latency", []).extend(latency)
    results["throughput"] = throughput
    results["memory"] = memory
    if wall:
        results["wall"] = wall


def bench_latency_via(args, target, results):
    """The second arm of the latency measurement: the same worker, one hop back.

    Run at a single size, because the question is not how the control plane
    scales but what it adds — and the answer only means anything if the pool is
    on the same box the direct arm used.
    """
    env_dir = args.env_dir or os.path.join(EXAMPLES, "counting_env")
    n = min(args.cp_n, args.max_n)
    log(f"\n── control plane, N = {n} " + "─" * 26)
    try:
        pool = target.pool(env_dir, n, create_timeout=args.create_timeout)
    except (BoltzLabsError, OSError) as exc:
        log(f"   skipped: {exc}")
        results.setdefault("skipped", []).append({"control_plane_latency": str(exc)[:400]})
        return
    try:
        pool.reset(seed=0)
        s = drive(pool, args.steps, args.warmup)
        results.setdefault("latency", []).append(
            {
                "n": n,
                "via": target.label,
                "worker_ms": summarise(s["worker_ms"]),
                "roundtrip_ms": summarise(s["roundtrip_ms"]),
                "overhead_p50_ms": round(
                    pct(s["roundtrip_ms"], 0.5) - pct(s["worker_ms"], 0.5), 4
                ),
            }
        )
        log(
            f"   step         worker p50 {pct(s['worker_ms'], .5):7.2f} ms   "
            f"round trip p50 {pct(s['roundtrip_ms'], .5):7.2f} ms   "
            f"(overhead {pct(s['roundtrip_ms'], .5) - pct(s['worker_ms'], .5):.2f} ms)"
        )
    finally:
        pool.close()


def bench_contamination(args, target, results):
    """The slide: what concurrency does to a reward that is a measurement.

    `perf_env` runs a fixed kernel and returns 1/elapsed, so on a quiet machine
    the reward is a constant and everything else is the machine. The coefficient
    of variation is reported alongside the raw standard deviation because the
    mean itself moves with N — comparing bare standard deviations across sizes
    would flatter whichever run happened to be slower.
    """
    env_dir = os.path.join(EXAMPLES, "perf_env")
    out = []

    for serialize in (False, True):
        for n in sizes_up_to(args.contamination_max_n):
            label = "serialized" if serialize else "concurrent"
            try:
                pool = target.pool(
                    env_dir,
                    n,
                    serialize_measurement=serialize,
                    measure_cpu=args.measure_cpu if serialize else None,
                    create_timeout=args.create_timeout,
                    # With the gate held, a batch is N kernels back to back. The
                    # per-environment deadline on the worker already accounts for
                    # the wait; this is only the client's patience.
                    timeout=max(120.0, n * 5.0),
                )
            except (BoltzLabsError, OSError) as exc:
                log(f"   {label} N={n}: create failed: {exc}")
                out.append({"n": n, "mode": label, "error": str(exc)[:400]})
                break

            try:
                pool.reset(seed=0)
                s = drive(pool, args.contamination_steps, args.warmup)
                rewards = s["rewards"]
                mean = statistics.fmean(rewards) if rewards else 0.0
                std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
                row = {
                    "n": n,
                    "mode": label,
                    "samples": len(rewards),
                    "reward_mean": round(mean, 3),
                    "reward_std": round(std, 3),
                    "reward_cv_pct": round(100 * std / mean, 3) if mean else None,
                    "kernel_ms_mean": round(1000 * statistics.fmean(s["elapsed_s"]), 3)
                    if s["elapsed_s"]
                    else None,
                    "step_ms_p50": round(pct(s["worker_ms"], 0.5), 3),
                }
                out.append(row)
                log(
                    f"   {label:<11} N={n:<5} reward {mean:9.1f} ± {std:8.1f}"
                    f"   CV {row['reward_cv_pct']:6.2f}%   kernel {row['kernel_ms_mean']} ms"
                )
            finally:
                pool.close()

    results["contamination"] = out


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def write_json(results, outdir, stamp):
    path = os.path.join(outdir, f"bench-{stamp}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    latest = os.path.join(outdir, "latest.json")
    with open(latest, "w") as fh:
        json.dump(results, fh, indent=2)
    return path


def write_charts(results, outdir, stamp):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("\nmatplotlib not installed — JSON only. `pip install boltzlabs[bench]` for charts.")
        return []

    written = []

    def save(fig, name):
        path = os.path.join(outdir, f"{name}-{stamp}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    cold = results.get("cold_start") or []
    if cold:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ns = [r["n"] for r in cold]
        ax.plot(ns, [r["spawn_p50_ms"] for r in cold], "o-", label="p50")
        ax.plot(ns, [r["spawn_p99_ms"] for r in cold], "s--", label="p99")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("environments in the pool")
        ax.set_ylabel("cold start (ms)")
        ax.set_title("Cold start: spawn to first answer")
        ax.grid(alpha=0.3)
        ax.legend()
        save(fig, "cold-start")

    lat = results.get("latency") or []
    if lat:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for via in sorted({r["via"] for r in lat}):
            rows = [r for r in lat if r["via"] == via]
            rows.sort(key=lambda r: r["n"])
            ns = [r["n"] for r in rows]
            ax.plot(ns, [r["worker_ms"]["p50"] for r in rows], "o-", label=f"{via}: worker p50")
            ax.plot(
                ns,
                [r["roundtrip_ms"]["p50"] for r in rows],
                "s--",
                label=f"{via}: round trip p50",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("environments in the pool")
        ax.set_ylabel("step latency (ms)")
        ax.set_title("Step latency: measured on the worker, and paid by the client")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, "step-latency")

    thr = results.get("throughput") or []
    if thr:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ns = [r["n"] for r in thr]
        ax.plot(ns, [r["env_steps_per_s"] for r in thr], "o-", label="environment-steps/sec")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("environments in the pool")
        ax.set_ylabel("environment-steps / sec")
        ax.set_title("Throughput")
        ax.grid(alpha=0.3, which="both")
        ax2 = ax.twinx()
        ax2.plot(ns, [r["episodes_per_s"] for r in thr], "^--", color="tab:orange",
                 label="episodes/sec")
        ax2.set_ylabel("episodes / sec")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
        save(fig, "throughput")

    mem = [r for r in (results.get("memory") or []) if r.get("envs_per_gb")]
    if mem:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ns = [r["n"] for r in mem]
        ax.plot(ns, [r["envs_per_gb"] for r in mem], "o-", color="tab:green")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("environments in the pool")
        ax.set_ylabel("environments per GB")
        # The rising line is the whole argument: the marginal environment gets
        # cheaper as N grows, because the interpreter's pages are shared.
        ax.set_title("Environments per GB (PSS) — the marginal cost of the next one")
        ax.grid(alpha=0.3)
        save(fig, "envs-per-gb")

    cont = [r for r in (results.get("contamination") or []) if r.get("reward_cv_pct") is not None]
    if cont:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for mode, style in (("concurrent", "o-"), ("serialized", "s--")):
            rows = sorted([r for r in cont if r["mode"] == mode], key=lambda r: r["n"])
            if rows:
                ax.plot(
                    [r["n"] for r in rows],
                    [r["reward_cv_pct"] for r in rows],
                    style,
                    label=f"{mode}"
                    + (" (serialize_measurement=True)" if mode == "serialized" else ""),
                )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("environments in the pool")
        ax.set_ylabel("reward spread, CV (%)")
        ax.set_title("Reward contamination: the same fixed kernel, timed under load")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, "contamination")

    return written


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direct", help="worker URL, e.g. http://127.0.0.1:9877")
    ap.add_argument("--token", help="worker token")
    ap.add_argument("--url", help="control plane URL — enables the second latency arm")
    ap.add_argument("--api-key")
    ap.add_argument("--env-dir", help="default: examples/counting_env")
    ap.add_argument("--max-n", type=int, default=1024, help="ceiling for the doubling sweep")
    ap.add_argument("--cp-n", type=int, default=64, help="pool size for the control-plane arm")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--contamination-steps", type=int, default=10)
    ap.add_argument("--contamination-max-n", type=int, default=64)
    ap.add_argument(
        "--measure-cpu",
        type=int,
        default=None,
        help="pin serialized runs to this CPU (only meaningful with the gate on)",
    )
    ap.add_argument("--create-timeout", type=float, default=1800.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument(
        "--only",
        default="sweep,control_plane,contamination",
        help="comma-separated: sweep, control_plane, contamination",
    )
    args = ap.parse_args()

    if not args.direct and not args.url:
        ap.error("pass --direct (a worker) and optionally --url (the control plane)")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    direct = (
        Target("worker", direct=args.direct, worker_token=args.token) if args.direct else None
    )
    control = Target("control-plane", url=args.url, api_key=args.api_key) if args.url else None

    results = {
        "meta": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "client": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
            "worker": worker_capacity(args.direct, args.token) if args.direct else None,
            "args": {
                k: v for k, v in vars(args).items() if k not in ("token", "api_key")
            },
            # Stated in the artefact rather than only in the README, because the
            # JSON is what gets pasted into a slide.
            "notes": [
                "worker_ms is measured on the worker and excludes the network;"
                " roundtrip_ms is the client's wall clock. They are never combined.",
                f"the first {args.warmup} steps of every pool are discarded as warm-up",
                "PSS is proportional set size: shared pages divided among the"
                " processes mapping them, so per-env is the marginal cost of the next one",
            ],
        }
    }

    started = time.perf_counter()
    try:
        if direct and "sweep" in only:
            bench_sweep(args, direct, results)
        if control and "control_plane" in only:
            bench_latency_via(args, control, results)
        if "contamination" in only:
            target = direct or control
            if target:
                log("\n══ reward contamination " + "═" * 30)
                bench_contamination(args, target, results)
    finally:
        results["meta"]["elapsed_s"] = round(time.perf_counter() - started, 1)
        path = write_json(results, args.out, stamp)
        charts = write_charts(results, args.out, stamp)
        log(f"\nwrote {path}")
        for c in charts:
            log(f"      {c}")
        if "wall" in results:
            log(
                f"\nbreaking point: N={results['wall']['n']} could not be created\n"
                f"  {results['wall']['error'][:300]}"
            )


if __name__ == "__main__":
    main()
