# boltzlabs

Sandboxes and RL environment pools, from Python.

```bash
pip install boltzlabs
```

Two things live here, over one origin and one API key.

```python
from boltzlabs import Sandbox

sb = Sandbox()                            # small / base / internet off

print(sb.run("print(sum(range(101)))"))   # code      → 5050
print(sb.exec("pip install requests"))    # shell
sb.terminal()                             # interactive shell

sb.delete()                               # stops the meter
```

```python
from boltzlabs import RLPool

with RLPool("./my_env", 1000) as pool:
    obs = pool.reset()
    obs, rewards, dones, infos = pool.step(actions)   # one request, 1000 envs
```

## The key

```
# .env
BOLTZLABS_API_KEY=ak_your_key_here
```

That is all the setup there is. The key is read from the environment, or from a
`.env` found by searching upwards from wherever you run your script. Real
environment variables win over `.env`, so `BOLTZLABS_API_KEY=... python train.py`
is a working one-off; passing `api_key=` wins over both.

The origin defaults to `https://boltzlabs.cloud`. Set `BOLTZLABS_API_URL` to
point elsewhere, or call `boltzlabs.use(api_key=..., url=...)` once at startup.

## Sandboxes

`Sandbox()` creates one. Every argument is a keyword with a default, so you name
only what you are changing and the call says what each value means:

```python
sb = Sandbox()                          # small / base
sb = Sandbox(environment="python")      # what it ships with
sb = Sandbox(machine="medium", environment="pytorch", name="trainer")

Sandbox(
    machine="small",       # nano | small | medium | large
    environment="base",    # runtime or coding-agent image
    name=None,             # defaults to the id the platform assigns
    internet=False,
    idle_timeout=None,     # seconds; None leaves the platform default
    max_lifetime=None,     # seconds; None leaves the platform default
)
```

An id is assigned by the platform, never chosen by the caller, so it is not a
constructor argument — reach an existing sandbox with `boltzlabs.sandbox(id)`.

Three verbs:

```python
sb.run("print(1)")        # a snippet — the language follows the environment
sb.exec("ls -la")         # a shell command
sb.terminal()             # an interactive shell; sb.terminal("cmd") for a transcript
```

Both `run` and `exec` return the same result. `print()` it and you get the
output; test it and you get success:

```python
print(sb.exec("ls"))                 # prints stdout
if sb.exec("test -f /app/x"):        # True when the exit code was 0
    ...
sb.exec("make").check()              # raises if it failed
```

`.stdout`, `.stderr`, `.exit_code` and `.duration_ms` are there when you want
them. A non-zero exit is **data, not an exception** — your program failed, not
the call.

A `with` block is the same three steps with the `delete()` written for you,
including when the body raises — the case that otherwise leaves a machine
billing until someone notices:

```python
with Sandbox() as sb:
    print(sb.run("print(sum(range(101)))"))
# destroyed here, however the block ended
```

Execution — one run, no sandbox at all. The language is always named; it is
never inferred from an extension or from the code:

```python
boltzlabs.execute("print(sum(range(101)))", language="python")   # 5050
boltzlabs.execute(file="train.py", language="python")
boltzlabs.execute("console.log(1)", language="node")
boltzlabs.execute(file="main.go", language="go")   # compiled, then run

boltzlabs.languages()     # python, node, go, c, cpp — from the platform
```

`go`, `c` and `cpp` are built before they run. Same call, and the same result
object; what changes is that `res.compile_ms` says how much of the time was the
compiler, and code that does not compile comes back with `res.compile_failed`
set and the compiler's message in `res.stderr` — a result, not an exception,
because the call worked and your code was rejected.

The rest, when you need it:

```python
import boltzlabs

boltzlabs.me()            # who your key belongs to      (bzlabs auth status)
boltzlabs.sandboxes()     # everything you have running  (bzlabs ls)
boltzlabs.sandbox(id)     # one of them, by id           (bzlabs status <id>)
boltzlabs.environments()  # runtime and coding-agent images  (bzlabs environments)
boltzlabs.machines()      # machines and prices          (bzlabs machines)
boltzlabs.languages()     # language codes for execute   (bzlabs languages)

sb.url(8080)           # public URL for a port inside the sandbox
sb.metrics()           # recorded cpu/memory samples
sb.delete()            # destroy it                    (bzlabs rm)
```

`Client` is underneath all of it and you rarely need to name it — reach for it
to hold two keys in one process, or for API-key management (`av.keys()`,
`av.create_key(name)`, `av.revoke_key(id)`).

### Terminal

`sb.terminal()` is `bzlabs connect`: a real PTY over a WebSocket, with the
remote shell owning echo, arrow keys, tab completion and ^C. Your terminal goes
into raw mode and is restored on every exit path, including an exception — a
missed restore leaves a shell that looks broken.

`sb.terminal("tty; whoami")` runs a script through that same PTY and returns the
transcript, for commands that only behave correctly with a terminal attached.

The WebSocket client is written here rather than pulled in as a dependency (one
endpoint, no subprotocol), and it is tested against a real `websockets` server
so "it framed something" is not mistaken for "it interoperates".

## RL pools

`step` is one HTTP request carrying all N actions and returning all N results.
That is the whole design. A trainer produces N actions at once and cannot
proceed until it has N results, so the unit of work is the batch — N requests
per step would put N network round trips inside the inner loop of training, and
every latency number on this platform would be a measurement of that instead.

The environments stay **warm and resident** between steps. Their state lives in
their own process variables; nothing is reloaded, re-imported, or re-serialised
per step. Code and interpreters are shared read-only across the pool (page cache
/ PSS), so memory stays low as N grows — there is **no snapshot store** and no
per-step container boot. Soft `reset()` is in-process; `reset(hard=True)` kills
and respawns for a clean start when you need it.

### The environment

Your side is one file. It reads a JSON line, acts, writes a JSON line.

```python
# my_env/env.py
from boltzlabs.env import serve

state = {"t": 0}

def reset(seed):
    state["t"] = 0
    return {"t": 0}

def step(action):
    state["t"] += 1
    return {"t": state["t"]}, float(action), state["t"] >= 100, {}

serve(reset=reset, step=step)
```

`serve` exists to own the three things that silently break this channel:

- **Buffering.** Without a flush, the interpreter holds the reply in its stdout
  buffer and a 0.2 ms step becomes seconds.
- **Stray output.** The protocol *is* stdout. One `print("debug")` — or a C
  library's `printf`, or a subprocess inheriting fd 1 — desynchronises it, and
  every reply after that is read as an answer to the previous message. `serve`
  takes a private duplicate of fd 1 and points the public one at stderr, so
  printing keeps working and lands in the environment's log.
- **Exceptions.** A traceback would kill the environment and quietly shrink the
  pool. Instead the step is answered with `done=True` and the error in `info`.

`boltzlabs/env.py` is vendored into the upload automatically — the sandbox mounts
your directory and nothing else, so the module has to travel with it. You do not
have to copy anything.

`step` may return any of:

```python
(obs, reward, done, info)                          # classic
(obs, reward, terminated, truncated, info)         # Gymnasium — both flags kept in info
(obs, reward, done)
{"obs": ..., "reward": ..., "done": ..., "info": ...}
```

### What `step` returns

| | |
| --- | --- |
| `obs` | list of length n — arbitrary JSON, exactly what each environment returned |
| `rewards` | `np.float32[n]` |
| `dones` | `np.bool_[n]` |
| `infos` | list of length n — arbitrary JSON |

`rewards` and `dones` are numpy because that is what the next line of a trainer
expects. `obs` and `infos` are left alone: coercing arbitrary JSON into an array
would be a guess about your observation space that this layer has no business
making. `BoltzLabsVecEnv` stacks them when they are numeric.

### Partial resets

```python
obs = pool.reset(where=dones)   # only the environments that finished
```

`where` takes the boolean mask `step` handed you, or a list of indices. The
returned list is always length n: environments that were not reset keep the
observation they last reported, so it goes straight back into the policy. The
splice happens once, here, and is covered by a test — getting it wrong labels
environment 7's observation as environment 3's, and nothing raises.

`reset(hard=True)` kills and respawns the sandboxes instead of calling the
environment's own reset. That is a cold start for those envs — slower, and the
only version that guarantees episode N+1 starts byte-identical to episode N.
It is not a memory snapshot restore.

### Timing, kept honest

```python
pool.timing.worker_ms      # the batch, measured on the worker
pool.timing.roundtrip_ms   # this process's wall clock around the whole call
pool.timing.overhead_ms    # the difference: network, proxy, JSON
pool.timing.stragglers     # environments that missed their deadline
```

These are never added together and never conflated. A step latency measured from
a trainer on another continent is a statement about the internet, not about the
worker, and the SDK reports both numbers rather than choosing the flattering one.

An environment that blows its per-step deadline on the worker comes back as
`done=True` with zero reward and `info["boltzlabs_straggler"]`, and is counted in
`timing.stragglers`. A degraded batch is visible in your data rather than
inferred from a stall.

### Gymnasium

```python
from boltzlabs import BoltzLabsVecEnv

env = BoltzLabsVecEnv(env_dir="./my_env", n=1000, url=..., api_key=...,
                   observation_space=Box(...), action_space=Discrete(4))
obs, infos = env.reset(seed=0)
obs, rewards, terminations, truncations, infos = env.step(actions)
```

Same-step autoreset (`AutoresetMode.SAME_STEP`): an environment that ends is
reset inside the same call and its terminal observation is preserved in
`info["final_observation"]`. Next-step autoreset cannot be expressed over this
wire — the pool steps all N environments in one request, so there is no way to
hold one back while the others advance.

Install with `pip install boltzlabs[gym]`.

### Talking to a worker directly

```python
RLPool(direct="http://127.0.0.1:9878", worker_token="xxx",
       env_dir="./my_env", n=64)
```

This bypasses the control plane. It is what local development uses (`cd worker/rl
&& make up`), and what the benchmark uses when it needs to measure the worker
with no extra hop in the path.

`BOLTZLABS_WORKER_URL` and `BOLTZLABS_WORKER_TOKEN` work here too, from the
environment or from `.env`, same as the platform pair.

### Cleaning up

```python
with RLPool(...) as pool:
    ...
```

A pool that is not closed is N processes still running on a rented box. `close()`
is idempotent, runs on `__exit__`, and is attempted from `__del__` so an
interrupted script does not leave a thousand environments behind.

### Memory

```python
pool.status()["pss_bytes_per_env"]
pool.envs_per_gb()
```

PSS, not RSS: proportional set size divides each shared page among the processes
mapping it, so this is the marginal cost of the *next* environment rather than a
number that counts one shared libpython N times. It is the metric that decides an
RL bill, and it falls as N rises — that is page-cache sharing, measured.

(Linux only: it comes from `/proc/<pid>/smaps_rollup`.)

## Examples

```
examples/counting_env/    the smallest thing that is still an environment
examples/grid_env/        5x5 grid world, returns the Gymnasium 5-tuple
examples/perf_env/        reward *is* a timing measurement — see below
examples/train_loop.py    the whole RL loop, with the numbers printed
examples/sandbox_tour.py  every CLI command, done from Python
```

```bash
python examples/train_loop.py --direct http://127.0.0.1:9878 --token xxx -n 64
```

## `perf_env` and reward contamination

Concurrency is what makes rollout fast. It is also what makes any timing-derived
reward a function of what the neighbours were doing — L3 contention, memory
bandwidth, frequency scaling. The training loop does not crash; it quietly learns
from noise.

`perf_env` runs a fixed kernel and returns `reward = 1/elapsed`, so the reward is
constant on a quiet machine and its spread is the contamination. Creating the
pool with `serialize_measurement=True` takes a process-wide gate around each
step, so exactly one environment's code runs at a time; `measure_cpu=<n>` also
pins them to one core, which is coherent precisely *because* of the gate.
Rollout stays concurrent — only the measurement funnels.

`bench/bench.py --only contamination` plots the reward's standard deviation
against N with the gate off and then on.

## Benchmarks

```bash
python bench/bench.py --direct http://127.0.0.1:9878 --token xxx
```

Writes JSON and PNG to `bench/results/`: cold start, step latency (worker vs
round trip, direct and through the control plane), throughput, environments per
GB, and reward contamination. It runs N up until creation fails and records where
it broke, because a curve with a wall on it is credible and one that stops at a
round number is not.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests build `worker/rl/cmd/rlworker` and run it against the sandy stub, so they cover
the real HTTP, routing, fan-out and straggler paths on macOS without Docker. The
sandbox half — namespaces, mounts, PSS — is `worker/rl/test/smoke.py` on Linux.
