# boltzlabs

Sandboxes, from Python.

Not on PyPI yet — install from git:

```bash
pip install git+https://github.com/boltzlabs/python-sdk.git

# or
pip3 install git+https://github.com/boltzlabs/python-sdk.git
uv add git+https://github.com/boltzlabs/python-sdk.git
```

```python
from boltzlabs import Sandbox

sandbox = Sandbox.create(environment="python")

print(sandbox.exec("python -c 'print(sum(range(101)))'"))   # → 5050
print(sandbox.exec("pip install requests"))
sandbox.terminal()                             # interactive shell

sandbox.delete()                               # stops the meter
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

`Sandbox.create()` creates one. Every argument is a keyword with a default, so you name
only what you are changing and the call says what each value means:

```python
sandbox = Sandbox.create()                          # small / base
sandbox = Sandbox.create(environment="python")      # what it ships with
sandbox = Sandbox.create(machine="medium", environment="node", name="builder")

Sandbox.create(
    machine="small",       # small | medium | large
    environment="base",    # runtime or coding-agent image
    name=None,             # defaults to the id the platform assigns
    internet=False,
    idle_timeout=None,     # seconds; None leaves the platform default
    max_lifetime=None,     # seconds; None leaves the platform default
)
```

An id is assigned by the platform, never chosen by the caller, so it is not a
constructor argument — reach an existing sandbox with `boltzlabs.sandbox(id)`.

Two verbs:

```python
sandbox.exec("ls -la")         # a shell command
sandbox.terminal()             # an interactive shell; sandbox.terminal("cmd") for a transcript
```

`exec` returns a result. `print()` it and you get the output; test it and you
get success:

```python
print(sandbox.exec("ls"))                 # prints stdout
if sandbox.exec("test -f /app/x"):        # True when the exit code was 0
    ...
sandbox.exec("make").check()              # raises if it failed
```

`.stdout`, `.stderr`, `.exit_code` and `.duration_ms` are there when you want
them. A non-zero exit is **data, not an exception** — your program failed, not
the call.

A `with` block is the same three steps with the `delete()` written for you,
including when the body raises — the case that otherwise leaves a machine
billing until someone notices:

```python
with Sandbox.create() as sandbox:
    print(sandbox.exec("python -c 'print(sum(range(101)))'"))
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

The result is the standard submission format: `str(res)` is what it printed,
`bool(res)` is whether it was Accepted, and `res.status`, `res.time` (CPU
seconds), `res.wall_time`, `res.memory` (KB), `res.stdout`, `res.stderr`,
`res.compile_output` and `res.json` (the whole response) are there when you want
them. `go`, `c` and `cpp` are built first; code that does not compile comes back
with status Compilation Error and the compiler's message in
`res.compile_output` — a result, not an exception.

Judging a solution — test input, the problem's limits (seconds, and KB for
memory) and the expected answer — and running a problem's test cases together:

```python
res = boltzlabs.execute(file="sol.py", language="python", stdin="1 2\n",
                        expected_output="3", cpu_time_limit=1, memory_limit=65536)
res.status["description"]        # Accepted, Wrong Answer, Time Limit Exceeded, ...

results = boltzlabs.execute_batch([{"code": src, "language": 113, "stdin": i, "expected_output": o}
                                   for i, o in tests])   # up to 20, in parallel
```

Batch submission is available to paid users. Each batch entry counts as one execution.
Batch waits default to 21 minutes to allow workers to start; set `timeout` in seconds to override.

The rest, when you need it:

```python
import boltzlabs

boltzlabs.me()            # who your key belongs to      (boltz auth status)
boltzlabs.sandboxes()     # everything you have running  (boltz ls)
boltzlabs.sandbox(id)     # one of them, by id           (boltz status <id>)
boltzlabs.environments()  # runtime and coding-agent images  (boltz environments)
boltzlabs.machines()      # machines and prices          (boltz machines)
boltzlabs.languages()     # language codes for execute   (boltz languages)

sandbox.url(8080)           # public URL for a port inside the sandbox
sandbox.metrics()           # recorded cpu/memory samples
sandbox.delete()            # destroy it                    (boltz rm)
```

`Client` is underneath all of it and you rarely need to name it — reach for it
to hold two keys in one process, or for API-key management (`av.keys()`,
`av.create_key(name)`, `av.revoke_key(id)`).

### Terminal

`sandbox.terminal()` is `boltz connect`: a real PTY over a WebSocket, with the
remote shell owning echo, arrow keys, tab completion and ^C. Your terminal goes
into raw mode and is restored on every exit path, including an exception — a
missed restore leaves a shell that looks broken.

`sandbox.terminal("tty; whoami")` runs a script through that same PTY and returns the
transcript, for commands that only behave correctly with a terminal attached.

The WebSocket client is written here rather than pulled in as a dependency (one
endpoint, no subprotocol), and it is tested against a real `websockets` server
so "it framed something" is not mistaken for "it interoperates".

## Examples

```
examples/sandbox_tour.py  every CLI command, done from Python
```

## RL pool startup

`RLPool(environment="cartpole", n=4)` starts creation and polls the saved pool
until it is ready. Each public HTTP request has a timeout of at most 60 seconds;
`create_timeout` bounds the complete startup (900 seconds by default). Startup
errors keep the API's status and message. If polling fails or times out, the SDK
attempts cancellation. If cleanup cannot reach the server, find and delete the
pool in your dashboard after reconnecting.

HTTP clients use `POST /api/rl/pools?wait=false` (202), then poll the returned
`Location` with the same API key once per second. Status is `creating`, `running`
or `failed`; failures include `error` and `error_status`. `DELETE` cancels pending
startup. Pending pools reserve quota and expire after 15 minutes, including
when a control-plane restart interrupts startup. The original synchronous POST
remains available for older clients; internal worker calls are unchanged.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests build `worker/rl/cmd/rlworker` and run it against the sandy stub, so they cover
the real HTTP, routing, fan-out and straggler paths on macOS without Docker. The
sandbox half — namespaces, mounts, PSS — is `worker/rl/test/smoke.py` on Linux.
