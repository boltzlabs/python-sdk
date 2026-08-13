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

sb = Sandbox()                            # small / base / internet off

print(sb.run("print(sum(range(101)))"))   # code      → 5050
print(sb.exec("pip install requests"))    # shell
sb.terminal()                             # interactive shell

sb.delete()                               # stops the meter
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

## Examples

```
examples/sandbox_tour.py  every CLI command, done from Python
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests build `worker/rl/cmd/rlworker` and run it against the sandy stub, so they cover
the real HTTP, routing, fan-out and straggler paths on macOS without Docker. The
sandbox half — namespaces, mounts, PSS — is `worker/rl/test/smoke.py` on Linux.
