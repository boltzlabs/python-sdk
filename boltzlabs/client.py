"""Sandboxes.

    from boltzlabs import Sandbox

    sb = Sandbox()                            # small / base / internet off

    print(sb.exec("pip install requests"))    # shell
    sb.terminal()                             # interactive

    sb.delete()

That is the whole thing: make a machine, use it, pause or destroy it. Active
compute bills only while it runs; `pause()` preserves the workspace and
`delete()` ends storage too. A `with` block deletes it for you, including when
something raises.

`Sandbox()` alone is enough. Every argument is a keyword with a default — name
only what you are changing:

    Sandbox(
        machine="small",       # small | medium | large
        environment="base",    # runtimes and prebuilt coding agents
        name=None,             # defaults to the id the platform assigns
        internet=False,
        idle_timeout=None,     # seconds; None leaves the platform default
        max_lifetime=None,     # seconds; None leaves the platform default
    )

Everything the `boltz` CLI does is here:

| `boltz …`             | Python                         |
| ---------------------- | ------------------------------ |
| `create`               | `Sandbox()`                    |
| `exec <id\|name> <cmd>` | `sb.exec("cmd")`              |
| `connect <id\|name>`    | `sb.terminal()`               |
| `rm <id\|name>`         | `sb.delete()`                 |
| `ls`                   | `boltzlabs.sandboxes()`           |
| `status <id\|name>`     | `boltzlabs.sandbox("scratch")`   |
| `environments`         | `boltzlabs.environments()`        |
| `machines`             | `boltzlabs.machines()`            |
| `auth status`          | `boltzlabs.me()`                  |
| `version`              | `boltzlabs.__version__`           |

`update` has no equivalent: upgrading a library is pip's job.

``Client`` is underneath all of it and you rarely need to name it — the
module-level functions use a default one, built from ``BOLTZLABS_API_KEY`` in the
environment or a ``.env``. Reach for it when you want two keys in one process,
or the less common calls: metrics, port proxying, API keys.
"""

import time
from pathlib import Path as _Path

from . import config
from ._http import Session
from .errors import BoltzLabsError, NotFoundError

__all__ = ["Client", "Sandbox", "ExecResult", "Environment", "Machine", "Language", "APIKey"]

# The platform applies these itself when a create names neither; they are stated
# here so the signature documents them rather than hiding them behind None.
DEFAULT_MACHINE = "small"
DEFAULT_ENVIRONMENT = "base"

# The environments the platform ships. Passing an unknown one is a 400 that
# names these.
ENVIRONMENTS = (
    "base",
    "python",
    "node",
    "opencode",
    "claude-code",
    "codex",
    "deepagents",
)

# The machines it offers, cheapest first.
MACHINES = ("small", "medium", "large")


class ExecResult:
    """What a command left behind.

    ``str()`` is the output and ``bool()`` is success, so both of these read the
    way you would say them::

        print(sb.exec("ls"))
        if sb.exec("test -f /app/x"):
            ...
    """

    __slots__ = (
        "stdout",
        "stderr",
        "exit_code",
        "duration_ms",
        "reason",
        "compile_ms",
        "compile_failed",
    )

    def __init__(
        self,
        stdout="",
        stderr="",
        exit_code=0,
        duration_ms=0,
        reason="",
        compile_ms=0,
        compile_failed=False,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.duration_ms = duration_ms
        self.reason = reason
        # Only ever set by execute(), and only for a compiled language.
        # ``compile_ms`` is how much of ``duration_ms`` was the compiler;
        # ``compile_failed`` means the program never ran and ``stderr`` is the
        # compiler's message rather than the program's.
        self.compile_ms = compile_ms
        self.compile_failed = compile_failed

    def __str__(self):
        return self.stdout if self.exit_code == 0 else (self.stdout + self.stderr)

    def __bool__(self):
        return self.exit_code == 0

    def __repr__(self):
        return f"<ExecResult exit={self.exit_code} {self.duration_ms}ms {self.stdout[:40]!r}>"

    def check(self):
        """Raise unless it succeeded. For a script that should stop here."""
        if self.exit_code != 0:
            what = "did not compile" if self.compile_failed else f"exited {self.exit_code}"
            raise BoltzLabsError(
                f"command {what}"
                + (f" ({self.reason})" if self.reason else "")
                + (f": {self.stderr.strip()[:500]}" if self.stderr.strip() else "")
            )
        return self

    @classmethod
    def _from_wire(cls, d):
        d = d or {}
        return cls(
            stdout=d.get("stdout") or "",
            stderr=d.get("stderr") or "",
            exit_code=int(d.get("exitCode") or 0),
            duration_ms=int(d.get("durationMs") or 0),
            reason=d.get("reason") or "",
            compile_ms=int(d.get("compileMs") or 0),
            compile_failed=bool(d.get("compileFailed")),
        )


class Environment:
    """The runtime and coding-agent images available to new sandboxes."""

    __slots__ = ("name", "default")

    def __init__(self, name, default=False):
        self.name, self.default = name, default

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<Environment {self.name}{' default' if self.default else ''}>"


class Machine:
    """A machine: small, medium, or large — and what it costs."""

    __slots__ = ("name", "vcpus", "memory_mb", "disk_gb", "rate_usd_per_hour")

    def __init__(self, name, vcpus=0, memory_mb=0, disk_gb=0, rate_usd_per_hour=0.0):
        self.name = name
        self.vcpus = vcpus
        self.memory_mb = memory_mb
        self.disk_gb = disk_gb
        self.rate_usd_per_hour = rate_usd_per_hour

    def __str__(self):
        return self.name

    def __repr__(self):
        return (
            f"<Machine {self.name} {self.vcpus}vCPU {self.memory_mb}MB "
            f"${self.rate_usd_per_hour:.2f}/hr>"
        )


class Language:
    """A language the exec plane runs. ``id`` is the ``language_id`` a
    submission sends; ``code`` (python, node, go, c, cpp) works in its place.

    ``compiled`` is the one difference you can see from out here: those runs
    build first, and code that does not compile comes back as a Compilation
    Error with the compiler's message rather than a traceback.
    """

    __slots__ = ("id", "name", "code", "extension", "compiled")

    def __init__(self, id=0, name="", code="", extension="", compiled=False):
        self.id, self.name, self.code = id, name, code
        self.extension, self.compiled = extension, compiled

    @property
    def label(self):
        return self.name

    def __str__(self):
        return self.code

    def __repr__(self):
        return f"<Language {self.id} {self.code} ({self.name})>"


class Submission:
    """One run on the exec plane, in the standard submission format.

    ``res.json`` is the response exactly as it came back; its fields are
    attributes too — ``stdout``, ``stderr``, ``compile_output``, ``message``,
    ``status`` ({"id", "description"}), ``time`` and ``wall_time`` (seconds, as
    strings), ``memory`` (KB), ``exit_code``, ``token``. ``str(res)`` is what it
    printed and ``bool(res)`` is whether it was Accepted::

        res = boltzlabs.execute("print(int(input()) * 2)", language="python",
                                stdin="21", expected_output="42", cpu_time_limit=1)
        res.status        # {"id": 3, "description": "Accepted"}
        res.time, res.memory
    """

    FIELDS = (
        "token", "stdout", "stderr", "compile_output", "message", "status",
        "time", "wall_time", "memory", "exit_code", "exit_signal", "language_id",
    )

    def __init__(self, data=None):
        self.json = dict(data or {})
        for field in self.FIELDS:
            setattr(self, field, self.json.get(field))

    @property
    def status_id(self):
        return (self.status or {}).get("id")

    @property
    def finished(self):
        """False while it is still In Queue or Processing."""
        return self.status_id not in (1, 2)

    @property
    def accepted(self):
        return self.status_id == 3

    def __bool__(self):
        return self.accepted

    def __str__(self):
        out = self.stdout or ""
        if not self.accepted:
            out += (self.compile_output or "") if self.status_id == 6 else (self.stderr or "")
        return out

    def __repr__(self):
        desc = (self.status or {}).get("description", "?")
        return f"<Submission {desc} time={self.time} memory={self.memory} {(self.stdout or '')[:40]!r}>"

    def check(self):
        """Raise unless it was Accepted. For a script that should stop here."""
        if not self.accepted:
            detail = (self.compile_output or self.stderr or self.message or "").strip()[:500]
            desc = (self.status or {}).get("description", "not finished")
            raise BoltzLabsError(f"run ended {desc}" + (f": {detail}" if detail else ""))
        return self


class APIKey:
    __slots__ = ("id", "name", "created_at", "last_used_at", "key")

    def __init__(self, id="", name="", created_at="", last_used_at="", key=""):
        self.id = id
        self.name = name
        self.created_at = created_at
        self.last_used_at = last_used_at
        # Only ever set on the response that created it — the platform stores a
        # hash and cannot show it again.
        self.key = key

    def __repr__(self):
        return f"<APIKey {self.id} {self.name!r}>"


class Sandbox:
    """A Linux workspace with an explicit create/exec/delete lifecycle.

        sandbox = Sandbox.create(environment="python")
        result = sandbox.exec("python -c 'print(1 + 1)'")
        print(result)
        sandbox.delete()

    Use ``with Sandbox.create(...) as sandbox`` for automatic cleanup,
    including when code raises. ``Sandbox(...)`` remains supported.
    Retrieve an existing workspace with ``boltzlabs.sandbox(id)``.
    """

    @classmethod
    def create(
        cls,
        *,
        machine=DEFAULT_MACHINE,
        environment=DEFAULT_ENVIRONMENT,
        name=None,
        internet=None,
        idle_timeout=None,
        max_lifetime=None,
        client=None,
        timeout=300.0,
    ):
        """Create a workspace. Call delete() when you no longer need it."""
        return cls(
            machine=machine,
            environment=environment,
            name=name,
            internet=internet,
            idle_timeout=idle_timeout,
            max_lifetime=max_lifetime,
            client=client,
            timeout=timeout,
        )

    def __init__(
        self,
        *,
        machine=DEFAULT_MACHINE,
        environment=DEFAULT_ENVIRONMENT,
        name=None,
        internet=None,
        idle_timeout=None,
        max_lifetime=None,
        client=None,
        timeout=300.0,
    ):
        self._client = client or _default_client()
        body = {"machine": machine, "environment": environment}
        if name:
            body["name"] = name
        if internet is not None:
            body["internet"] = bool(internet)
        if idle_timeout is not None:
            body["idleTimeoutSecs"] = int(idle_timeout)
        if max_lifetime is not None:
            body["maxLifetimeSecs"] = int(max_lifetime)
        # Booting a machine is not a step; give it its own deadline rather than
        # the client-wide one.
        self._fill(self._client._post("/api/sandboxes", body, timeout=timeout))

    # -- the two verbs -------------------------------------------------------

    def exec(self, command, timeout=None):
        """Run one shell command. `boltz exec <id> <cmd…>`."""
        body = {"command": command}
        if timeout:
            body["timeoutS"] = int(timeout)
        return ExecResult._from_wire(
            self._client._post(f"/api/sandboxes/{self.id}/exec", body, timeout=_wait(timeout))
        )

    def terminal(self, script=None, timeout=60.0, **kw):
        """A real shell. `boltz connect <id>`.

        With no arguments it hands over your keyboard until you exit. With a
        ``script`` it runs that instead and returns the transcript — the same
        PTY, for commands that only behave correctly with a terminal attached.
        """
        from . import _terminal

        if script is None:
            return _terminal.attach(self._client, self.id, **kw)
        return _terminal.run_script(self._client, self.id, script, timeout=timeout, **kw)

    # -- files ---------------------------------------------------------------

    def push(self, local, remote="/workspace", timeout=300.0):
        """Copy a local file or directory into the sandbox. `boltz cp <src> <id>:<dst>`.

        Uploading ``./src`` lands it as ``<remote>/src``, the way scp does.
        """
        from . import _files

        return _files.push(self._client, self.id, local, remote, timeout=timeout)

    def pull(self, remote, local=".", timeout=300.0):
        """Copy a path out of the sandbox. `boltz cp <id>:<src> <dst>`."""
        from . import _files

        return _files.pull(self._client, self.id, remote, local, timeout=timeout)

    # -- the rest ------------------------------------------------------------

    def pause(self):
        """Stop compute and preserve files. Paused storage is free for 3 days.

        Files under /workspace are archived to object storage shortly after,
        which is what lets a paused sandbox cost nothing and come back on a
        different machine. Packages installed outside /workspace do not survive
        that; your files do.
        """
        self._fill(self._client._post(f"/api/sandboxes/{self.id}/pause"))
        return self

    def resume(self, timeout=300.0):
        """Restart a paused sandbox after capacity and credit checks.

        Seconds if the sandbox is still on its machine, longer if it has to be
        rebuilt from its archived workspace — hence its own deadline rather than
        the client-wide one.
        """
        self._fill(self._client._post(f"/api/sandboxes/{self.id}/resume", timeout=timeout))
        return self

    def fork(self, name=None, timeout=600.0):
        """A new sandbox that starts as a copy of this one. `boltz fork <id>`.

        Files under /workspace are copied; running processes and packages
        installed outside /workspace are not. This sandbox keeps running, held
        still only for as long as its workspace takes to copy. The fork is a
        sandbox like any other: it counts against your concurrent limit and
        bills on its own clock until you pause or delete it.
        """
        body = {"name": name} if name else {}
        wire = self._client._post(f"/api/sandboxes/{self.id}/fork", body, timeout=timeout)
        return Sandbox._attach(wire, self._client)

    def delete(self):
        """Destroy it. `boltz rm <id>`."""
        self._client._delete(f"/api/sandboxes/{self.id}")
        return True

    def refresh(self):
        """Re-read this sandbox from the platform, in place."""
        self._fill(self._client._get(f"/api/sandboxes/{self.id}"))
        return self

    def metrics(self):
        """CPU/memory samples the platform recorded for this sandbox."""
        return self._client._get(f"/api/sandboxes/{self.id}/metrics")

    def url(self, port, path=""):
        """The public URL that reaches a port inside the sandbox.

        Returned rather than fetched: what is listening there is your service,
        and you will want your own client, headers and timeout for it.
        """
        return f"{self._client.url}/api/sandboxes/{self.id}/proxy/{int(port)}" + (
            "/" + path.lstrip("/") if path else ""
        )

    def wait_until_running(self, timeout=180.0, poll=2.0):
        """Block until it reports running.

        Not needed today — create returns a running sandbox — so this exists for
        the case where that changes, and so a sandbox that died on boot fails
        here with its real status instead of at the first exec with a 409.
        """
        deadline = time.time() + timeout
        while True:
            self.refresh()
            if self.status == "running":
                return self
            if self.status in ("failed", "deleted", "error"):
                raise BoltzLabsError(f"sandbox {self.id} is {self.status}, not running")
            if time.time() >= deadline:
                raise BoltzLabsError(f"sandbox {self.id} was still {self.status!r} after {timeout:.0f}s")
            time.sleep(poll)

    # -- plumbing ------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        try:
            self.delete()
        except NotFoundError:
            pass
        return False

    def __repr__(self):
        return f"<Sandbox {self.id} {self.status} {self.environment}/{self.machine}>"

    def _fill(self, d):
        d = d or {}
        self.id = d.get("id") or ""
        self.name = d.get("name") or ""
        self.status = d.get("status") or ""
        self.machine = d.get("machine") or ""
        self.environment = d.get("environment") or ""
        self.vcpus = int(d.get("vcpus") or 0)
        self.memory_mb = int(d.get("memoryMb") or 0)
        self.disk_gb = int(d.get("diskGb") or 0)
        self.created_at = d.get("createdAt") or ""
        self.runtime_label = d.get("runtimeLabel") or ""
        self.runtime_minutes = int(d.get("runtimeMinutes") or 0)
        self.cost_usd = float(d.get("costUsd") or 0.0)
        self.rate_usd_per_hour = float(d.get("rateUsdPerHour") or 0.0)
        # Anything the platform adds later is kept rather than dropped, so a
        # newer backend does not need a new SDK release to be usable.
        self.raw = d
        return self

    @classmethod
    def _attach(cls, d, client):
        sb = cls.__new__(cls)
        sb._client = client
        return sb._fill(d)


def _wait(timeout):
    """The client's deadline sits above the server's, so a server-side timeout
    comes back as a result rather than as a dropped connection."""
    return None if not timeout else float(timeout) + 30.0


class Client:
    """The platform, over the public origin.

    ``Client()`` takes no arguments in the normal case: the URL defaults to the
    production origin and the key comes from ``BOLTZLABS_API_KEY`` in the
    environment or in a ``.env``. Pass ``url=``/``api_key=`` to point somewhere
    else, which is what a test against a dev server does.
    """

    def __init__(self, api_key=None, url=None, timeout=60.0):
        self.url, self._api_key = config.resolve(url, api_key)
        self._session = Session(
            self.url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=timeout
        )
        self.timeout = timeout

    def __repr__(self):
        return f"<boltzlabs.Client {self.url} key={config.mask(self._api_key)}>"

    # -- what you actually call ---------------------------------------------

    def execute(
        self,
        code=None,
        *,
        language=None,
        file=None,
        stdin=None,
        expected_output=None,
        cpu_time_limit=None,
        wall_time_limit=None,
        memory_limit=None,
        supersede_key=None,
        wait=True,
    ):
        """Run one piece of code on the exec plane. `boltz run`.

        Either the code itself or a path to read it from, and always the
        language — an id (100) or a code ("python")::

            boltzlabs.execute("print(sum(range(101)))", language="python")
            boltzlabs.execute(file="main.go", language="go")

        Judging a solution: the test input, the problem's limits (seconds, and
        KB for memory — they only ever lower the platform's own) and the
        expected answer::

            res = boltzlabs.execute(file="sol.py", language="python", stdin="3\n1 2 3\n",
                                    expected_output="6", cpu_time_limit=1, memory_limit=65536)
            res.status   # {"id": 3, "description": "Accepted"}, Wrong Answer, Time Limit Exceeded, ...

        Returns a :class:`Submission`. ``wait=False`` returns at once with just
        its ``token``; :meth:`submission` fetches it later. ``supersede_key``: a
        newer run with the same key replaces this one (an editor's Run pressed
        again). Nothing is created and nothing is left over — use a
        :class:`Sandbox` when you want state to survive between commands.
        """
        if (code is None) == (file is None):
            raise ValueError("pass either code or file, not both and not neither")
        if language is None or language == "":
            raise ValueError(
                "language is required — it is never inferred. "
                "See boltzlabs.languages() for the ids and codes."
            )
        if file is not None:
            # The path is resolved here, on the caller's machine: the platform
            # never sees a path it would have to trust or resolve.
            code = _Path(file).read_text()
        body = self._submission(
            code, language, stdin=stdin, expected_output=expected_output,
            cpu_time_limit=cpu_time_limit, wall_time_limit=wall_time_limit,
            memory_limit=memory_limit, supersede_key=supersede_key,
        )
        path = "/api/execute?wait=true&fields=*" if wait else "/api/execute"
        return Submission(self._post(path, body, timeout=180))

    def execute_batch(self, submissions, *, wait=True, poll_interval=0.25, timeout=300.0):
        """Run up to 20 submissions at once — the test cases of one problem,
        say. Each is a dict of :meth:`execute`'s keywords (``code`` or
        ``source_code``, ``language``, ``stdin``, ``expected_output``, limits).
        Returns their :class:`Submission` results, in order, once all finish."""
        items = []
        for item in submissions:
            item = dict(item)
            code = item.pop("code", None) or item.pop("source_code", None)
            language = item.pop("language", None) or item.pop("language_id", None)
            items.append(self._submission(code, language, **item))
        answer = self._post("/api/execute/batch", {"submissions": items})
        bad = [(i, a) for i, a in enumerate(answer) if "token" not in a]
        if bad:
            raise BoltzLabsError(f"invalid submissions in batch: {bad}")
        tokens = [a["token"] for a in answer]
        if not wait:
            return [Submission({"token": t, "status": {"id": 1, "description": "In Queue"}}) for t in tokens]
        deadline = time.monotonic() + timeout
        while True:
            got = self._get("/api/execute/batch?fields=*&tokens=" + ",".join(tokens))["submissions"]
            results = [Submission(g) for g in got]
            if all(r.finished for r in results):
                return results
            if time.monotonic() > deadline:
                raise TimeoutError(f"batch still unfinished after {timeout}s")
            time.sleep(poll_interval)

    def submission(self, token):
        """A submission by token, as it is now."""
        return Submission(self._get(f"/api/execute/{token}?fields=*"))

    def _submission(self, code, language, **fields):
        body = {"source_code": code, "language_id": self._language_id(language)}
        body.update({k: v for k, v in fields.items() if v is not None})
        return body

    def _language_id(self, language):
        if isinstance(language, int) or str(language).isdigit():
            return int(language)
        ids = getattr(self, "_language_ids", None)
        if ids is None:
            ids = {l.code: l.id for l in self.languages()}
            self._language_ids = ids
        if language not in ids:
            raise ValueError(f"unknown language {language!r}: one of {sorted(ids)} or an id")
        return ids[language]

    def languages(self):
        """The language codes execution accepts. `boltz languages`."""
        return [
            Language(
                int(l.get("id") or 0),
                l.get("name") or "",
                l.get("code") or "",
                l.get("extension") or "",
                bool(l.get("compiled")),
            )
            for l in self._get("/api/languages") or []
        ]

    def create_sandbox(self, **kw):
        """Create a sandbox on this client. Same as ``Sandbox.create(...)``."""
        return Sandbox.create(client=self, **kw)

    def sandbox(self, id):
        """One sandbox by id, as it is now. `boltz status <id>`."""
        return Sandbox._attach(self._get(f"/api/sandboxes/{id}"), self)

    def sandboxes(self):
        """Every sandbox this key can see. `boltz ls`."""
        body = self._get("/api/sandboxes")
        return [Sandbox._attach(s, self) for s in (body or {}).get("sandboxes") or []]

    def environments(self):
        """What a sandbox can ship with. `boltz environments`."""
        body = self._get("/api/environments")
        return [
            Environment(e.get("name") or "", bool(e.get("default")))
            for e in (body or {}).get("environments") or []
        ]

    def machines(self):
        """Machines and prices. `boltz machines`."""
        body = self._get("/api/machines")
        return [
            Machine(
                m.get("name") or "",
                int(m.get("vcpus") or 0),
                int(m.get("memoryMb") or 0),
                int(m.get("diskGb") or 0),
                float(m.get("rateUsdPerHour") or 0.0),
            )
            for m in (body or {}).get("machines") or []
        ]

    def me(self):
        """Who this key belongs to. `boltz auth status`."""
        return self._get("/api/me")

    # -- api keys ------------------------------------------------------------

    def keys(self):
        body = self._get("/api/keys")
        return [
            APIKey(
                id=str(k.get("id") or ""),
                name=k.get("name") or "",
                created_at=k.get("createdAt") or k.get("created_at") or "",
                last_used_at=k.get("lastUsedAt") or k.get("last_used_at") or "",
            )
            for k in (body or {}).get("keys") or []
        ]

    def create_key(self, name):
        """Mint an API key. The secret is in the response and nowhere else."""
        body = self._post("/api/keys", {"name": name}) or {}
        return APIKey(
            id=str(body.get("id") or ""),
            name=body.get("name") or name,
            created_at=body.get("createdAt") or "",
            key=body.get("key") or body.get("apiKey") or "",
        )

    def revoke_key(self, key_id):
        self._delete(f"/api/keys/{key_id}")
        return True

    # -- rl ------------------------------------------------------------------

    def pool(self, env_dir=None, n=None, **kw):
        """An RL pool on this origin with this key — the ordinary ``RLPool``.

        Either your own environment directory, or one the platform ships::

            boltzlabs.pool("./my_env", n=64)
            boltzlabs.pool(environment="cartpole", n=64)
        """
        from .pool import RLPool

        return RLPool(env_dir, n, url=self.url, api_key=self._api_key, **kw)

    def rl_environments(self):
        """The ready-made RL environments a pool can be launched with."""
        body = self._get("/api/rl/environments")
        return (body or {}).get("environments") or []

    def pools(self):
        return (self._get("/api/rl/pools") or {}).get("pools") or []

    # -- transport -----------------------------------------------------------

    def _get(self, path, timeout=None):
        return self._session.call("GET", path, timeout=timeout)[0]

    def _post(self, path, body=None, timeout=None):
        return self._session.call("POST", path, body, timeout=timeout)[0]

    def _delete(self, path, timeout=120.0):
        return self._session.call("DELETE", path, timeout=timeout)[0]

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


# The default client, built on first use so that importing boltzlabs never needs a
# key and never opens a socket. `boltzlabs.sandboxes()` and friends go through it.
_default = None


def _default_client():
    global _default
    if _default is None:
        _default = Client()
    return _default


def use(api_key=None, url=None):
    """Point the module-level functions somewhere else.

    For a script with a key that is not in the environment, or a test against a
    dev server. Everything built afterwards uses it.
    """
    global _default
    _default = Client(api_key=api_key, url=url)
    return _default
