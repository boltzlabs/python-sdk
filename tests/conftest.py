"""A real rlworker, on this machine, for the tests to drive.

The sandbox is Linux-only, so a macOS checkout cannot run the real thing — but
the *worker* is not Linux-only, and it is the half the SDK talks to. So these
tests build `cmd/rlworker` and start it with `SANDY_BIN` pointing at the same
stub the Go tests use (`internal/pool/testdata/sandy-stub.sh`), which reproduces
sandy's argv and its stdio passthrough and nothing else.

What that buys: every assertion here goes over real HTTP, through the real
routing, the real fan-out, the real straggler path and the real JSON on both
sides. What it does not cover is the sandbox itself — namespaces, mounts, PSS —
which is `rlworker/test/smoke.py`'s job on Linux.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RLWORKER = os.path.join(REPO, "worker", "rl")
STUB = os.path.join(RLWORKER, "internal", "pool", "testdata", "sandy-stub.sh")
TOKEN = "test-worker-token"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(url, proc, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"rlworker exited early ({proc.returncode})")
        try:
            with urllib.request.urlopen(url + "/worker/rl/health", timeout=1) as r:
                if r.status == 200:
                    return json.loads(r.read())
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    raise RuntimeError("rlworker never became healthy")


@pytest.fixture(scope="session")
def worker(tmp_path_factory):
    """Start one worker for the whole session; yield its url and token."""
    if shutil.which("go") is None:
        pytest.skip("go toolchain not available")
    if not os.path.isfile(STUB):
        pytest.skip(f"sandy stub missing at {STUB}")

    tmp = tmp_path_factory.mktemp("rlworker")
    binary = str(tmp / "rlworker")
    build = subprocess.run(
        ["go", "build", "-o", binary, "./cmd/rlworker"],
        cwd=RLWORKER,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"could not build rlworker: {build.stderr.strip()[:400]}")

    # The stub only reads `mounts` back out to find the code directory, so the
    # thinnest possible profile is enough — and keeping it thin means this test
    # is not silently depending on a real box's generated profile.
    profile = tmp / "sandy-run.json"
    profile.write_text(json.dumps({"mounts": []}))

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env.update(
        {
            "RL_BIND": f"127.0.0.1:{port}",
            "RL_ROOT": str(tmp / "root"),
            "RL_WORKER_TOKEN": TOKEN,
            "SANDY_BIN": STUB,
            "SANDY_PROFILE": str(profile),
            # The stub execs the interpreter directly, so "python3" has to be the
            # one running these tests rather than whatever is first on PATH.
            "RL_SPAWN_CONCURRENCY": "16",
        }
    )
    # The runtime name in the request is exec'd by the stub. Make sure the
    # interpreter under test is the one that gets used.
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    log = open(tmp / "rlworker.log", "w")
    proc = subprocess.Popen([binary], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_health(url, proc)
        yield {"url": url, "token": TOKEN, "log": str(tmp / "rlworker.log")}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@pytest.fixture(scope="session")
def examples_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
