"""Client parity with the CLI, against a stand-in platform.

The fake answers the same routes and the same JSON shapes as the control plane
(camelCase and all), so what is under test is the SDK's mapping onto them: the
paths it calls, the bodies it sends, the Bearer header it presents, and the way
it turns an error status into an exception a caller can branch on.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import boltzlabs
from boltzlabs import Client, Sandbox
from boltzlabs.errors import AuthError, BoltzLabsError, NotFoundError

SANDBOX = {
    "id": "sb-123",
    "name": "mybox",
    "status": "running",
    "machine": "small",
    "environment": "python",
    "vcpus": 1,
    "memoryMb": 1024,
    "diskGb": 5,
    "createdAt": "2026-08-07T00:00:00Z",
    "runtimeLabel": "12m",
    "runtimeMinutes": 12,
    "costUsd": 0.02,
    "rateUsdPerHour": 0.1,
}


class FakePlatform(BaseHTTPRequestHandler):
    calls = []  # (method, path, body) for every request that arrived

    def log_message(self, *_a):  # keep pytest output clean
        pass

    # -- plumbing -----------------------------------------------------------

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return None
        return json.loads(self.rfile.read(n))

    def _send(self, status, payload=None):
        raw = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if raw:
            self.wfile.write(raw)

    def _route(self, method):
        body = self._body()
        type(self).calls.append((method, self.path, body))

        # Every route is owner-scoped behind a Bearer key; a client that forgets
        # the header should fail here rather than somewhere confusing later.
        if self.headers.get("Authorization") != "Bearer testkey":
            return self._send(401, {"error": "unauthorized"})

        p, m = self.path, method
        if p == "/api/me" and m == "GET":
            return self._send(200, {"id": "u1", "email": "a@b.c", "name": "A"})
        if p == "/api/environments" and m == "GET":
            return self._send(200, {"environments": [{"name": "base"}, {"name": "python", "default": True}]})
        if p == "/api/machines" and m == "GET":
            return self._send(200, {"machines": [
                {"name": "small", "vcpus": 1, "memoryMb": 1024, "diskGb": 5, "rateUsdPerHour": 0.1}
            ]})
        if p == "/api/languages" and m == "GET":
            return self._send(200, [
                {"id": 102, "name": "JavaScript (Node.js 26.10.0)", "code": "node", "extension": ".js", "compiled": False},
                {"id": 103, "name": "C (GCC 15.2)", "code": "c", "extension": ".c", "compiled": True},
                {"id": 113, "name": "Python (3.14)", "code": "python", "extension": ".py", "compiled": False},
            ])
        if p.startswith("/api/execute/batch") and m == "POST":
            return self._send(201, [{"token": f"t{i}"} for i, _ in enumerate(body["submissions"])])
        if p.startswith("/api/execute/batch") and m == "GET":
            tokens = p.split("tokens=")[1].split("&")[0].split(",")
            return self._send(200, {"submissions": [
                {"token": t, "stdout": f"out {t}\n", "status": {"id": 3, "description": "Accepted"}} for t in tokens
            ]})
        if p.startswith("/api/execute/") and m == "GET":
            return self._send(200, {"token": p.split("/")[3].split("?")[0], "status": {"id": 3, "description": "Accepted"}})
        if p.startswith("/api/execute") and m == "POST":
            # A compiled language that does not compile: the program never ran,
            # so compile_output is the compiler's and there is no stdout.
            if body.get("language_id") == 103 and "syntax error" in body["source_code"]:
                return self._send(201, {
                    "stdout": None, "stderr": None, "compile_output": "main.c:1: expected ';'",
                    "status": {"id": 6, "description": "Compilation Error"}, "token": "tok-c",
                })
            if "expected_output" in body:
                ok = body["expected_output"] == "6"
                return self._send(201, {
                    "stdout": "6\n", "stderr": None, "time": "0.012", "memory": 9216, "token": "tok-j",
                    "status": {"id": 3, "description": "Accepted"} if ok else {"id": 4, "description": "Wrong Answer"},
                })
            return self._send(201, {
                "stdout": body["source_code"], "stderr": None, "time": "0.001", "memory": 1024,
                "status": {"id": 3, "description": "Accepted"}, "token": "tok-1",
            })
        if p == "/api/sandboxes" and m == "GET":
            return self._send(200, {"sandboxes": [SANDBOX]})
        if p == "/api/sandboxes" and m == "POST":
            return self._send(201, dict(
                SANDBOX,
                name=body.get("name") or "",
                environment=body["environment"],
                machine=body["machine"],
            ))
        if p == "/api/sandboxes/sb-123" and m == "GET":
            return self._send(200, SANDBOX)
        if p == "/api/sandboxes/sb-123" and m == "DELETE":
            return self._send(204)
        if p == "/api/sandboxes/sb-123/pause" and m == "POST":
            return self._send(200, dict(SANDBOX, status="paused"))
        if p == "/api/sandboxes/sb-123/resume" and m == "POST":
            return self._send(200, dict(SANDBOX, status="running"))
        if p == "/api/sandboxes/sb-123/fork" and m == "POST":
            return self._send(201, dict(SANDBOX, id="sb-456", name=body.get("name") or "sb-456", forkedFrom="sb-123"))
        if p == "/api/sandboxes/sb-123/exec" and m == "POST":
            return self._send(200, {"stdout": "hi\n", "stderr": "", "exitCode": 0, "durationMs": 12})
        if p == "/api/sandboxes/sb-123/metrics" and m == "GET":
            return self._send(200, {"metrics": [{"cpu": 0.1}]})
        if p == "/api/sandboxes/sb-fail" and m == "GET":
            return self._send(200, dict(SANDBOX, id="sb-fail"))
        if p == "/api/sandboxes/sb-fail/exec" and m == "POST":
            return self._send(200, {"stdout": "", "stderr": "boom\n", "exitCode": 2, "reason": "error"})
        if p == "/api/keys" and m == "GET":
            return self._send(200, {"keys": [{"id": 7, "name": "laptop", "createdAt": "t"}]})
        if p == "/api/keys" and m == "POST":
            return self._send(201, {"id": 8, "name": body["name"], "key": "ak_secret"})
        if p == "/api/keys/8" and m == "DELETE":
            return self._send(204)
        if p == "/api/rl/pools" and m == "GET":
            return self._send(200, {"pools": [{"pool_id": "rp-1", "n": 4}]})

        return self._send(404, {"error": "sandbox not found"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_DELETE(self):
        self._route("DELETE")


@pytest.fixture
def platform():
    FakePlatform.calls = []
    srv = HTTPServer(("127.0.0.1", 0), FakePlatform)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def av(platform):
    client = Client(url=platform, api_key="testkey")
    yield client
    client.close()


# -- the CLI's commands, one by one -----------------------------------------


def test_me(av):
    assert av.me()["email"] == "a@b.c"


def test_environments_and_machines(av):
    environments = av.environments()
    assert [e.name for e in environments] == ["base", "python"]
    assert [e.default for e in environments] == [False, True]

    machines = av.machines()
    assert machines[0].name == "small" and machines[0].memory_mb == 1024
    assert machines[0].rate_usd_per_hour == 0.1


def test_list_and_get(av):
    boxes = av.sandboxes()
    assert len(boxes) == 1 and boxes[0].id == "sb-123"
    assert boxes[0].environment == "python" and boxes[0].memory_mb == 1024

    one = av.sandbox("sb-123")
    assert one.id == "sb-123" and one.status == "running"


def test_create_is_all_keywords_with_documented_defaults(av):
    """Every argument is named at the call site, and the two that have defaults
    send them explicitly so the request says what it asked for."""
    sb = av.create_sandbox(environment="python", name="mybox")
    assert sb.id == "sb-123" and sb.name == "mybox" and sb.environment == "python"
    method, path, body = FakePlatform.calls[-1]
    assert (method, path) == ("POST", "/api/sandboxes")
    assert body == {"machine": "small", "environment": "python", "name": "mybox"}

    av.create_sandbox(environment="python", machine="medium")
    assert FakePlatform.calls[-1][2]["machine"] == "medium"

    # The optional ones stay off the wire until asked for, so an unset argument
    # leaves the platform default in charge rather than pinning it client-side.
    av.create_sandbox(internet=False, idle_timeout=600, max_lifetime=3600)
    body = FakePlatform.calls[-1][2]
    assert body["internet"] is False
    assert body["idleTimeoutSecs"] == 600 and body["maxLifetimeSecs"] == 3600

    av.create_sandbox()
    body = FakePlatform.calls[-1][2]
    assert body == {"machine": "small", "environment": "base"}


def test_exec(av):
    sb = av.sandbox("sb-123")

    res = sb.exec("echo hi")
    assert res.stdout == "hi\n" and res.exit_code == 0 and res.duration_ms == 12
    assert res  # truthy on success — `if sb.exec(...)` has to work
    assert str(res) == "hi\n"  # and printable, so print(sb.exec(...)) works


def test_timeout_uses_the_field_the_backend_reads(av):
    """The API decodes `timeoutS`; anything else is silently ignored and the
    command runs to the server's default instead of the caller's."""
    sb = av.sandbox("sb-123")
    sb.exec("sleep 1", timeout=5)
    assert FakePlatform.calls[-1][2] == {"command": "sleep 1", "timeoutS": 5}


def test_failed_exec_is_a_result_not_an_exception(av):
    """A non-zero exit is data. It is the caller's program that failed, not the
    call — but check() is there when a script should stop."""
    res = av.sandbox("sb-fail").exec("false")
    assert not res and res.exit_code == 2 and res.stderr == "boom\n"
    with pytest.raises(Exception) as exc:
        res.check()
    assert "exited 2" in str(exc.value)


def test_delete_and_context_manager(av):
    sb = av.sandbox("sb-123")
    assert sb.delete() is True
    assert FakePlatform.calls[-1][:2] == ("DELETE", "/api/sandboxes/sb-123")

    # A context manager owns the whole retained workspace, so it must destroy it.
    with av.create_sandbox(environment="python"):
        pass
    assert FakePlatform.calls[-1][:2] == ("DELETE", "/api/sandboxes/sb-123")


def test_pause_and_resume_update_the_object(av):
    sb = av.sandbox("sb-123")
    assert sb.pause() is sb and sb.status == "paused"
    assert FakePlatform.calls[-1][:2] == ("POST", "/api/sandboxes/sb-123/pause")
    assert sb.resume() is sb and sb.status == "running"
    assert FakePlatform.calls[-1][:2] == ("POST", "/api/sandboxes/sb-123/resume")

def test_fork_returns_a_new_sandbox_and_leaves_this_one(av):
    sb = av.sandbox("sb-123")
    fork = sb.fork(name="branch")
    assert FakePlatform.calls[-1][:2] == ("POST", "/api/sandboxes/sb-123/fork")
    assert fork is not sb and fork.id == "sb-456" and fork.name == "branch"
    assert sb.id == "sb-123"


def test_metrics_and_port_url(av, platform):
    sb = av.sandbox("sb-123")
    assert sb.metrics() == {"metrics": [{"cpu": 0.1}]}
    assert sb.url(8080) == f"{platform}/api/sandboxes/sb-123/proxy/8080"
    assert sb.url(8080, "docs") == f"{platform}/api/sandboxes/sb-123/proxy/8080/docs"


def test_api_keys(av):
    assert [k.name for k in av.keys()] == ["laptop"]
    made = av.create_key("ci")
    assert made.key == "ak_secret" and made.id == "8"
    assert av.revoke_key("8") is True


def test_pools_listing(av):
    assert av.pools()[0]["pool_id"] == "rp-1"


def test_unknown_sandbox_is_not_found(av):
    with pytest.raises(NotFoundError):
        av.sandbox("sb-nope")


def test_bad_key_is_an_auth_error(platform):
    with Client(url=platform, api_key="wrong") as bad:
        with pytest.raises(AuthError):
            bad.me()


def test_key_and_url_come_from_the_environment(platform, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # away from any .env in the repo
    monkeypatch.setenv("BOLTZLABS_API_URL", platform)
    monkeypatch.setenv("BOLTZLABS_API_KEY", "testkey")
    with Client() as av:
        assert av.me()["email"] == "a@b.c"


def test_module_level_calls_need_no_client(platform, monkeypatch, tmp_path):
    """`boltzlabs.sandboxes()` with nothing built first — the shortest thing that
    should work."""
    monkeypatch.chdir(tmp_path)
    boltzlabs.use(api_key="testkey", url=platform)
    assert boltzlabs.me()["email"] == "a@b.c"
    assert [s.id for s in boltzlabs.sandboxes()] == ["sb-123"]
    assert [str(e) for e in boltzlabs.environments()] == ["base", "python"]
    assert [str(m) for m in boltzlabs.machines()] == ["small"]
    assert boltzlabs.sandbox("sb-123").id == "sb-123"

    # Sandbox() with no client= uses that same default.
    sb = Sandbox()
    assert sb.id == "sb-123"


def test_key_comes_from_a_dotenv_file(platform, monkeypatch, tmp_path):
    monkeypatch.delenv("BOLTZLABS_API_KEY", raising=False)
    monkeypatch.delenv("BOLTZLABS_API_URL", raising=False)
    (tmp_path / ".env").write_text(
        f"# a comment\nexport BOLTZLABS_API_URL={platform}\nBOLTZLABS_API_KEY='testkey'  \n"
    )
    monkeypatch.chdir(tmp_path)

    from boltzlabs import config

    config._cache.clear()
    with Client() as av:
        assert av.me()["email"] == "a@b.c"


def test_missing_key_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.delenv("BOLTZLABS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    from boltzlabs import config

    config._cache.clear()
    with pytest.raises(AuthError) as exc:
        Client()
    assert "BOLTZLABS_API_KEY" in str(exc.value)


def test_default_url_is_the_public_origin(monkeypatch, tmp_path):
    """Never the Go control plane: it binds loopback and is not reachable."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BOLTZLABS_API_URL", raising=False)
    monkeypatch.setenv("BOLTZLABS_API_KEY", "testkey")
    from boltzlabs import config

    config._cache.clear()
    assert Client().url == "https://boltzlabs.cloud"


def test_execute_takes_code_or_a_file(av, tmp_path):
    """The exec plane: send code, get what it printed. Nothing is created
    first and nothing is left over. The wire is the standard submission JSON."""
    res = av.execute("print(1)", language="python")
    assert str(res) == "print(1)" and res and res.status == {"id": 3, "description": "Accepted"}
    _, path, body = FakePlatform.calls[-1]
    assert path == "/api/execute?wait=true&fields=*"
    assert body == {"source_code": "print(1)", "language_id": 113}

    # A file path is resolved here, not on the platform — the wire only ever
    # carries code, so the server never resolves a path it did not write.
    script = tmp_path / "train.py"
    script.write_text("print('from a file')")
    av.execute(file=str(script), language=113)
    assert FakePlatform.calls[-1][2] == {"source_code": "print('from a file')", "language_id": 113}


def test_execute_needs_exactly_one_of_code_or_file(av, tmp_path):
    with pytest.raises(ValueError):
        av.execute(language="python")
    with pytest.raises(ValueError):
        av.execute("print(1)", file=str(tmp_path / "x.py"), language="python")


def test_language_is_required_in_both_forms(av, tmp_path):
    """Never inferred, from an extension or otherwise: a .py file is as likely to
    be torch as plain python, and inline code has no extension at all."""
    with pytest.raises(ValueError):
        av.execute("print(1)")

    script = tmp_path / "train.py"
    script.write_text("print(1)")
    with pytest.raises(ValueError):
        av.execute(file=str(script))
    with pytest.raises(ValueError):
        av.execute("print(1)", language="cobol")


def test_languages_are_listed(av):
    langs = av.languages()
    assert [(l.id, l.code) for l in langs] == [(102, "node"), (103, "c"), (113, "python")]
    assert langs[1].compiled and langs[2].name == "Python (3.14)" and langs[2].label == "Python (3.14)"


def test_a_program_that_does_not_compile_is_a_result_not_an_exception(av):
    """The program never ran, so what comes back is the compiler's message as
    compile_output, under Compilation Error."""
    res = av.execute("int main(void){ syntax error }", language="c")
    assert res.status_id == 6 and not res
    assert "expected ';'" in str(res)
    with pytest.raises(BoltzLabsError, match="Compilation Error"):
        res.check()


def test_execute_judges_a_solution(av):
    """Test input, the problem's limits and the expected answer go out in the
    standard fields; the status, time and memory come back in them."""
    res = av.execute(
        "print(sum(map(int, input().split())))", language="python",
        stdin="1 2 3\n", expected_output="6", cpu_time_limit=1, memory_limit=65536, supersede_key="tab-1",
    )
    body = FakePlatform.calls[-1][2]
    assert body == {
        "source_code": "print(sum(map(int, input().split())))", "language_id": 113, "stdin": "1 2 3\n",
        "expected_output": "6", "cpu_time_limit": 1, "memory_limit": 65536, "supersede_key": "tab-1",
    }
    assert res.accepted and res.time == "0.012" and res.memory == 9216 and res.json["token"] == "tok-j"

    wrong = av.execute("print(7)", language="python", expected_output="7")
    assert wrong.status_id == 4 and not wrong
    with pytest.raises(BoltzLabsError, match="Wrong Answer"):
        wrong.check()


def test_execute_batch_runs_the_test_cases_together(av):
    results = av.execute_batch([
        {"code": "print(input())", "language": "python", "stdin": "a"},
        {"source_code": "print(input())", "language_id": 113, "stdin": "b"},
    ])
    posted = [c for c in FakePlatform.calls if c[1] == "/api/execute/batch"][-1][2]
    assert posted == {"submissions": [
        {"source_code": "print(input())", "language_id": 113, "stdin": "a"},
        {"source_code": "print(input())", "language_id": 113, "stdin": "b"},
    ]}
    assert [r.stdout for r in results] == ["out t0\n", "out t1\n"] and all(results)


def test_superseded_is_its_own_error_not_a_quota_error():
    """A run replaced by a newer one under the same supersede_key is not out of
    quota; a caller branching on the class must be able to tell."""
    from boltzlabs.errors import QuotaError, SupersededError, from_status

    assert isinstance(from_status(409, "replaced", {"code": "superseded"}), SupersededError)
    assert isinstance(from_status(409, "limit", {"error": "limit"}), QuotaError)
    assert boltzlabs.SupersededError is SupersededError
