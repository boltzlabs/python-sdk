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
            return self._send(200, {"languages": [
                {"code": "python", "label": "Python 3", "extension": ".py", "compiled": False},
                {"code": "node", "label": "Node.js", "extension": ".js", "compiled": False},
                {"code": "c", "label": "C", "extension": ".c", "compiled": True},
            ]})
        if p == "/api/execute" and m == "POST":
            # A compiled language that does not compile: the program never ran,
            # so stderr is the compiler's and there is no stdout.
            if body.get("language") == "c" and "syntax error" in body["code"]:
                return self._send(200, {
                    "stdout": "", "stderr": "main.c:1: expected ';'", "exitCode": 1,
                    "durationMs": 90, "compileMs": 90, "compileFailed": True,
                })
            return self._send(200, {
                "stdout": body["code"], "stderr": "", "exitCode": 0, "durationMs": 7,
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
        if p == "/api/sandboxes/sb-123/exec" and m == "POST":
            return self._send(200, {"stdout": "hi\n", "stderr": "", "exitCode": 0, "durationMs": 12})
        if p == "/api/sandboxes/sb-123/run" and m == "POST":
            return self._send(200, {"stdout": body["code"], "exitCode": 0, "durationMs": 3})
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


def test_exec_and_run(av):
    sb = av.sandbox("sb-123")

    res = sb.exec("echo hi")
    assert res.stdout == "hi\n" and res.exit_code == 0 and res.duration_ms == 12
    assert res  # truthy on success — `if sb.exec(...)` has to work
    assert str(res) == "hi\n"  # and printable, so print(sb.exec(...)) works

    # The language follows the sandbox type, so the common call is one argument.
    res = sb.run("print(1)")
    assert res.stdout == "print(1)"
    _, path, body = FakePlatform.calls[-1]
    assert path == "/api/sandboxes/sb-123/run"
    assert body == {"language": "python", "code": "print(1)"}

    sb.run("console.log(1)", language="node")
    assert FakePlatform.calls[-1][2]["language"] == "node"


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

    # A sandbox bills for as long as it exists, so `with` has to destroy it.
    with av.create_sandbox(environment="python"):
        pass
    assert FakePlatform.calls[-1][:2] == ("DELETE", "/api/sandboxes/sb-123")


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
    """The whole execution product: send code, get what it printed. Nothing is
    created first and nothing is left over."""
    res = av.execute("print(1)", language="python")
    assert str(res) == "print(1)" and res.exit_code == 0
    _, path, body = FakePlatform.calls[-1]
    assert path == "/api/execute"
    assert body == {"code": "print(1)", "language": "python"}

    # A file path is resolved here, not on the platform — the wire only ever
    # carries code, so the server never resolves a path it did not write.
    script = tmp_path / "train.py"
    script.write_text("print('from a file')")
    res = av.execute(file=str(script), language="python")
    body = FakePlatform.calls[-1][2]
    assert body["code"] == "print('from a file')"
    assert body["filename"] == "train.py"


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


def test_execute_passes_language_and_timeout(av):
    av.execute("console.log(1)", language="node", timeout=45)
    body = FakePlatform.calls[-1][2]
    assert body["language"] == "node" and body["timeoutS"] == 45


def test_languages_are_listed(av):
    langs = av.languages()
    assert [str(l) for l in langs] == ["python", "node", "c"]
    assert langs[0].label == "Python 3" and langs[0].extension == ".py"
    # Compiled is what tells a caller why part of their run was the compiler.
    assert [l.compiled for l in langs] == [False, False, True]


def test_a_program_that_does_not_compile_is_a_result_not_an_exception(av):
    """The call succeeded; the code was rejected. Those are different, and only
    compile_failed distinguishes "never ran" from "ran and printed to stderr"."""
    res = av.execute("int main(void){ syntax error }", language="c")
    assert res.compile_failed is True
    assert res.exit_code == 1
    assert res.stdout == ""
    assert "expected ';'" in res.stderr
    assert res.compile_ms == 90

    # check() names the compiler rather than an exit code that explains nothing.
    with pytest.raises(BoltzLabsError, match="did not compile"):
        res.check()


def test_an_interpreted_run_carries_no_compile_fields(av):
    res = av.execute("print(1)", language="python")
    assert res.compile_failed is False and res.compile_ms == 0


def test_a_quoted_value_ends_at_its_closing_quote(tmp_path):
    """`KEY="v"  # note` is the shape that used to hand back the quotes as part
    of the secret — a 401 from a key that looks correct in the file."""
    from boltzlabs.config import _parse

    env = tmp_path / ".env"
    env.write_text(
        'BOLTZLABS_API_KEY="ak_quoted"  # trailing comment\n'
        'KEEPS_HASH="has#hash"\n'
        "UNQUOTED=plain # note\n"
    )
    parsed = _parse(str(env))
    assert parsed["BOLTZLABS_API_KEY"] == "ak_quoted"
    assert parsed["KEEPS_HASH"] == "has#hash"
    assert parsed["UNQUOTED"] == "plain"
