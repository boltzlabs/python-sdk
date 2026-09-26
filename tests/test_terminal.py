"""`boltz connect`, as Python — against a stand-in terminal endpoint.

The fake behaves the way the real one does: raw bytes in, raw bytes out, and the
server closes when the shell exits. What is under test is the bridging — that
input reaches the far end, that output reaches the caller's stream, and that EOF
on a pipe does not cut the session off before the output comes back, which is
the bug that would make every scripted use silently return nothing.
"""

import io
import threading

import pytest

from boltzlabs import Client, Sandbox
from boltzlabs import _terminal

pytest.importorskip("websockets")

from wsserver import Server  # noqa: E402 — after the skip


class Recorder(io.RawIOBase):
    """A stdout stand-in that can be read back, and that `attach` can flush."""

    def __init__(self):
        self.chunks = []
        self._lock = threading.Lock()

    def write(self, b):
        with self._lock:
            self.chunks.append(bytes(b))
        return len(b)

    def flush(self):
        pass

    def value(self):
        with self._lock:
            return b"".join(self.chunks)


@pytest.fixture
def terminal(monkeypatch):
    """A sandbox pointed at a fake terminal, plus what the shell received."""
    received = []
    servers = []

    def _make(transcript=b"$ ", close_after=None):
        async def handler(ws):
            await ws.send(transcript)
            try:
                async for msg in ws:
                    received.append(msg)
                    # Echo like a shell would, so the caller sees its own input
                    # come back the way a PTY sends it.
                    await ws.send(b"> " + msg)
                    if close_after and close_after in msg:
                        break
            except Exception:  # noqa: BLE001 — client hung up first
                pass

        server = Server(handler)
        url = server.start()
        servers.append(server)
        # attach()/run_script() build ws:// from the client's http:// origin.
        http_url = "http://" + url[len("ws://") :].rstrip("/")
        client = Client(url=http_url, api_key="testkey")
        # Built from a wire record rather than fetched: the fake serves the
        # terminal socket and nothing else, and this is the same object a
        # listing hands back.
        sb = Sandbox._attach({"id": "sb-1", "environment": "base"}, client)
        return sb, received

    yield _make
    for s in servers:
        s.stop()


def test_run_script_sends_the_script_and_returns_the_transcript(terminal):
    sb, received = terminal(transcript=b"welcome\n", close_after=b"exit")
    out = sb.terminal("echo hello", timeout=10)

    # One write carrying the script and the exit that ends the session.
    assert b"echo hello\n" in received[0]
    assert b"exit\n" in received[0]
    # Everything the far end printed, in order.
    assert out.startswith("welcome\n")
    assert "echo hello" in out


def test_attach_pipes_stdin_to_the_shell_and_output_back(terminal, monkeypatch):
    # Piped input, not a TTY: EOF must not tear the session down before the
    # shell's answer arrives. Shorten the idle wait so the test is not 2s long.
    monkeypatch.setattr(_terminal, "IDLE_AFTER_EOF", 0.3)

    sb, received = terminal(transcript=b"ready\n")
    out = Recorder()
    sb.terminal(stdin=io.BytesIO(b"whoami\n"), stdout=out, raw=False)

    assert received and b"whoami\n" in received[0]
    text = out.value()
    assert b"ready\n" in text
    assert b"> whoami\n" in text  # the shell's reply arrived after local EOF


def test_terminal_url_is_ws_over_the_public_origin():
    """https → wss, and the path the control plane actually serves."""
    client = Client(url="https://boltzlabs.cloud", api_key="k")
    assert (
        _terminal._ws_url(client, "sb-9")
        == "wss://boltzlabs.cloud/api/sandboxes/sb-9/terminal"
    )
    local = Client(url="http://localhost:5173", api_key="k")
    assert _terminal._ws_url(local, "sb-9") == "ws://localhost:5173/api/sandboxes/sb-9/terminal"
