"""The hand-written WebSocket client, against a real server.

`websockets` is a dev dependency and never a runtime one. Testing the framer
against it is the whole point: a framer only proves anything by interoperating,
so masking, both extended length encodings, fragmentation and ping/pong are all
exercised against an implementation this code did not write.
"""

import asyncio

import pytest

from boltzlabs._ws import OP_BINARY, OP_TEXT, connect
from boltzlabs.errors import AuthError, NotFoundError

pytest.importorskip("websockets")

from wsserver import Server  # noqa: E402 — after the skip, so a missing dep skips cleanly


@pytest.fixture
def serve():
    made = []

    def _make(handler, auth=None):
        s = Server(handler, auth)
        made.append(s)
        return s.start()

    yield _make
    for s in made:
        s.stop()


def test_echo_round_trip(serve):
    async def handler(ws):
        async for msg in ws:
            await ws.send(msg)

    url = serve(handler)
    with connect(url) as ws:
        ws.send(b"hello")
        op, payload = ws.recv()
        assert op == OP_BINARY and payload == b"hello"

        ws.send_text("text frame")
        op, payload = ws.recv()
        assert op == OP_TEXT and payload == b"text frame"


@pytest.mark.parametrize("size", [125, 126, 1000, 70000])
def test_every_length_encoding(serve, size):
    """7-bit, 16-bit and 64-bit payload lengths — the three framing branches."""

    async def handler(ws):
        async for msg in ws:
            await ws.send(msg)

    url = serve(handler)
    with connect(url) as ws:
        blob = bytes(range(256)) * (size // 256) + b"x" * (size % 256)
        ws.send(blob)
        _, payload = ws.recv()
        assert payload == blob


def test_fragmented_server_message_is_reassembled(serve):
    async def handler(ws):
        await ws.recv()
        await ws.send([b"one ", b"two ", b"three"])  # sent as fragments

    url = serve(handler)
    with connect(url) as ws:
        ws.send(b"go")
        _, payload = ws.recv()
        assert payload == b"one two three"


def test_ping_is_answered(serve):
    """A ping that goes unanswered gets the connection dropped, so the client
    has to reply without the caller knowing pings exist."""

    async def handler(ws):
        pong = await ws.ping()
        await asyncio.wait_for(pong, timeout=5)
        await ws.send(b"ponged")

    url = serve(handler)
    with connect(url) as ws:
        op, payload = ws.recv()
        assert payload == b"ponged"


def test_close_ends_recv(serve):
    async def handler(ws):
        await ws.send(b"bye")

    url = serve(handler)
    with connect(url) as ws:
        assert ws.recv()[1] == b"bye"
        assert ws.recv() is None  # a clean close, not an exception


def test_auth_header_is_sent_and_failures_map_to_exceptions(serve):
    async def handler(ws):
        await ws.send(b"in")

    url = serve(handler, auth="Bearer good")

    with connect(url, headers={"Authorization": "Bearer good"}) as ws:
        assert ws.recv()[1] == b"in"

    with pytest.raises(AuthError):
        connect(url, headers={"Authorization": "Bearer bad"})

    with pytest.raises(NotFoundError):
        connect(url + "missing/terminal", headers={"Authorization": "Bearer good"})
