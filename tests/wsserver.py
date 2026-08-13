"""A `websockets` server on a background event loop, shared by the tests.

Threads rather than pytest-asyncio: the client under test is synchronous and
blocking, which is what a terminal wants, so the tests have to drive it from an
ordinary thread while the server runs its own loop.

`websockets` is a dev dependency and never a runtime one — testing a hand-written
framer against an implementation this repo did not write is the entire point.
"""

import asyncio
import threading

import pytest

websockets = pytest.importorskip("websockets")


class Server:
    """A websockets server on a background event loop.

    Threads rather than pytest-asyncio: the client under test is synchronous and
    blocking, which is what a terminal wants, so the test has to drive it from
    an ordinary thread.
    """

    def __init__(self, handler, auth=None):
        self.handler = handler
        self.auth = auth
        self.loop = asyncio.new_event_loop()
        self.port = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())
        self.loop.run_forever()

    async def _serve(self):
        async def process_request(connection, request):
            if self.auth is None:
                return None
            if request.headers.get("Authorization") != self.auth:
                return connection.respond(401, "nope\n")
            if request.path.endswith("/missing/terminal"):
                return connection.respond(404, "no such sandbox\n")
            return None

        self.server = await websockets.serve(
            self.handler, "127.0.0.1", 0, process_request=process_request
        )
        self.port = self.server.sockets[0].getsockname()[1]
        self._ready.set()

    def start(self):
        self._thread.start()
        assert self._ready.wait(10), "server never started"
        return f"ws://127.0.0.1:{self.port}/"

    def stop(self):
        # Close the listener and let pending handlers unwind before the loop
        # goes away, or teardown raises "no running event loop" out of them.
        async def shutdown():
            self.server.close()
            await self.server.wait_closed()
            self.loop.stop()

        asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
        self._thread.join(timeout=5)
