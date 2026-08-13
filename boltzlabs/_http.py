"""A small keep-alive JSON client, on http.client rather than a dependency.

``step`` is the call a training loop makes thousands of times, and on a fast
worker the batch itself is single-digit milliseconds. A fresh TCP connection
(and a fresh TLS handshake) per step would dominate that completely — the number
the SDK reports would be a measurement of connection setup. So the connection is
opened once and reused, and this module is where that, the timeouts and the
retry rule are all written down in one place.
"""

import http.client
import json
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit

from .errors import TransportError, from_status

USER_AGENT = "boltzlabs-python/0.1.0"


class Session:
    """One reusable connection to one origin.

    Not safe to share across threads for pipelined use — the lock serialises
    requests rather than parallelising them. That is the honest shape: HTTP/1.1
    on one socket is serial, and pretending otherwise would hide queueing time
    inside the latency numbers this SDK exists to report.
    """

    def __init__(self, base_url, headers=None, timeout=60.0):
        parts = urlsplit(base_url if "://" in base_url else "http://" + base_url)
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"unsupported scheme {parts.scheme!r} in {base_url!r}")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        # A control plane may live under a path prefix behind a reverse proxy.
        self.prefix = parts.path.rstrip("/")
        self.origin = f"{self.scheme}://{parts.netloc}"
        self.timeout = timeout
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "keep-alive",
        }
        self.headers.update(headers or {})

        self._lock = threading.Lock()
        self._conn = None

    # -- connection ---------------------------------------------------------

    def _connect(self, timeout):
        if self.scheme == "https":
            conn = http.client.HTTPSConnection(
                self.host, self.port, timeout=timeout, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        conn.connect()
        # Nagle would add up to 40 ms to a small request that is followed by a
        # blocking read. That is an order of magnitude more than the step it is
        # carrying.
        try:
            conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return conn

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001 — closing must never raise
                    pass
                self._conn = None

    # -- requests -----------------------------------------------------------

    def request(self, method, path, body=None, timeout=None):
        """Send one JSON request, return ``(status, decoded_body, elapsed_ms)``.

        ``elapsed_ms`` is the socket round trip alone; the caller adds its own
        encode/decode cost to get the number a trainer actually pays.
        """
        timeout = self.timeout if timeout is None else timeout
        payload, headers = self._encode(body)

        with self._lock:
            # One retry, and only for a connection that was already open. A
            # server closing an idle keep-alive socket is the ordinary case here,
            # and it fails before the request is ever handled — so re-sending is
            # safe even for a non-idempotent POST like /step. A connection that
            # failed on its first use is a real failure and is raised as one:
            # re-sending that could double-apply a step.
            for attempt in (0, 1):
                reused = self._conn is not None
                if not reused:
                    try:
                        self._conn = self._connect(timeout)
                    except (OSError, http.client.HTTPException) as exc:
                        self._conn = None
                        raise TransportError(
                            f"cannot reach {self.origin}: {exc}"
                        ) from exc
                else:
                    # Per-request deadline on a connection that outlives it:
                    # creating a pool is minutes and a step is milliseconds, and
                    # neither should have to pay for a new socket to say so.
                    try:
                        self._conn.sock.settimeout(timeout)
                    except (AttributeError, OSError):
                        pass

                started = time.perf_counter()
                try:
                    self._conn.request(method, self.prefix + path, body=payload, headers=headers)
                    resp = self._conn.getresponse()
                    raw = resp.read()
                except (OSError, http.client.HTTPException) as exc:
                    try:
                        self._conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._conn = None
                    if reused and attempt == 0:
                        continue
                    if isinstance(exc, socket.timeout):
                        raise TransportError(
                            f"{method} {path} timed out after {timeout:.0f}s"
                        ) from exc
                    raise TransportError(f"{method} {path} failed: {exc}") from exc

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if resp.will_close:
                    # The server is not keeping it; drop it rather than hand the
                    # next call a socket that is already half closed.
                    try:
                        self._conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._conn = None

                return resp.status, _decode(raw), elapsed_ms

    def call(self, method, path, body=None, timeout=None):
        """``request`` plus the error mapping. Returns ``(body, elapsed_ms)``."""
        status, decoded, elapsed_ms = self.request(method, path, body, timeout)
        if status >= 400:
            raise from_status(status, _message(decoded, status), decoded)
        return decoded, elapsed_ms

    # -- encoding -----------------------------------------------------------

    def _encode(self, body):
        # No Content-Encoding on the way out. Go's net/http does not decompress
        # request bodies, so a gzipped request would reach the worker's JSON
        # decoder as bytes — and the one body large enough to be worth
        # compressing is the create, which is already carrying a gzipped tar.
        headers = dict(self.headers)
        if body is None:
            return None, headers
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(payload))
        return payload, headers


def _decode(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        # An HTML error page from a proxy in front of the control plane. Keep the
        # text: "<html>502 Bad Gateway" is a more useful message than "invalid
        # JSON", because it names the hop that actually failed.
        return {"error": raw.decode("utf-8", "replace")[:500]}


def _message(decoded, status):
    if isinstance(decoded, dict):
        for key in ("error", "message", "detail"):
            if decoded.get(key):
                return str(decoded[key])
    return http.client.responses.get(status, "request failed")
