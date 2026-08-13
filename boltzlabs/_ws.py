"""A small RFC 6455 client, for one endpoint: the sandbox terminal.

Hand-written rather than a dependency. The terminal is a raw byte pipe — no
subprotocol, no extensions, no compression — so what is needed is the handshake,
masked client frames, unmasked server frames, and ping/pong. That is this file.
Making `pip install boltzlabs` pull in a WebSocket stack for a feature most callers
never touch would be the worse trade.

It is tested against a real `websockets` server (a dev dependency, never a
runtime one), including fragmentation, ping/pong and both length encodings —
because "I wrote a framer" is not evidence that it interoperates.
"""

import base64
import os
import socket
import ssl
import struct
from urllib.parse import urlsplit

from .errors import TransportError, from_status

__all__ = ["WebSocket", "connect"]

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocket:
    """One connection. Not thread-safe for sending; reads and writes on
    separate threads are fine, which is exactly what a terminal needs."""

    def __init__(self, sock, url):
        self.sock = sock
        self.url = url
        self.closed = False
        self._buf = b""

    # -- reading ------------------------------------------------------------

    def _read_exact(self, n):
        while len(self._buf) < n:
            try:
                chunk = self.sock.recv(65536)
            except (socket.timeout, TimeoutError):
                raise
            except OSError as exc:
                raise TransportError(f"websocket read failed: {exc}") from exc
            if not chunk:
                self.closed = True
                return None
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv(self):
        """Next application message as ``(opcode, payload)``, or ``None`` at close.

        Control frames are handled here rather than surfaced: a ping that is not
        ponged closes the connection from the other end, and no caller of a
        terminal wants to be responsible for that. Continuation frames are
        reassembled, so a caller sees whole messages.
        """
        frames = []
        first_op = None

        while True:
            header = self._read_exact(2)
            if header is None:
                return None
            b0, b1 = header[0], header[1]
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F

            if length == 126:
                ext = self._read_exact(2)
                if ext is None:
                    return None
                length = struct.unpack("!H", ext)[0]
            elif length == 127:
                ext = self._read_exact(8)
                if ext is None:
                    return None
                length = struct.unpack("!Q", ext)[0]

            mask_key = None
            if masked:
                # A server must not mask. Refusing is the correct reading of the
                # spec and keeps a broken peer from looking like corrupt output.
                mask_key = self._read_exact(4)
                if mask_key is None:
                    return None

            payload = b"" if length == 0 else self._read_exact(length)
            if payload is None:
                return None
            if mask_key:
                payload = _xor(payload, mask_key)

            if opcode == OP_PING:
                self.send(payload, OP_PONG)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                try:
                    self.send(payload[:2], OP_CLOSE)
                except Exception:  # noqa: BLE001 — the peer is already leaving
                    pass
                self.closed = True
                return None

            if opcode == OP_CONT:
                frames.append(payload)
            else:
                first_op, frames = opcode, [payload]

            if fin:
                return first_op, b"".join(frames)

    # -- writing ------------------------------------------------------------

    def send(self, data, opcode=OP_BINARY):
        """Send one frame. Client frames are always masked, as the spec requires
        — an unmasked one is dropped by every conformant server."""
        if isinstance(data, str):
            data = data.encode("utf-8")
            opcode = OP_TEXT if opcode == OP_BINARY else opcode

        header = bytearray()
        header.append(0x80 | opcode)  # FIN set: no fragmentation on this side
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < (1 << 16):
            header.append(0x80 | 126)
            header += struct.pack("!H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", n)

        mask_key = os.urandom(4)
        header += mask_key
        try:
            self.sock.sendall(bytes(header) + _xor(data, mask_key))
        except OSError as exc:
            self.closed = True
            raise TransportError(f"websocket write failed: {exc}") from exc

    def send_text(self, text):
        self.send(text, OP_TEXT)

    # -- lifecycle ----------------------------------------------------------

    def close(self, code=1000):
        if self.closed:
            return
        self.closed = True
        try:
            self.send(struct.pack("!H", code), OP_CLOSE)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def settimeout(self, timeout):
        self.sock.settimeout(timeout)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _xor(data, key):
    # bytes are immutable and this runs per frame on an interactive stream;
    # a bytearray in place is measurably cheaper than a comprehension.
    out = bytearray(data)
    for i in range(len(out)):
        out[i] ^= key[i & 3]
    return bytes(out)


def connect(url, headers=None, timeout=30.0):
    """Open a WebSocket. ``url`` may be ws://, wss://, http:// or https://."""
    parts = urlsplit(url)
    secure = parts.scheme in ("wss", "https")
    host = parts.hostname
    if not host:
        raise ValueError(f"no host in {url!r}")
    port = parts.port or (443 if secure else 80)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query

    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        raise TransportError(f"cannot reach {host}:{port}: {exc}") from exc
    if secure:
        try:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        except OSError as exc:
            sock.close()
            raise TransportError(f"TLS handshake with {host} failed: {exc}") from exc

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    lines = [
        f"GET {path} HTTP/1.1",
        f"Host: {parts.netloc}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    # No Origin header on purpose. The control plane treats an absent Origin as
    # a non-browser client and skips the browser origin allowlist — the same
    # thing the Go CLI relies on.
    for k, v in (headers or {}).items():
        lines.append(f"{k}: {v}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))

    status, resp_headers, leftover = _read_handshake(sock)
    if status != 101:
        sock.close()
        # A real HTTP status here is the useful error: 401 means the key, 404
        # means the sandbox, 409 means it is not running. Reuse the same mapping
        # the REST calls use so callers catch the same exceptions.
        raise from_status(status, _handshake_error(status), None)

    accept = resp_headers.get("sec-websocket-accept", "")
    import hashlib

    expected = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
    if accept != expected:
        sock.close()
        raise TransportError("websocket handshake failed: bad Sec-WebSocket-Accept")

    ws = WebSocket(sock, url)
    ws._buf = leftover
    return ws


def _read_handshake(sock):
    """Read the response head, and nothing past it.

    Byte at a time: anything read beyond the blank line is frame data, and
    over-reading it into a discarded buffer would lose the first output of the
    session. (It is ~200 bytes, once per connection.)
    """
    data = b""
    while b"\r\n\r\n" not in data:
        try:
            chunk = sock.recv(1)
        except OSError as exc:
            raise TransportError(f"websocket handshake failed: {exc}") from exc
        if not chunk:
            raise TransportError("websocket handshake failed: connection closed")
        data += chunk
        if len(data) > 64 * 1024:
            raise TransportError("websocket handshake failed: response head too large")

    head, _, rest = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    try:
        status = int(lines[0].split(" ")[1])
    except (IndexError, ValueError):
        raise TransportError(f"websocket handshake failed: bad status line {lines[0]!r}")

    headers = {}
    for line in lines[1:]:
        k, _, v = line.partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers, rest


def _handshake_error(status):
    return {
        401: "unauthorized — the API key is invalid or revoked",
        403: "forbidden",
        404: "sandbox not found",
        409: "sandbox is not running",
        501: "this origin does not proxy the terminal WebSocket",
        503: "no worker attached — the terminal is unavailable",
    }.get(status, f"terminal connection refused (HTTP {status})")
