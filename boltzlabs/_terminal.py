"""`boltz connect`, as Python: an interactive shell on a sandbox.

Two entry points over the same WebSocket:

* ``attach`` — bridges the local terminal, for a person at a keyboard.
* ``run_script`` — sends a script, collects everything printed, returns it. For
  the cases where a command only behaves correctly with a PTY attached and
  ``exec`` is not enough.

The endpoint is the public origin's ``/api/sandboxes/{id}/terminal``; SvelteKit
performs the upgrade and proxies it to the control plane, which attaches to the
worker holding the sandbox. The API key rides in the ``Authorization`` header —
something a browser cannot set on a WebSocket, and a native client can.
"""

import os
import sys
import threading
import time

from ._ws import connect
from .errors import TransportError

__all__ = ["attach", "run_script"]

# How long a non-interactive session waits for further output after its input
# ends. Only reached when a script does not end with `exit`.
IDLE_AFTER_EOF = 2.0


def _ws_url(client, sandbox_id):
    url = client.url
    if url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    return f"{url}/api/sandboxes/{sandbox_id}/terminal"


def _open(client, sandbox_id, timeout=30.0):
    return connect(
        _ws_url(client, sandbox_id),
        headers={"Authorization": f"Bearer {client._api_key}"},
        timeout=timeout,
    )


def attach(client, sandbox_id, stdin=None, stdout=None, raw=None):
    """Bridge the local terminal to the sandbox's shell until either end closes.

    Returns when the remote closes (`exit`, Ctrl-D) — the same contract as
    `boltz connect`.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    in_fd = _fileno(stdin)
    out = getattr(stdout, "buffer", stdout)
    is_tty = raw if raw is not None else (in_fd is not None and _isatty(in_fd))

    ws = _open(client, sandbox_id)
    last_output = [time.monotonic()]
    stop = threading.Event()

    def pump_out():
        # Raw bytes straight to the local terminal: the remote shell owns echo,
        # colour and cursor movement, and decoding here would break anything
        # that is not valid UTF-8 mid-frame.
        try:
            while not stop.is_set():
                msg = ws.recv()
                if msg is None:
                    break
                out.write(msg[1])
                out.flush()
                last_output[0] = time.monotonic()
        except (TransportError, OSError, ValueError):
            pass
        finally:
            stop.set()

    reader = threading.Thread(target=pump_out, daemon=True)
    reader.start()

    restore = _raw_mode(in_fd) if is_tty else None
    try:
        src = getattr(stdin, "buffer", stdin)
        while not stop.is_set():
            chunk = _read_some(src, in_fd)
            if chunk:
                ws.send(chunk)
                continue
            # EOF on stdin must not tear the session down. With piped input, EOF
            # arrives immediately after the bytes are written, and closing here
            # would cut the shell off before its output came back — the command
            # would appear to produce nothing.
            if is_tty:
                break
            while not stop.is_set():
                if time.monotonic() - last_output[0] > IDLE_AFTER_EOF:
                    break
                time.sleep(0.05)
            break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if restore:
            restore()
        ws.close()
        reader.join(timeout=2.0)
    return True


def run_script(client, sandbox_id, script, timeout=60.0, idle=2.0):
    """Send ``script`` to a shell and return everything it printed.

    The transcript is raw terminal output: it contains the shell's own echo of
    what was typed, its prompts, and any escape sequences. That is what a PTY
    produces, and cleaning it up here would be guessing at which parts the
    caller considers noise.
    """
    ws = _open(client, sandbox_id)
    ws.settimeout(min(timeout, 5.0))
    chunks = []
    deadline = time.monotonic() + timeout
    try:
        body = script if script.endswith("\n") else script + "\n"
        # `exit` so the remote closes the connection when the script is done,
        # which is a definite end rather than an inference from silence.
        ws.send((body + "exit\n").encode("utf-8"))

        last = time.monotonic()
        while True:
            if time.monotonic() > deadline:
                break
            try:
                msg = ws.recv()
            except (TimeoutError, OSError):
                if time.monotonic() - last > idle:
                    break
                continue
            if msg is None:
                break
            chunks.append(msg[1])
            last = time.monotonic()
    finally:
        ws.close()
    return b"".join(chunks).decode("utf-8", "replace")


# -- local terminal plumbing -------------------------------------------------


def _fileno(stream):
    try:
        return stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None


def _isatty(fd):
    try:
        return os.isatty(fd)
    except OSError:
        return False


def _raw_mode(fd):
    """Put the local terminal in raw mode; return the restore callable.

    Restoring on *every* exit path is the point — a missed restore leaves the
    user's shell with no echo and no line editing, which looks like the machine
    broke. Returns None where termios does not exist (Windows), so the feature
    degrades to line-at-a-time instead of failing.
    """
    if fd is None:
        return None
    try:
        import termios
        import tty
    except ImportError:
        return None
    try:
        saved = termios.tcgetattr(fd)
    except (termios.error, OSError):
        return None
    tty.setraw(fd)

    def restore():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except (termios.error, OSError):
            pass

    return restore


def _read_some(src, fd):
    """One read, as small as is available.

    ``os.read`` on the raw fd rather than a buffered ``read(n)``: the buffered
    one blocks until it has n bytes, which in an interactive session means the
    first keystroke is never sent.
    """
    if fd is not None:
        try:
            return os.read(fd, 4096)
        except (OSError, ValueError):
            return b""
    data = src.read(4096)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data or b""
