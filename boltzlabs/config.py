"""Where the SDK points, and what it authenticates with.

**The endpoint is always the public SvelteKit origin, never the Go control
plane.** The control plane binds loopback and is not reachable from the
internet; everything public — the dashboard, the CLI, and this SDK — enters
through SvelteKit, which proxies `/api/*` and the terminal's WebSocket upgrade
to it. `RLPool(direct=...)` is the one deliberate exception, and it dials a
*worker*, not the control plane, from inside the deployment or over a tunnel.

The SDK stands on its own: it does not read the CLI's login. Credentials come
from the environment, and from a `.env` file next to the code that uses it —
which is where a training script's secrets already live.

Resolution order, highest first:

1. what the caller passed
2. the real environment (`BOLTZLABS_API_KEY`, `BOLTZLABS_API_URL`)
3. a `.env` file, searched from the current directory upwards
4. the production origin, for the URL only

Real environment variables beat `.env` deliberately: that is what makes
`BOLTZLABS_API_KEY=... python train.py` work as a one-off override, and it is what
every other dotenv implementation does, so the precedence is not a surprise.

A missing key is an error rather than an anonymous request: every route this SDK
calls is owner-scoped, so an unauthenticated call can only ever become a 401
further from where the mistake was made.
"""

import os

from .errors import AuthError

__all__ = ["DEFAULT_API_URL", "resolve", "get", "load_dotenv", "find_dotenv"]

# The public origin. Deliberately the SvelteKit one — see the module docstring.
DEFAULT_API_URL = "https://boltzlabs.cloud"

DOTENV_NAME = ".env"

# Parsed once per path. A training loop constructs pools and clients freely, and
# none of them should re-read the filesystem.
_cache = {}


def find_dotenv(start=None, name=DOTENV_NAME):
    """Nearest `.env` at or above ``start``. ``None`` if there isn't one.

    Walking upwards is what makes `python experiments/train.py` work from a
    repository root as well as from inside the directory, without a copy of the
    file in both.
    """
    d = os.path.abspath(start or os.getcwd())
    while True:
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _parse(path):
    """A small dotenv parser: KEY=VALUE, `export` prefixes, # comments, quotes.

    Deliberately not a dependency. This is thirty lines of well-understood
    parsing, and an RL SDK asking a user to install a package to read a two-line
    file would be a worse trade than owning it.
    """
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return out

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if value[:1] in ("'", '"'):
            # A quoted value ends at its closing quote, and anything after it is
            # a comment. Testing the last character instead would mis-read
            # `KEY="v"  # note` as unquoted and hand back the quotes as part of
            # the secret — which is exactly the kind of bug that shows up as a
            # 401 with a key that looks correct in the file.
            close = value.find(value[0], 1)
            value = value[1:] if close == -1 else value[1:close]
        elif "#" in value:
            # An unquoted trailing comment.
            value = value.split("#", 1)[0].strip()
        if key:
            out[key] = value
    return out


def load_dotenv(path=None, start=None):
    """Return the nearest `.env` as a dict. Never touches ``os.environ``.

    Leaving the process environment alone matters: a library that quietly
    exported a caller's file would change the behaviour of every other library
    in the process, and of any subprocess it spawns.
    """
    path = path or find_dotenv(start)
    if not path:
        return {}
    if path not in _cache:
        _cache[path] = _parse(path)
    return _cache[path]


def get(name, default=None, dotenv_path=None):
    """One setting: real environment first, then `.env`, then the default."""
    value = os.environ.get(name)
    if value:
        return value
    value = load_dotenv(dotenv_path).get(name)
    return value if value else default


def resolve(url=None, api_key=None, require_key=True, dotenv_path=None):
    """Return ``(url, api_key)`` following the order in the module docstring."""
    url = url or get("BOLTZLABS_API_URL", dotenv_path=dotenv_path) or DEFAULT_API_URL
    api_key = api_key or get("BOLTZLABS_API_KEY", default="", dotenv_path=dotenv_path)

    if require_key and not api_key:
        where = find_dotenv() or "a .env file"
        raise AuthError(
            401,
            f"no API key: set BOLTZLABS_API_KEY in the environment or in {where}, "
            f"or pass api_key=",
        )
    return url.rstrip("/"), api_key


def mask(api_key):
    """Enough of a key to tell which one is in use, and no more."""
    if len(api_key) <= 8:
        return "••••"
    return api_key[:12] + "••••" + api_key[-4:]
