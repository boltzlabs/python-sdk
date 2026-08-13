"""Turn a directory on the caller's disk into the tar the worker unpacks.

Two decisions worth stating.

**The archive is deterministic.** Timestamps, ownership and walk order are all
pinned, so the same directory produces the same bytes every time. That makes an
upload diffable and cacheable, and it means "the pool I benchmarked" and "the
pool I deployed" can be compared by hash rather than by hope.

**``boltzlabs/env.py`` is vendored in automatically.** The sandbox mounts the
uploaded directory at ``/env`` and nothing else — no site-packages, no pip. So
``from boltzlabs.env import serve`` at the top of a user's ``env.py`` can only work
if the module travels with it. Making the user copy a file by hand is a
documentation problem that becomes a support problem; putting it in the tar is
four lines here.
"""

import fnmatch
import gzip
import io
import os
import tarfile

__all__ = ["pack_env_dir", "DEFAULT_EXCLUDES"]

# The worker's limit is 32 MB and the control plane rejects above it. An
# environment is a program, not a dataset.
MAX_CODE_BYTES = 32 << 20

DEFAULT_EXCLUDES = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    ".DS_Store",
    "*.egg-info",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "bench_results",
)

_SHIM_INIT = '''"""Vendored by the boltzlabs SDK so `from boltzlabs.env import serve` works in the sandbox.

Deliberately not the installed package's __init__: that one imports RLPool, which
imports numpy and an HTTP client. Inside the sandbox there is neither, and there
is nothing to reach — the environment talks over stdio, not over the network.
"""

from .env import serve  # noqa: F401

__all__ = ["serve"]
'''.encode(
    "utf-8"
)


def _excluded(name, patterns):
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _env_module_source():
    """The bytes of this package's own env.py."""
    try:  # 3.9+: works from a wheel, a zip, or a source checkout
        from importlib.resources import files

        return files(__package__).joinpath("env.py").read_bytes()
    except Exception:  # noqa: BLE001 — fall back to the plain filesystem
        with open(os.path.join(os.path.dirname(__file__), "env.py"), "rb") as fh:
            return fh.read()


def _add_bytes(tf, name, data, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tf.addfile(info, io.BytesIO(data))


def pack_env_dir(
    env_dir,
    entrypoint="env.py",
    vendor=True,
    excludes=DEFAULT_EXCLUDES,
    max_bytes=MAX_CODE_BYTES,
):
    """Pack ``env_dir`` into gzipped tar bytes.

    Raises ``FileNotFoundError`` if the entrypoint is missing — a create that
    failed on the worker for that reason costs a round trip and returns an error
    about a path inside a sandbox the caller has never seen.
    """
    root = os.path.abspath(os.path.expanduser(env_dir))
    if not os.path.isdir(root):
        raise NotADirectoryError(f"{env_dir!r} is not a directory")
    if not os.path.isfile(os.path.join(root, entrypoint)):
        raise FileNotFoundError(
            f"{env_dir!r} has no {entrypoint!r} — that file is what the pool runs. "
            f"Pass entrypoint= if yours is named something else."
        )

    buf = io.BytesIO()
    names = set()
    skipped_links = []

    # mtime=0 on the gzip header too: otherwise the container carries a
    # timestamp even though everything inside it is pinned.
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if not _excluded(d, excludes))
                for fname in sorted(filenames):
                    if _excluded(fname, excludes):
                        continue
                    full = os.path.join(dirpath, fname)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    if os.path.islink(full):
                        # The worker skips symlinks when unpacking — a link to
                        # /etc/shadow would otherwise be mounted into every
                        # sandbox in the pool. Say so here rather than let the
                        # file quietly not exist at the far end.
                        skipped_links.append(rel)
                        continue
                    with open(full, "rb") as fh:
                        data = fh.read()
                    mode = os.stat(full).st_mode & 0o777
                    _add_bytes(tf, rel, data, mode)
                    names.add(rel)

            if vendor and "boltzlabs/env.py" not in names:
                _add_bytes(tf, "boltzlabs/__init__.py", _SHIM_INIT)
                _add_bytes(tf, "boltzlabs/env.py", _env_module_source())

    blob = buf.getvalue()
    if len(blob) > max_bytes:
        raise ValueError(
            f"{env_dir!r} packs to {len(blob)/1e6:.1f} MB, over the {max_bytes/1e6:.0f} MB limit. "
            f"An environment is a program, not a dataset — bake large assets into the image."
        )
    return blob, skipped_links
