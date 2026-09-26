"""File transfer to and from a sandbox — the SDK half of ``boltz cp``.

The wire format is a tar stream, which is what lets one request carry a whole
tree with its permissions and layout intact, and it is what the ``/fs`` endpoint
speaks on both sides. It rides an exec rather than a socket, so it works on a
sandbox created with internet off.
"""

import io
import os
import tarfile
from urllib.parse import quote


def push(client, sandbox_id, local, remote="/workspace", timeout=300.0):
    """Upload a file or directory into ``remote`` inside the sandbox."""
    local = os.path.abspath(local)
    if not os.path.exists(local):
        raise FileNotFoundError(local)

    buf = io.BytesIO()
    # The archive is named relative to the parent, so uploading ./src lands as
    # <remote>/src rather than spilling its contents across <remote>.
    base = os.path.dirname(local.rstrip(os.sep))
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(local, arcname=os.path.relpath(local, base))

    client._session.raw(
        "POST",
        f"/api/sandboxes/{sandbox_id}/fs?path={quote(remote)}",
        buf.getvalue(),
        content_type="application/x-tar",
        timeout=timeout,
    )
    return remote


def pull(client, sandbox_id, remote, local=".", timeout=300.0):
    """Download ``remote`` out of the sandbox, extracting it under ``local``."""
    blob = client._session.raw(
        "GET",
        f"/api/sandboxes/{sandbox_id}/fs?path={quote(remote)}",
        timeout=timeout,
    )
    os.makedirs(local, exist_ok=True)
    _extract(io.BytesIO(blob), local)
    return os.path.abspath(local)


def _extract(fileobj, dest):
    """Unpack a tar under ``dest``.

    The archive is not trusted input — a sandbox may be running code its owner
    did not write — so every member is checked to land inside ``dest``. Without
    that, a member named ``../../.ssh/authorized_keys`` writes outside the
    directory the caller pointed at. Links are refused for the same reason: a
    symlink out of the tree followed by a write through it is the same escape in
    two steps.
    """
    dest = os.path.abspath(dest)
    with tarfile.open(fileobj=fileobj, mode="r:*") as tf:
        for member in tf:
            target = os.path.abspath(os.path.join(dest, member.name))
            if target != dest and not target.startswith(dest + os.sep):
                raise ValueError(f"refusing archive entry outside {dest}: {member.name!r}")
            if member.issym() or member.islnk():
                continue
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            if not member.isfile():
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            os.chmod(target, member.mode & 0o777)
