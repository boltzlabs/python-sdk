"""The extraction guard. A sandbox may be running code its owner did not write,
so its archive is untrusted input."""

import io
import os
import tarfile

import pytest

from boltzlabs._files import _extract


def _tar(name, body=b"pwned"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(body)
        info.mode = 0o644
        tf.addfile(info, io.BytesIO(body))
    buf.seek(0)
    return buf


@pytest.mark.parametrize(
    "name", ["../escaped.txt", "../../etc/passwd", "sub/../../escaped.txt"]
)
def test_refuses_entries_that_escape_the_destination(tmp_path, name):
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="outside"):
        _extract(_tar(name), str(dest))
    # And nothing may have been written above it either.
    assert not (tmp_path / "escaped.txt").exists()


def test_extracts_ordinary_entries(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        d = tarfile.TarInfo("app")
        d.type = tarfile.DIRTYPE
        d.mode = 0o755
        tf.addfile(d)
        body = b"hello"
        f = tarfile.TarInfo("app/index.js")
        f.size = len(body)
        f.mode = 0o644
        tf.addfile(f, io.BytesIO(body))
    buf.seek(0)

    _extract(buf, str(tmp_path))
    assert (tmp_path / "app" / "index.js").read_bytes() == b"hello"


def test_skips_links_rather_than_following_them(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    buf.seek(0)

    _extract(buf, str(tmp_path))
    assert not os.path.lexists(tmp_path / "escape")
