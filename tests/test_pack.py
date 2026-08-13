"""What goes into the tar, and what stays out."""

import gzip
import io
import os
import tarfile

import pytest

from boltzlabs._pack import pack_env_dir


def names(blob):
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as tf:
        return sorted(tf.getnames())


def write(d, rel, text):
    path = os.path.join(str(d), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def test_vendors_the_env_module(tmp_path):
    write(tmp_path, "env.py", "from boltzlabs.env import serve\n")
    blob, _ = pack_env_dir(str(tmp_path))
    assert "boltzlabs/env.py" in names(blob)
    assert "boltzlabs/__init__.py" in names(blob)

    # And the vendored module is the real one, not a stub of it.
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as tf:
        src = tf.extractfile("boltzlabs/env.py").read().decode()
    assert "def serve(" in src


def test_does_not_clobber_a_users_own_boltzlabs(tmp_path):
    write(tmp_path, "env.py", "x = 1\n")
    write(tmp_path, "boltzlabs/env.py", "MINE = True\n")
    blob, _ = pack_env_dir(str(tmp_path))
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as tf:
        assert tf.extractfile("boltzlabs/env.py").read() == b"MINE = True\n"


def test_excludes_junk(tmp_path):
    write(tmp_path, "env.py", "x = 1\n")
    write(tmp_path, "__pycache__/env.cpython-311.pyc", "junk")
    write(tmp_path, ".git/config", "junk")
    write(tmp_path, "node_modules/left-pad/index.js", "junk")
    write(tmp_path, "helper.py", "y = 2\n")

    got = names(pack_env_dir(str(tmp_path))[0])
    assert "env.py" in got and "helper.py" in got
    assert not [n for n in got if "__pycache__" in n or ".git" in n or "node_modules" in n]


def test_is_deterministic(tmp_path):
    write(tmp_path, "env.py", "x = 1\n")
    write(tmp_path, "pkg/mod.py", "y = 2\n")
    first, _ = pack_env_dir(str(tmp_path))
    # Touch the files: a timestamp must not change the archive, or two uploads of
    # the same code would be two different artefacts.
    os.utime(os.path.join(str(tmp_path), "env.py"), (10_000, 10_000))
    second, _ = pack_env_dir(str(tmp_path))
    assert first == second


def test_missing_entrypoint_fails_before_the_upload(tmp_path):
    write(tmp_path, "main.py", "x = 1\n")
    with pytest.raises(FileNotFoundError) as exc:
        pack_env_dir(str(tmp_path))
    assert "env.py" in str(exc.value)
    # Named alternative works
    blob, _ = pack_env_dir(str(tmp_path), entrypoint="main.py")
    assert "main.py" in names(blob)


def test_symlinks_are_reported_not_shipped(tmp_path):
    write(tmp_path, "env.py", "x = 1\n")
    os.symlink("/etc/hosts", os.path.join(str(tmp_path), "secrets"))
    blob, skipped = pack_env_dir(str(tmp_path))
    assert "secrets" not in names(blob)
    assert skipped == ["secrets"]


def test_size_limit(tmp_path):
    write(tmp_path, "env.py", "x = 1\n")
    # Incompressible, so the gzip does not hide it.
    with open(os.path.join(str(tmp_path), "blob.bin"), "wb") as fh:
        fh.write(os.urandom(200_000))
    with pytest.raises(ValueError) as exc:
        pack_env_dir(str(tmp_path), max_bytes=100_000)
    assert "not a dataset" in str(exc.value)
