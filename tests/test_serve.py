"""serve() over real pipes — the same channel the worker uses.

Driven as a subprocess rather than by calling serve() in-process, because half of
what serve() is for happens at the file-descriptor level: taking fd 1 away from
the program so a stray print cannot corrupt the protocol. Monkeypatching
sys.stdout in-process would test a different thing than the one that ships.
"""

import json
import os
import subprocess
import sys

import pytest

SDK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Env:
    """A serve() loop in a subprocess, driven one line at a time."""

    def __init__(self, body):
        script = f"import sys\nsys.path.insert(0, {SDK!r})\n" + body
        self._err = None
        self.proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, "environment closed the channel: " + self.stderr()
        return json.loads(line)

    def stderr(self):
        self.close()
        return self._err or ""

    def close(self):
        # communicate() closes stdin itself, which is the EOF that ends serve()'s
        # loop. Closing it here first would leave communicate() flushing a closed
        # file — tolerated on 3.14, a ValueError on 3.13.
        if self._err is not None:
            return
        try:
            _, self._err = self.proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            _, self._err = self.proc.communicate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


BASIC = """
from boltzlabs.env import serve
t = {"n": 0}
def reset(seed):
    t["n"] = 0
    return {"t": 0, "seed": seed}
def step(action):
    t["n"] += 1
    return {"t": t["n"]}, float(action), t["n"] >= 3, {"a": action}
serve(reset=reset, step=step)
"""


def test_reset_and_step():
    with Env(BASIC) as env:
        assert env.send({"op": "reset", "seed": 7}) == {"obs": {"t": 0, "seed": 7}}
        r = env.send({"op": "step", "action": 2})
        assert r["obs"] == {"t": 1} and r["reward"] == 2.0 and r["done"] is False
        assert r["info"] == {"a": 2}
        env.send({"op": "step", "action": 1})
        assert env.send({"op": "step", "action": 1})["done"] is True


def test_print_does_not_corrupt_the_channel():
    """The trap this helper exists for.

    A user's print() lands on the protocol channel in a hand-rolled loop and
    every reply after it is read as an answer to the previous message. Here it
    has to reach stderr instead, and the replies have to stay aligned.
    """
    body = """
import subprocess, sys
from boltzlabs.env import serve
def reset(seed):
    print("hello from reset")
    return {"t": 0}
def step(action):
    print("step", action)
    sys.stderr.write("to stderr\\n")
    # A child process inheriting fd 1 — the case that survives a sys.stdout swap
    # but not a dup2, which is why serve() does the dup2.
    subprocess.run([sys.executable, "-c", "print('from a subprocess')"])
    return {"t": action}, 1.0, False, {}
serve(reset=reset, step=step)
"""
    with Env(body) as env:
        assert env.send({"op": "reset", "seed": None}) == {"obs": {"t": 0}}
        for i in range(3):
            assert env.send({"op": "step", "action": i})["obs"] == {"t": i}
        err = env.stderr()
    assert "hello from reset" in err
    assert "from a subprocess" in err


def test_exception_is_answered_not_fatal():
    body = """
from boltzlabs.env import serve
def reset(seed): return {"t": 0}
def step(action):
    if action == "boom":
        raise ValueError("kaboom")
    return {"t": action}, 0.0, False, {}
serve(reset=reset, step=step)
"""
    with Env(body) as env:
        env.send({"op": "reset", "seed": None})
        r = env.send({"op": "step", "action": "boom"})
        assert r["done"] is True and "kaboom" in r["info"]["boltzlabs_error"]
        # Still alive: one bad step must not cost the environment.
        assert env.send({"op": "step", "action": 5})["obs"] == {"t": 5}


def test_unknown_op_and_bad_line():
    with Env(BASIC) as env:
        r = env.send({"op": "teleport"})
        assert "unknown op" in r["info"]["boltzlabs_error"]
        env.proc.stdin.write("this is not json\n")
        env.proc.stdin.flush()
        r = json.loads(env.proc.stdout.readline())
        assert "boltzlabs_error" in r["info"]
        assert env.send({"op": "reset", "seed": 1})["obs"]["t"] == 0


def test_gymnasium_five_tuple_keeps_truncation():
    body = """
from boltzlabs.env import serve
def reset(seed): return [0.0, 0.0]
def step(action):
    return [1.0, 2.0], 0.5, action == "term", action == "trunc", {"k": 1}
serve(reset=reset, step=step)
"""
    with Env(body) as env:
        env.send({"op": "reset", "seed": None})
        r = env.send({"op": "step", "action": "trunc"})
        assert r["done"] is True
        assert r["info"]["truncated"] is True and r["info"]["terminated"] is False
        r = env.send({"op": "step", "action": "term"})
        assert r["info"]["terminated"] is True and r["info"]["truncated"] is False


def test_gymnasium_reset_tuple_is_unwrapped():
    body = """
from boltzlabs.env import serve
def reset(seed): return {"t": 0}, {"info": "ignored"}
def step(action): return {"t": 1}, 0.0, False, {}
serve(reset=reset, step=step)
"""
    with Env(body) as env:
        assert env.send({"op": "reset", "seed": None}) == {"obs": {"t": 0}}


def test_three_tuple_and_dict_returns():
    body = """
from boltzlabs.env import serve
def reset(seed): return 0
def step(action):
    if action == "dict":
        return {"obs": 9, "reward": 3.0, "done": True}
    return action, 1.0, False
serve(reset=reset, step=step)
"""
    with Env(body) as env:
        env.send({"op": "reset", "seed": None})
        assert env.send({"op": "step", "action": 4})["obs"] == 4
        r = env.send({"op": "step", "action": "dict"})
        assert r["obs"] == 9 and r["reward"] == 3.0 and r["done"] is True


def test_numpy_observations_are_serialised():
    pytest.importorskip("numpy")
    body = """
import numpy as np
from boltzlabs.env import serve
def reset(seed): return np.zeros(3, dtype=np.float32)
def step(action): return np.arange(3, dtype=np.float32), np.float32(1.5), False, {}
serve(reset=reset, step=step)
"""
    with Env(body) as env:
        assert env.send({"op": "reset", "seed": None})["obs"] == [0.0, 0.0, 0.0]
        r = env.send({"op": "step", "action": 0})
        assert r["obs"] == [0.0, 1.0, 2.0] and r["reward"] == 1.5
