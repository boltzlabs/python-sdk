"""The loop that runs *inside* the sandbox: one JSON line in, one JSON line out.

This module is deliberately dependency-free and self-contained. The SDK vendors
it into the tar it uploads (see ``_pack.py``), so it has to import cleanly on a
box that has nothing but a Python interpreter — no numpy, no boltzlabs, no pip.

    from boltzlabs.env import serve

    def reset(seed):  return {"t": 0}
    def step(action): return obs, reward, done, info

    serve(reset=reset, step=step)

``serve`` exists to own the three things that silently break the channel:

1. **Buffering.** Without a flush the interpreter holds the reply in its stdout
   buffer and a 0.2 ms step becomes seconds, or a whole episode of them.
2. **Stray output.** The protocol *is* stdout. A single ``print("debug")`` — or a
   C library's ``printf``, or a subprocess inheriting fd 1 — desynchronises it
   and every subsequent reply is read as an answer to the previous message.
   ``serve`` takes a private duplicate of fd 1 and points the public one at
   stderr, so ordinary printing keeps working and lands in the environment's log
   instead of the wire.
3. **Exceptions.** A traceback out of user code would kill the environment; the
   worker would report it as a straggler and the pool would quietly shrink.
   Instead the step is answered with ``done=true`` and the error in ``info``, so
   a broken environment is visible in the training loop's own data.
"""

import base64
import json
import os
import sys
import traceback

__all__ = ["serve", "ProtocolError"]


class ProtocolError(Exception):
    """Raised for a message this side cannot make sense of."""


# ---------------------------------------------------------------------------
# the channel
# ---------------------------------------------------------------------------


def _take_channel():
    """Take stdout away from the program and hand back a private handle to it.

    The dup2 is what makes this airtight. Reassigning ``sys.stdout`` alone only
    covers Python-level ``print``; a native extension writing to fd 1, or a
    ``subprocess.run`` inheriting it, would still land on the protocol channel.
    After this call fd 1 *is* stderr for everyone except the returned handle.
    """
    try:
        saved = os.dup(1)
        os.dup2(2, 1)
        chan = os.fdopen(saved, "w", encoding="utf-8", newline="\n")
    except (OSError, ValueError):
        # No real fds (a test harness, an embedded interpreter). Fall back to the
        # Python-level swap, which is still better than nothing.
        chan = sys.stdout
    sys.stdout = sys.stderr
    return chan


def _jsonable(o):
    """Last-resort coercion for values ``json`` does not know.

    Written against duck types rather than imports: numpy is the common case but
    this module must not require it, and the same two attributes cover torch
    tensors and anything else array-shaped.
    """
    if hasattr(o, "tolist"):  # ndarray, torch.Tensor, numpy scalar
        return o.tolist()
    if hasattr(o, "item"):  # 0-d array, numpy scalar
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, (bytes, bytearray)):
        return base64.b64encode(bytes(o)).decode("ascii")
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    raise TypeError(f"{type(o).__name__} is not JSON-serialisable")


# ---------------------------------------------------------------------------
# reply shaping
# ---------------------------------------------------------------------------


def _reset_obs(ret):
    """Unwrap what a reset returned.

    Gymnasium's ``reset`` returns ``(obs, info)``; plenty of hand-written
    environments return the observation alone. Both are accepted. The pool's
    reset carries no info field, so an info dict here is dropped — if your
    observation is genuinely a pair whose second element is a dict, return it as
    a list so it is not mistaken for the Gymnasium shape.
    """
    if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
        return ret[0]
    return ret


def _step_reply(ret):
    """Normalise every step return shape onto the wire object.

    Accepted, in the order they are tried:

    * ``dict`` with ``obs``/``reward``/``done``/``info`` — passed through
    * ``(obs, reward, terminated, truncated, info)`` — the Gymnasium 5-tuple
    * ``(obs, reward, done, info)`` — the classic 4-tuple
    * ``(obs, reward, done)`` / ``(obs, reward)``

    The 5-tuple collapses to a single ``done`` because the pool's wire format has
    one flag, but ``terminated`` and ``truncated`` are both preserved in ``info``
    — a trainer that bootstraps value estimates needs to tell the two apart, and
    losing that distinction here would be a silent correctness bug in the
    training loop rather than a visible one.
    """
    if isinstance(ret, dict) and ("obs" in ret or "observation" in ret):
        out = dict(ret)
        if "observation" in out:
            out["obs"] = out.pop("observation")
        out.setdefault("reward", 0.0)
        out.setdefault("done", False)
        out["reward"] = float(out["reward"])
        out["done"] = bool(out["done"])
        return out

    if not isinstance(ret, (tuple, list)):
        raise ProtocolError(
            "step() must return (obs, reward, done, info) or a dict with an "
            f"'obs' key, got {type(ret).__name__}"
        )

    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        info = dict(info) if isinstance(info, dict) else {"info": info}
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        return {
            "obs": obs,
            "reward": float(reward),
            "done": bool(terminated) or bool(truncated),
            "info": info,
        }
    if len(ret) == 4:
        obs, reward, done, info = ret
        return {"obs": obs, "reward": float(reward), "done": bool(done), "info": info}
    if len(ret) == 3:
        obs, reward, done = ret
        return {"obs": obs, "reward": float(reward), "done": bool(done)}
    if len(ret) == 2:
        obs, reward = ret
        return {"obs": obs, "reward": float(reward), "done": False}
    raise ProtocolError(f"step() returned a {len(ret)}-tuple; expected 2 to 5 elements")


def _error_reply(what, exc):
    """Answer a message that could not be served, without dying.

    ``done=true`` so a training loop ends the episode rather than carrying a
    poisoned state forward, and the traceback goes to stderr where it lands in
    the environment's log next to the pool id.
    """
    traceback.print_exc(file=sys.stderr)
    return {
        "obs": None,
        "reward": 0.0,
        "done": True,
        "info": {"boltzlabs_error": f"{what}: {type(exc).__name__}: {exc}"},
    }


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def serve(reset, step, stdin=None):
    """Run the environment loop until stdin closes.

    ``reset(seed)`` returns an observation. ``seed`` is ``None`` when the caller
    did not ask for one; seed it anyway if you want reproducible episodes.

    ``step(action)`` returns any of the shapes ``_step_reply`` accepts.

    Both are called on this thread, one message at a time. There is no
    concurrency to reason about inside an environment: the pool's parallelism is
    across environments, not within one.
    """
    channel = _take_channel()
    stdin = stdin if stdin is not None else sys.stdin
    started = False

    def emit(obj):
        # separators: no spaces. At a thousand environments the batch is the
        # payload, and the spaces are a measurable fraction of it.
        channel.write(json.dumps(obj, default=_jsonable, separators=(",", ":")) + "\n")
        channel.flush()

    while True:
        line = stdin.readline()
        if not line:  # EOF — the pool was destroyed, or this env was killed
            return
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
            if not isinstance(msg, dict):
                raise ProtocolError("message was not a JSON object")
        except Exception as exc:  # noqa: BLE001 — never let a bad line kill the env
            emit(_error_reply("bad message", exc))
            continue

        op = msg.get("op")

        if op == "reset":
            try:
                obs = _reset_obs(reset(msg.get("seed")))
                started = True
                emit({"obs": obs})
            except Exception as exc:  # noqa: BLE001
                emit(_error_reply("reset() raised", exc))

        elif op == "step":
            try:
                # The worker always resets at spawn, so this can only be reached
                # if a caller drove the protocol by hand. Reset rather than hand
                # the user's step() an uninitialised environment.
                if not started:
                    reset(None)
                    started = True
                emit(_step_reply(step(msg.get("action"))))
            except Exception as exc:  # noqa: BLE001
                emit(_error_reply("step() raised", exc))

        else:
            # Unknown op: answer, do not crash. A protocol that grew a message
            # this environment predates should degrade to one bad step, not to a
            # dead environment for the rest of the run.
            emit(
                {
                    "obs": None,
                    "reward": 0.0,
                    "done": False,
                    "info": {"boltzlabs_error": f"unknown op {op!r}"},
                }
            )
