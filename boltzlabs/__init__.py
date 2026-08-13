"""boltzlabs — sandboxes and RL environments, from Python.

    from boltzlabs import Sandbox

    sb = Sandbox()                            # small / base / internet off

    print(sb.run("print(sum(range(101)))"))   # code
    print(sb.exec("pip install requests"))    # shell
    sb.terminal()                             # interactive shell

    sb.delete()                               # stops the meter

    from boltzlabs import RLPool

    with RLPool("./my_env", 1000) as pool:
        obs = pool.reset()
        obs, rewards, dones, infos = pool.step(actions)   # one request, 1000 envs

The key comes from ``BOLTZLABS_API_KEY`` in the environment or in a ``.env``.
Nothing else is required.

Everything else is a detail you can look up when you need it: ``boltzlabs.me()``,
``boltzlabs.sandboxes()``, ``boltzlabs.sandbox(id)``, ``boltzlabs.environments()``,
``boltzlabs.machines()``, ``boltzlabs.use()`` to point at another key or origin, and
``Client`` when you want to hold two.
"""

from .client import (
    APIKey,
    Client,
    Environment,
    ExecResult,
    Language,
    Machine,
    Sandbox,
    _default_client,
    use,
)
from .env import serve
from .errors import (
    APIError,
    AuthError,
    BoltzLabsError,
    CapacityError,
    NotFoundError,
    PayloadTooLargeError,
    PoolGoneError,
    QuotaError,
    TransportError,
)
from .pool import RLPool, Timing

__version__ = "0.1.0"

__all__ = [
    "Sandbox",
    "RLPool",
    "serve",
    "Client",
    "use",
    "me",
    "execute",
    "sandbox",
    "sandboxes",
    "environments",
    "machines",
    "languages",
    "ExecResult",
    "Environment",
    "Machine",
    "Language",
    "APIKey",
    "Timing",
    "BoltzLabsVecEnv",
    "BoltzLabsError",
    "TransportError",
    "APIError",
    "AuthError",
    "NotFoundError",
    "QuotaError",
    "CapacityError",
    "PoolGoneError",
    "PayloadTooLargeError",
    "__version__",
]


# The listing calls, without making anyone build a Client first. They share one
# lazily-built default client, so importing boltzlabs still needs no key and opens
# no socket. `_default_client()` is called inside each rather than held in a
# module attribute, because `boltzlabs.client` is already the submodule.
def me():
    """Who your key belongs to. `bzlabs auth status`."""
    return _default_client().me()


def sandboxes():
    """Your sandboxes. `bzlabs ls`."""
    return _default_client().sandboxes()


def execute(code=None, *, language=None, file=None, timeout=None, filename=None):
    """Run one piece of code and get back what it printed. `bzlabs run`.

        boltzlabs.execute("print(sum(range(101)))", language="python")
        boltzlabs.execute(file="train.py", language="python")

    The language is always named — see ``boltzlabs.languages()`` for the codes.
    """
    return _default_client().execute(
        code, language=language, file=file, timeout=timeout, filename=filename
    )


def languages():
    """The language codes execution accepts. `bzlabs languages`."""
    return _default_client().languages()


def sandbox(id):
    """One sandbox by id. `bzlabs status <id>`.

    The counterpart to constructing one: ``Sandbox(...)`` creates, this reaches
    something the platform already assigned an id to.
    """
    return _default_client().sandbox(id)


def environments():
    """What a sandbox can ship with. `bzlabs environments`."""
    return _default_client().environments()


def machines():
    """Machines and prices. `bzlabs machines`."""
    return _default_client().machines()


def rl_environments():
    """The ready-made RL environments a pool can be launched with.

    Each is the same JSON-lines protocol in the same sandbox as your own code —
    the only difference is that it was already on the box::

        RLPool(environment="cartpole", n=64)
    """
    return _default_client().rl_environments()


def __getattr__(name):
    # BoltzLabsVecEnv reaches for gymnasium, and importing this package must not
    # require it.
    if name == "BoltzLabsVecEnv":
        from .vec import BoltzLabsVecEnv

        return BoltzLabsVecEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
