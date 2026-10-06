"""Creation protocol tests without starting a worker."""
import pytest
import boltzlabs.pool as module
from boltzlabs.errors import APIError, TransportError


class Session:
    replies = []
    calls = []

    def __init__(self, *args, **kwargs):
        self.closed = False

    def call(self, method, path, body=None, timeout=None):
        self.calls.append((method, path, timeout))
        return (None if method == "DELETE" else self.replies.pop(0)), 1

    def close(self):
        self.closed = True


@pytest.fixture
def transport(monkeypatch):
    Session.calls = []
    Session.replies = []
    monkeypatch.setattr(module, "Session", Session)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return Session


def test_creation_polls_until_ready(transport):
    transport.replies = [
        {"pool_id": "rp-1", "status": "creating"},
        {"pool_id": "rp-1", "status": "creating"},
        {"pool_id": "rp-1", "status": "running", "n": 2, "ready": 2, "spawn_ms": {"p50": 12}},
    ]
    pool = module.RLPool(environment="cartpole", n=2, url="https://boltzlabs.cloud", api_key="k")
    assert (pool.pool_id, pool.ready, pool.spawn_ms) == ("rp-1", 2, {"p50": 12})
    assert [(method, path) for method, path, _ in transport.calls] == [
        ("POST", "/api/rl/pools?wait=false"),
        ("GET", "/api/rl/pools/rp-1?creation=true"),
        ("GET", "/api/rl/pools/rp-1?creation=true"),
    ]
    assert all(timeout <= 60 for _, _, timeout in transport.calls)


def test_startup_failure_keeps_error_and_cancels(transport):
    transport.replies = [{"pool_id": "rp-1", "status": "creating"},
                         {"status": "failed", "error": "env.py failed", "error_status": 400}]
    with pytest.raises(APIError, match="env.py failed") as failure:
        module.RLPool(environment="cartpole", n=1, url="https://boltzlabs.cloud", api_key="k")
    assert failure.value.status == 400
    assert transport.calls[-1][:2] == ("DELETE", "/api/rl/pools/rp-1")


def test_creation_timeout_cancels(transport, monkeypatch):
    transport.replies = [{"pool_id": "rp-1", "status": "creating"}]
    ticks = iter([0, 2])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TransportError, match="creation timed out"):
        module.RLPool(environment="cartpole", n=1, url="https://boltzlabs.cloud", api_key="k", create_timeout=1)
    assert transport.calls[-1][:2] == ("DELETE", "/api/rl/pools/rp-1")


def test_internal_worker_creation_remains_synchronous(transport):
    transport.replies = [{"pool_id": "rp-1", "n": 1, "ready": 1}]
    pool = module.RLPool(environment="cartpole", n=1, direct="http://localhost:9877")
    assert transport.calls == [("POST", "/worker/rl/pools", 900.0)]
    pool.close()
