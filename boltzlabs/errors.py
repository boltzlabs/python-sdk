"""Exceptions, separated by what the caller can do about them.

The distinction that matters to a training script is retry-or-not. A pool that
failed to create because ``env.py`` raises on import will fail again in a second;
one that failed because no RL worker is attached to the fleet may not. The
control plane already draws that line in its status codes (see
``statusForRLError`` in backend/internal/handler/rl.go) and this mirrors it, so a
caller can write ``except CapacityError: sleep and retry`` and mean it.
"""

__all__ = [
    "BoltzLabsError",
    "TransportError",
    "APIError",
    "AuthError",
    "NotFoundError",
    "QuotaError",
    "CapacityError",
    "PoolGoneError",
    "PayloadTooLargeError",
]


class BoltzLabsError(Exception):
    """Base for everything this package raises."""


class TransportError(BoltzLabsError):
    """The request never got an HTTP answer — DNS, connect, timeout, reset."""


class APIError(BoltzLabsError):
    """The server answered with an error status."""

    def __init__(self, status, message, body=None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.body = body


class AuthError(APIError):
    """401/403 — the API key is missing, wrong, or not allowed here."""


class NotFoundError(APIError):
    """404 — no such pool, or not yours. The control plane does not distinguish
    the two on purpose: telling a caller that someone else's pool id exists is
    itself a leak."""


class QuotaError(APIError):
    """409 — the account's environment limit is in the way."""


class CapacityError(APIError):
    """503 — no worker could take this. Retryable, unlike everything else."""


class PoolGoneError(APIError):
    """410 — the worker holding this pool left the fleet. The environments are
    gone with it; a new pool has to be created."""


class PayloadTooLargeError(APIError):
    """413 — the uploaded environment directory is over the limit."""


def from_status(status, message, body=None):
    """Map an HTTP status onto the class a caller would branch on."""
    cls = {
        401: AuthError,
        403: AuthError,
        404: NotFoundError,
        409: QuotaError,
        410: PoolGoneError,
        413: PayloadTooLargeError,
        502: CapacityError,
        503: CapacityError,
        504: CapacityError,
    }.get(status, APIError)
    return cls(status, message, body)
