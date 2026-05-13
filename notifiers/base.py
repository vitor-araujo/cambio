"""
Notifier protocol — the abstraction every messaging backend implements.

Keep this interface tiny (ISP). New providers only need `send`, plus the
two helpers that let configure.py and rate_alert.py introspect config state.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Notifier(Protocol):
    """A pluggable backend for short text alerts.

    Implementations live in `notifiers/<provider>.py` and register themselves
    in the `_REGISTRY` of `notifiers/__init__.py`.
    """

    name: str
    """Stable provider id (e.g. 'telegram', 'whatsapp')."""

    def is_configured(self) -> bool:
        """True iff every required env var is set."""
        ...

    def missing_keys(self) -> list[str]:
        """Names of required env vars that are absent."""
        ...

    def send(self, message: str) -> tuple[bool, str]:
        """Send a plain-text message. Returns (ok, status_or_error)."""
        ...
