"""
Pluggable messaging providers for cambio rate alerts.

Add a new provider in three steps (OCP — no existing code is modified):
  1. Implement the `Notifier` protocol in `notifiers/<name>.py`
  2. Register it in `_REGISTRY` below
  3. Add a setup branch in `configure.py`
"""

import os

from .base import Notifier
from .telegram import TelegramNotifier
from .whatsapp import WhatsAppNotifier

DEFAULT_PROVIDER = "telegram"
ENV_VAR = "NOTIFIER_PROVIDER"

_REGISTRY: dict[str, type] = {
    "telegram": TelegramNotifier,
    "whatsapp": WhatsAppNotifier,
}


def get_notifier(provider: str | None = None) -> Notifier:
    """Return a `Notifier` instance for the named provider.

    Resolution order: explicit arg → env var → DEFAULT_PROVIDER.
    """
    name = (provider or os.getenv(ENV_VAR) or DEFAULT_PROVIDER).lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown provider '{name}'. options: {available()}")
    return cls()


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["Notifier", "get_notifier", "available", "DEFAULT_PROVIDER", "ENV_VAR"]
