"""
Telegram notifier — Bot API over HTTPS (stdlib only).

Setup is handled by configure.py: user creates a bot via @BotFather, pastes
the token, sends one message to the bot, and we auto-discover the chat_id
through getUpdates.
"""

import json
import os
import urllib.parse
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_BASE = "https://api.telegram.org/bot{token}"
REQUIRED_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


class TelegramNotifier:
    """Sends messages via the Telegram Bot API. Free, unlimited, official."""

    name = "telegram"

    def is_configured(self) -> bool:
        return all(os.getenv(k) for k in REQUIRED_KEYS)

    def missing_keys(self) -> list[str]:
        return [k for k in REQUIRED_KEYS if not os.getenv(k)]

    def send(self, message: str) -> tuple[bool, str]:
        if not self.is_configured():
            return False, f"missing env vars: {', '.join(self.missing_keys())}"

        token = os.environ["TELEGRAM_BOT_TOKEN"]
        chat_id = os.environ["TELEGRAM_CHAT_ID"]
        url = f"{API_BASE.format(token=token)}/sendMessage"

        payload = urllib.parse.urlencode(
            {"chat_id": chat_id, "text": message, "disable_web_page_preview": "false"}
        ).encode()
        req = Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "cambio/0.4.1",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("ok"):
                    return True, f"telegram {resp.status}"
                return False, f"telegram error: {data.get('description', 'unknown')}"
        except HTTPError as e:
            body = e.read().decode(errors="ignore")[:200]
            return False, f"telegram HTTP {e.code}: {body}"
        except URLError as e:
            return False, f"network: {e.reason}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


def discover_chat_id(token: str) -> Optional[str]:
    """Read the latest message the bot received and return its chat_id.

    Used by configure.py so the user never has to copy/paste the id manually:
    they just message the bot and we extract it. Returns None if no recent
    message exists or the API fails.
    """
    url = f"{API_BASE.format(token=token)}/getUpdates"
    try:
        with urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    if not data.get("ok"):
        return None
    results = data.get("result") or []
    if not results:
        return None
    last = results[-1]
    msg = last.get("message") or last.get("edited_message") or last.get("channel_post")
    if not msg:
        return None
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    return str(chat_id) if chat_id is not None else None
