"""
WhatsApp notifier via Twilio's REST API.

Kept as a working backend for the day a real WhatsApp channel becomes
viable (Twilio paid, Meta Cloud API, or a BSP). All Twilio specifics are
isolated here — switching to Meta Cloud API would mean replacing this file
and registering the new class in `notifiers/__init__.py`. Nothing else
in the codebase changes.
"""

import base64
import os
import urllib.parse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
REQUIRED_KEYS = (
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_FROM",
    "WHATSAPP_TO",
)


class WhatsAppNotifier:
    """Sends WhatsApp messages via Twilio. Requires a Twilio account."""

    name = "whatsapp"

    def is_configured(self) -> bool:
        return all(os.getenv(k) for k in REQUIRED_KEYS)

    def missing_keys(self) -> list[str]:
        return [k for k in REQUIRED_KEYS if not os.getenv(k)]

    def send(self, message: str) -> tuple[bool, str]:
        if not self.is_configured():
            return False, f"missing env vars: {', '.join(self.missing_keys())}"

        sid = os.environ["TWILIO_ACCOUNT_SID"]
        tok = os.environ["TWILIO_AUTH_TOKEN"]
        src = _normalize_phone(os.environ["TWILIO_FROM"])
        dst = _normalize_phone(os.environ["WHATSAPP_TO"])

        payload = urllib.parse.urlencode(
            {"From": f"whatsapp:{src}", "To": f"whatsapp:{dst}", "Body": message}
        ).encode()
        creds = base64.b64encode(f"{sid}:{tok}".encode()).decode()
        req = Request(
            TWILIO_API.format(sid=sid),
            data=payload,
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "cambio/0.4.1",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                return resp.status in (200, 201), f"twilio {resp.status}"
        except HTTPError as e:
            body = e.read().decode(errors="ignore")[:200]
            return False, f"twilio HTTP {e.code}: {body}"
        except URLError as e:
            return False, f"network: {e.reason}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


def _normalize_phone(p: str) -> str:
    """Force E.164 (+<digits>) — strip everything else."""
    digits = "".join(c for c in p.strip() if c.isdigit())
    return f"+{digits}" if digits else p.strip()
