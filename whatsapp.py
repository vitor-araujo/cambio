"""
WhatsApp — sends rate-spike alerts via Twilio's WhatsApp API.

Zero external dependencies: just stdlib urllib + a tiny .env loader.
Configure with `python configure.py` (writes .env, gitignored).

State (last alert timestamp + anchor rate) persists in .fx_whatsapp.state
so cooldown survives across watch-loop restarts.
"""

import base64
import json
import os
import urllib.parse
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENV_PATH = ".env"
STATE_PATH = ".fx_whatsapp.state"
TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
DEFAULT_THRESHOLD_PCT = 1.0
DEFAULT_COOLDOWN_MIN = 60


# ── env file loader (no python-dotenv dependency) ────────────────────────────
def load_env(path: str = ENV_PATH) -> bool:
    """Populate os.environ from a .env file. Returns True if file was found."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    return True


# ── config introspection ─────────────────────────────────────────────────────
def is_configured() -> bool:
    return all(
        os.getenv(k)
        for k in (
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_FROM",
            "WHATSAPP_TO",
        )
    )


def missing_keys() -> list[str]:
    return [
        k
        for k in (
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_FROM",
            "WHATSAPP_TO",
        )
        if not os.getenv(k)
    ]


# ── persistent state ─────────────────────────────────────────────────────────
def load_state(path: str = STATE_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict, path: str = STATE_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


# ── twilio send ──────────────────────────────────────────────────────────────
def send(message: str) -> tuple[bool, str]:
    """Send a WhatsApp message via Twilio. Returns (ok, status_or_error)."""
    if not is_configured():
        return False, f"missing env vars: {', '.join(missing_keys())}"

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
            "User-Agent": "cambio/0.4.0",
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
    """Ensure phone is in E.164 (+<digits>) — strip everything else."""
    p = p.strip()
    digits = "".join(c for c in p if c.isdigit())
    return f"+{digits}" if digits else p


# ── rate-spike alert (the public entry point used by --watch) ────────────────
def maybe_alert_on_rise(
    rate_now: float,
    *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    cooldown_min: int = DEFAULT_COOLDOWN_MIN,
    state_path: str = STATE_PATH,
) -> dict:
    """
    Fire a WhatsApp alert if rate_now rose >= threshold_pct vs the anchor.

    Anchor logic:
      - first call ever: anchor = rate_now (no alert)
      - subsequent calls: alert when delta >= threshold AND cooldown elapsed
      - on alert: anchor moves to rate_now, last_alert_ts = now
      - if rate falls below anchor: anchor moves down (tracks the local low)
        so the next +X% rally fires from the floor, not from an old high

    Returns a dict: {fired, reason, delta_pct, anchor, sent_ok, error}
    """
    state = load_state(state_path)
    anchor = state.get("anchor_rate")
    last_ts_str = state.get("last_alert_ts")

    out = {
        "fired": False,
        "reason": "",
        "delta_pct": 0.0,
        "anchor": anchor,
        "sent_ok": False,
        "error": "",
    }

    # bootstrap: no anchor yet → set it and return
    if anchor is None or anchor <= 0:
        state["anchor_rate"] = rate_now
        save_state(state, state_path)
        out["reason"] = "anchor initialized"
        out["anchor"] = rate_now
        return out

    delta_pct = (rate_now / anchor - 1.0) * 100.0
    out["delta_pct"] = delta_pct

    # track the local low — never anchor above current
    if rate_now < anchor:
        state["anchor_rate"] = rate_now
        save_state(state, state_path)
        out["reason"] = f"anchor lowered to {rate_now:.4f}"
        out["anchor"] = rate_now
        return out

    if delta_pct < threshold_pct:
        out["reason"] = f"delta {delta_pct:+.2f}% < {threshold_pct:.2f}%"
        return out

    # cooldown check
    if last_ts_str:
        try:
            last_ts = datetime.fromisoformat(last_ts_str)
            if datetime.now() - last_ts < timedelta(minutes=cooldown_min):
                remaining = cooldown_min - int(
                    (datetime.now() - last_ts).total_seconds() // 60
                )
                out["reason"] = f"cooldown ({remaining} min left)"
                return out
        except ValueError:
            pass

    # fire
    msg = _format_message(rate_now, anchor, delta_pct)
    ok, info = send(msg)
    out["sent_ok"] = ok
    out["fired"] = ok
    out["error"] = "" if ok else info

    if ok:
        state["anchor_rate"] = rate_now
        state["last_alert_ts"] = datetime.now().isoformat(timespec="seconds")
        state["last_rate"] = rate_now
        save_state(state, state_path)
        out["anchor"] = rate_now
        out["reason"] = "alert sent"
    else:
        out["reason"] = f"send failed: {info}"

    return out


def _format_message(rate_now: float, anchor: float, delta_pct: float) -> str:
    ts = datetime.now().strftime("%d/%m %H:%M")
    return (
        f"🇧🇷 cambio · USD/BRL {ts}\n\n"
        f"R$ {rate_now:.4f}  ({delta_pct:+.2f}% vs R$ {anchor:.4f})\n\n"
        f"Janela de oportunidade — confere o sinal antes de converter.\n"
        f"https://higlobe.com"
    )


def reset_state(state_path: str = STATE_PATH) -> None:
    """Wipe the anchor + cooldown — useful after a manual exchange."""
    if os.path.exists(state_path):
        os.remove(state_path)
