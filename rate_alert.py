"""
Rate-spike alert orchestration — provider-agnostic.

Depends only on the `Notifier` Protocol (DIP), not on any concrete provider.
Holds the anchor/threshold/cooldown state machine, the message formatter,
and the .env loader.

State (day-anchor + last-alert timestamp) persists in `.fx_alert.state` so the
cooldown and intra-day reference survive watch-loop restarts.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

from notifiers import Notifier, get_notifier

ENV_PATH = ".env"
STATE_PATH = ".fx_alert.state"
DEFAULT_THRESHOLD_PCT = 1.0
DEFAULT_COOLDOWN_MIN = 5


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


def reset_state(path: str = STATE_PATH) -> None:
    """Wipe the anchor + cooldown — call after a manual exchange."""
    if os.path.exists(path):
        os.remove(path)


# ── public entry point ───────────────────────────────────────────────────────
def maybe_alert_on_rise(
    rate_now: float,
    *,
    notifier: Optional[Notifier] = None,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    cooldown_min: int = DEFAULT_COOLDOWN_MIN,
    state_path: str = STATE_PATH,
) -> dict:
    """
    Fire an alert if rate_now rose >= threshold_pct vs the day's anchor.

    Intra-day anchor logic:
      - first call of the day: day anchor = rate_now (no alert)
      - subsequent calls: alert fires when rate_now >= day_anchor × (1 + threshold_pct/100)
        AND cooldown has elapsed since the last alert.
      - day anchor is fixed per day — it NEVER resets after an alert,
        so intra-day accumulation is tracked correctly regardless of poll interval.
      - when the date changes, the day anchor resets to the new first rate.

    The `notifier` parameter is the only seam between this module and the
    concrete provider — pass any object satisfying the Notifier protocol.
    Defaults to the provider from NOTIFIER_PROVIDER env (or 'telegram').
    """
    state = load_state(state_path)
    today_str = datetime.now().strftime("%Y-%m-%d")
    day_anchor = state.get("day_anchor")
    day_anchor_date = state.get("day_anchor_date")
    last_ts_str = state.get("last_alert_ts")

    out = {
        "fired": False,
        "reason": "",
        "delta_pct": 0.0,
        "anchor": day_anchor,
        "sent_ok": False,
        "error": "",
        "provider": "",
    }

    # bootstrap or new day — reset the intra-day reference
    if day_anchor is None or day_anchor_date != today_str:
        state["day_anchor"] = rate_now
        state["day_anchor_date"] = today_str
        save_state(state, state_path)
        out.update({"reason": f"day anchor set to {rate_now:.4f}", "anchor": rate_now})
        return out

    delta_pct = (rate_now / day_anchor - 1.0) * 100.0
    out["delta_pct"] = delta_pct

    if delta_pct < threshold_pct:
        out["reason"] = f"delta {delta_pct:+.2f}% < {threshold_pct:.2f}%"
        return out

    # cooldown
    if last_ts_str:
        try:
            last_ts = datetime.fromisoformat(last_ts_str)
            elapsed = datetime.now() - last_ts
            if elapsed < timedelta(minutes=cooldown_min):
                remaining = cooldown_min - int(elapsed.total_seconds() // 60)
                out["reason"] = f"cooldown ({remaining} min left)"
                return out
        except ValueError:
            pass

    # fire — day anchor stays fixed so we keep tracking intra-day growth
    n = notifier or get_notifier()
    out["provider"] = n.name
    ok, info = n.send(_format_message(rate_now, day_anchor, delta_pct))
    out["sent_ok"] = ok
    out["fired"] = ok

    if ok:
        state["last_alert_ts"] = datetime.now().isoformat(timespec="seconds")
        save_state(state, state_path)
        out.update({"anchor": day_anchor, "reason": "alert sent"})
    else:
        out.update({"error": info, "reason": f"send failed: {info}"})

    return out


def _format_message(rate_now: float, anchor: float, delta_pct: float) -> str:
    """Plain-text alert body — works on every provider without markup."""
    ts = datetime.now().strftime("%d/%m %H:%M")
    return (
        f"🇧🇷 cambio · USD/BRL · {ts}\n\n"
        f"R$ {rate_now:.4f}  ({delta_pct:+.2f}% intra-day vs R$ {anchor:.4f})\n\n"
        f"Janela de oportunidade — confere o sinal antes de converter.\n"
        f"https://higlobe.com/webapp/en/login"
    )
