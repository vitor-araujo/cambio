"""
Journal — append every live decision to a CSV so the model's calls
can be audited day-to-day and rendered as "last call vs today" feedback.

Schema is flat, no dependencies beyond the stdlib.
"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

JOURNAL_PATH = ".fx_journal.csv"
FIELDS = [
    "ts",
    "rate_signal",
    "rate_live",
    "decision",
    "p_now",
    "p_split",
    "p_wait",
    "composite",
    "agreement",
    "regime",
    "notified",
]


@dataclass
class Entry:
    ts: datetime
    rate_signal: float
    rate_live: Optional[float]
    decision: str
    p_now: float
    p_split: float
    p_wait: float
    composite: float
    agreement: float
    regime: float
    notified: bool = False


def append(entry: Entry, path: str = JOURNAL_PATH) -> None:
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        w.writerow(
            [
                entry.ts.isoformat(timespec="seconds"),
                f"{entry.rate_signal:.4f}",
                f"{entry.rate_live:.4f}" if entry.rate_live is not None else "",
                entry.decision,
                f"{entry.p_now:.4f}",
                f"{entry.p_split:.4f}",
                f"{entry.p_wait:.4f}",
                f"{entry.composite:+.4f}",
                f"{entry.agreement:.4f}",
                f"{entry.regime:+.4f}",
                "1" if entry.notified else "0",
            ]
        )


def _row_to_entry(r: dict) -> Optional[Entry]:
    try:
        return Entry(
            ts=datetime.fromisoformat(r["ts"]),
            rate_signal=float(r["rate_signal"]),
            rate_live=float(r["rate_live"]) if r.get("rate_live") else None,
            decision=r["decision"],
            p_now=float(r["p_now"]),
            p_split=float(r["p_split"]),
            p_wait=float(r["p_wait"]),
            composite=float(r["composite"]),
            agreement=float(r["agreement"]),
            regime=float(r["regime"]),
            notified=r.get("notified", "0") == "1",
        )
    except Exception:
        return None


def last_entry(path: str = JOURNAL_PATH) -> Optional[Entry]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    for r in reversed(rows):
        e = _row_to_entry(r)
        if e is not None:
            return e
    return None


def last_notified(path: str = JOURNAL_PATH) -> Optional[Entry]:
    """Last entry where we actually fired the browser alert."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    for r in reversed(rows):
        if r.get("notified") == "1":
            return _row_to_entry(r)
    return None


def render_summary(prev: Optional[Entry], cur_rate: float) -> Optional[str]:
    """One-line feedback: last call, age, did the rate move our way."""
    if prev is None:
        return None
    age = datetime.now() - prev.ts
    days = age.days
    hours = age.seconds // 3600
    delta_pct = (cur_rate / prev.rate_signal - 1) * 100

    if prev.decision == "exchange_now":
        ok = delta_pct < -0.1
    elif prev.decision == "wait":
        ok = delta_pct > 0.1
    else:
        ok = abs(delta_pct) < 0.3

    mark = "✓" if ok else "✗" if abs(delta_pct) > 0.3 else "·"
    age_str = f"{days}d" if days >= 1 else f"{hours}h"
    label = {"exchange_now": "NOW", "wait": "WAIT", "split": "SPLIT"}.get(
        prev.decision, prev.decision
    )

    return (
        f"último sinal há {age_str}: {label} @ R$ {prev.rate_signal:.4f}  "
        f"→ agora R$ {cur_rate:.4f}  ({delta_pct:+.2f}%) {mark}"
    )


def should_notify(
    decision: str,
    p_now: float,
    threshold: float = 0.55,
    cooldown_hours: int = 6,
    path: str = JOURNAL_PATH,
) -> bool:
    """
    Trigger the browser alert when:
      - decision is exchange_now
      - p_now >= threshold (avoid noise on weak NOWs)
      - we have not already alerted within the cooldown window
    """
    if decision != "exchange_now" or p_now < threshold:
        return False
    last = last_notified(path)
    if last is None:
        return True
    return (datetime.now() - last.ts) > timedelta(hours=cooldown_hours)
