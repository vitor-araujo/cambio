"""
Journal (SQLite) — same API as journal.py but backed by a local SQLite DB.

Migrates from .fx_journal.csv automatically on first access.
Handles concurrent writes safely with WAL mode.
"""

import csv
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = ".fx_journal.db"
CSV_PATH = ".fx_journal.csv"

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Thread-local connection in WAL mode."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        _ensure_table(conn)
        _migrate_csv_if_needed(conn)
    return _local.conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            rate_signal REAL NOT NULL,
            rate_live REAL,
            decision TEXT NOT NULL,
            size REAL NOT NULL DEFAULT 0.25,
            p_now REAL NOT NULL,
            p_split REAL NOT NULL,
            p_wait REAL NOT NULL,
            composite REAL NOT NULL,
            agreement REAL NOT NULL,
            regime REAL NOT NULL,
            notified INTEGER NOT NULL DEFAULT 0,
            executed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_notified ON journal(notified)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_executed ON journal(executed)")
    conn.commit()


def _migrate_csv_if_needed(conn: sqlite3.Connection) -> None:
    if not os.path.exists(CSV_PATH):
        return
    row = conn.execute("SELECT COUNT(*) as cnt FROM journal").fetchone()
    if row["cnt"] > 0:
        return
    try:
        with open(CSV_PATH) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return
    for r in rows:
        try:
            ts = datetime.fromisoformat(r.get("ts", ""))
            conn.execute(
                """INSERT INTO journal
                   (ts, rate_signal, rate_live, decision, size,
                    p_now, p_split, p_wait, composite, agreement, regime,
                    notified, executed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts.isoformat(timespec="seconds"),
                    float(r["rate_signal"]),
                    float(r["rate_live"]) if r.get("rate_live") else None,
                    r["decision"],
                    float(r.get("size") or 0.25),
                    float(r["p_now"]),
                    float(r["p_split"]),
                    float(r["p_wait"]),
                    float(r["composite"]),
                    float(r["agreement"]),
                    float(r["regime"]),
                    1 if r.get("notified", "0") == "1" else 0,
                    1 if r.get("executed", "0") == "1" else 0,
                ),
            )
        except Exception:
            continue
    conn.commit()
    print(f"  [journal] migrated {len(rows)} entries from {CSV_PATH}")


# ── dataclass (identical to journal.py) ──────────────────────────────────────


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
    size: float = 0.25
    executed: bool = False


def _row_to_entry(r: sqlite3.Row) -> Entry:
    return Entry(
        ts=datetime.fromisoformat(r["ts"]),
        rate_signal=r["rate_signal"],
        rate_live=r["rate_live"],
        decision=r["decision"],
        p_now=r["p_now"],
        p_split=r["p_split"],
        p_wait=r["p_wait"],
        composite=r["composite"],
        agreement=r["agreement"],
        regime=r["regime"],
        notified=bool(r["notified"]),
        size=r["size"],
        executed=bool(r["executed"]),
    )


# ── public API (matches journal.py) ───────────────────────────────────────────


def append(entry: Entry) -> None:
    conn = _conn()
    conn.execute(
        """INSERT INTO journal
           (ts, rate_signal, rate_live, decision, size,
            p_now, p_split, p_wait, composite, agreement, regime,
            notified, executed)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            entry.ts.isoformat(timespec="seconds"),
            entry.rate_signal,
            entry.rate_live,
            entry.decision,
            entry.size,
            entry.p_now,
            entry.p_split,
            entry.p_wait,
            entry.composite,
            entry.agreement,
            entry.regime,
            1 if entry.notified else 0,
            1 if entry.executed else 0,
        ),
    )
    conn.commit()


def last_entry() -> Optional[Entry]:
    r = _conn().execute("SELECT * FROM journal ORDER BY id DESC LIMIT 1").fetchone()
    return _row_to_entry(r) if r else None


def last_notified() -> Optional[Entry]:
    r = (
        _conn()
        .execute("SELECT * FROM journal WHERE notified = 1 ORDER BY id DESC LIMIT 1")
        .fetchone()
    )
    return _row_to_entry(r) if r else None


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
    threshold: float = 0.40,
    cooldown_hours: int = 6,
) -> bool:
    if decision != "exchange_now" or p_now < threshold:
        return False
    last = last_notified()
    if last is None:
        return True
    return (datetime.now() - last.ts) > timedelta(hours=cooldown_hours)


def last_executed() -> Optional[Entry]:
    r = (
        _conn()
        .execute("SELECT * FROM journal WHERE executed = 1 ORDER BY id DESC LIMIT 1")
        .fetchone()
    )
    return _row_to_entry(r) if r else None


def mark_executed(when: Optional[datetime] = None) -> Optional[Entry]:
    conn = _conn()
    if when is not None:
        r = conn.execute(
            "SELECT id FROM journal WHERE ts = ? ORDER BY id DESC LIMIT 1",
            (when.isoformat(timespec="seconds"),),
        ).fetchone()
    else:
        r = conn.execute("SELECT id FROM journal ORDER BY id DESC LIMIT 1").fetchone()
    if not r:
        return None
    conn.execute("UPDATE journal SET executed = 1 WHERE id = ?", (r["id"],))
    conn.commit()
    row = conn.execute("SELECT * FROM journal WHERE id = ?", (r["id"],)).fetchone()
    return _row_to_entry(row) if row else None


def days_until_deadline(deadline_days: int = 15) -> Optional[int]:
    anchor = last_executed()
    if anchor is None:
        return None
    elapsed = (datetime.now() - anchor.ts).days
    return max(0, deadline_days - elapsed)


def audit_summary(days: int = 30) -> dict:
    entries = all_entries()
    cutoff = datetime.now() - timedelta(days=days)
    recent = [e for e in entries if e.ts >= cutoff]

    alerts = [e for e in recent if e.notified]
    executed = [e for e in recent if e.executed]

    missed: list[Entry] = []
    for a in alerts:
        followup = [
            e
            for e in entries
            if e.executed and a.ts <= e.ts <= a.ts + timedelta(hours=24)
        ]
        if not followup:
            missed.append(a)

    override_rate = (len(missed) / len(alerts)) if alerts else 0.0

    return {
        "window_days": days,
        "total_runs": len(recent),
        "alerts": len(alerts),
        "executed": len(executed),
        "missed_alerts": len(missed),
        "override_rate": override_rate,
        "missed_entries": missed,
    }


# ── extended API for the UI ──────────────────────────────────────────────────


def all_entries(limit: int = 0) -> list[Entry]:
    """All entries, newest first. limit=0 means no limit."""
    sql = "SELECT * FROM journal ORDER BY id DESC"
    if limit > 0:
        sql += f" LIMIT {limit}"
    return [_row_to_entry(r) for r in _conn().execute(sql).fetchall()]


def notified_entries(limit: int = 20) -> list[Entry]:
    rows = (
        _conn()
        .execute(
            "SELECT * FROM journal WHERE notified = 1 ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        .fetchall()
    )
    return [_row_to_entry(r) for r in rows]


def show_journal(limit: int = 20) -> None:
    """CLI table of recent entries."""
    entries = all_entries(limit)
    if not entries:
        print("  (journal vazio)")
        return
    print()
    print(
        f"  {'data':<12} {'dec':<6} {'taxa':>10} {'p(agora)':>8} {'comp':>7} {'regime':>7} {'size':>5} {'alert':>5}"
    )
    print("  " + "─" * 62)
    for e in entries[:limit]:
        dec = {"exchange_now": "NOW", "wait": "WAIT", "split": "SPLIT"}.get(
            e.decision, e.decision
        )
        rate = e.rate_live or e.rate_signal
        alert = "📱" if e.notified else "—"
        print(
            f"  {e.ts.strftime('%H:%M %d/%m'):<12} {dec:<6} "
            f"R$ {rate:>7.4f} {e.p_now:>7.1%} "
            f"{e.composite:>+6.2f} {e.regime:>+6.2f} "
            f"{e.size:>4.0%} {alert:>5}"
        )
    print()
