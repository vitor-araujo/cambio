"""Small, dependency-free terminal UI for cambio.

The web app is the primary cockpit; this module keeps long-running CLI sessions
quiet and useful. Interactive terminals get an in-place market tape, while
redirected logs only receive state changes and periodic heartbeats.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TextIO


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
INK = "\033[38;5;153m"
COPPER = "\033[38;5;215m"
MINT = "\033[38;5;115m"
ROSE = "\033[38;5;210m"
SLATE = "\033[38;5;67m"


def supports_color(stream: TextIO = sys.stderr) -> bool:
    return bool(
        hasattr(stream, "isatty")
        and stream.isatty()
        and os.getenv("NO_COLOR") is None
        and os.getenv("TERM", "") != "dumb"
    )


def paint(value: object, *codes: str, stream: TextIO = sys.stderr) -> str:
    text = str(value)
    if not supports_color(stream):
        return text
    return "".join(codes) + text + RESET


def startup_banner(
    *,
    interval_min: int,
    cadence_days: int,
    stream: TextIO = sys.stderr,
) -> None:
    """Render one compact launch card instead of a wall of setup logs."""
    brand = paint("CAMBIO", BOLD, INK, stream=stream)
    mode = paint("LIVE EXECUTION", BOLD, COPPER, stream=stream)
    print(file=stream)
    print(f"  {brand}  /  {mode}", file=stream)
    print(
        "  "
        + paint("━" * 58, SLATE, stream=stream),
        file=stream,
    )
    print(
        f"  feed  {interval_min}m   ·   tranche cadence  {cadence_days}d"
        "   ·   Ctrl+C to stop",
        file=stream,
    )
    print(file=stream)


@dataclass(frozen=True)
class WatchSnapshot:
    decision: str
    rate: float
    p_now: float
    opportunity_score: float
    size: float
    opportunity_threshold: float
    due_label: str
    trigger: str = ""
    reason: str = ""
    notified: bool = False


class WatchConsole:
    """A restrained live tape with TTY redraw and log-friendly throttling."""

    def __init__(
        self,
        *,
        stream: TextIO = sys.stderr,
        heartbeat_cycles: int = 15,
    ) -> None:
        self.stream = stream
        self.heartbeat_cycles = max(1, heartbeat_cycles)
        self._last_state: Optional[tuple[str, str]] = None
        self._cycle = 0
        self._interactive = supports_color(stream)
        self._line_open = False

    def _badge(self, decision: str) -> str:
        if decision == "exchange_now":
            return paint(" EXECUTE ", BOLD, MINT, stream=self.stream)
        if decision == "split":
            return paint(" STAGE   ", BOLD, COPPER, stream=self.stream)
        return paint(" WATCH   ", BOLD, INK, stream=self.stream)

    def update(self, snapshot: WatchSnapshot) -> None:
        self._cycle += 1
        state = (snapshot.decision, snapshot.due_label)
        materially_changed = state != self._last_state
        if not self._interactive and not (
            materially_changed or self._cycle == 1 or self._cycle % self.heartbeat_cycles == 0
        ):
            return

        now = datetime.now().strftime("%H:%M:%S")
        badge = self._badge(snapshot.decision)
        rate = paint(f"R$ {snapshot.rate:.4f}", BOLD, stream=self.stream)
        if snapshot.trigger in {"initial_fill", "cadence_due"}:
            prob = f"cadence due · quality {snapshot.opportunity_score:>4.0%}"
        else:
            prob = (
                f"quality {snapshot.opportunity_score:>4.0%} / "
                f"hurdle {snapshot.opportunity_threshold:>4.0%}"
            )
        prob += f" · p(now) {snapshot.p_now:>4.0%}"
        tranche = paint(f"{snapshot.size:.0%}", COPPER, stream=self.stream)
        alert = paint("  ● alert", MINT, stream=self.stream) if snapshot.notified else ""
        line = (
            f"  {paint(now, DIM, stream=self.stream)}  {badge}  {rate}"
            f"  ·  {prob}  ·  tranche {tranche}  ·  {snapshot.due_label}{alert}"
        )

        if self._interactive:
            print("\r\033[2K" + line, end="", file=self.stream, flush=True)
            self._line_open = True
        else:
            print(line, file=self.stream, flush=True)
        self._last_state = state

    def event(self, message: str, *, error: bool = False) -> None:
        if self._line_open:
            print(file=self.stream)
            self._line_open = False
        color = ROSE if error else COPPER
        print(f"  {paint('◆', color, stream=self.stream)}  {message}", file=self.stream)

    def close(self) -> None:
        if self._line_open:
            print(file=self.stream)
            self._line_open = False
