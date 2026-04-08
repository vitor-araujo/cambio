#!/usr/bin/env python3
"""
USD/BRL Exchange Timing Model
  python fx_timing.py             — live signal analysis
  python fx_timing.py --backtest  — walk-forward backtest (2nd & 17th since 2022)

pip install yfinance pandas numpy
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from signals import Signal, build_signals

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
LIVE_FETCH_DAYS = 180
BACKTEST_START = "2022-01-01"
ORACLE_HORIZON = 14  # days ahead for correctness evaluation
ORACLE_THRESH = 0.003  # 0.3 % move to call it directional (next-check-date oracle)
MAX_WAIT = 6  # max consecutive WAIT decisions
AMOUNT = 10_000  # USD per scenario
REGIME_STRENGTH = 0.55  # how hard the regime filter shifts probabilities

TICKERS = {
    "usdbrl": "BRL=X",
    "dxy": "DX-Y.NYB",
    "brent": "BZ=F",
    "vale": "VALE",
    "vix": "^VIX",
    "ibov": "^BVSP",
    "us_rate": "^IRX",  # 13-week T-bill (FFR proxy, % per year)
}

BCB_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"
    "?formato=json&dataInicial={start}"
)


# ── Data Fetching ─────────────────────────────────────────────────────────────
def fetch(start: str, end: Optional[datetime] = None) -> dict[str, pd.Series]:
    """
    Fetch Close series for all tickers.
    For USD/BRL also stores 'usdbrl_high' and 'usdbrl_low' (needed for ADX).
    """
    end = end or datetime.now()
    out: dict[str, pd.Series] = {}

    for key, ticker in TICKERS.items():
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            )
            if df.empty:
                continue

            close = df["Close"].squeeze().dropna()
            if close.empty:
                continue

            out[key] = close

            if key == "usdbrl":
                if "High" in df.columns and "Low" in df.columns:
                    out["usdbrl_high"] = df["High"].squeeze().dropna()
                    out["usdbrl_low"] = df["Low"].squeeze().dropna()

        except Exception:
            pass

    return out


def fetch_selic(start: str) -> Optional[pd.Series]:
    """
    BCB open API — series 432 (CDI overnight daily rate, % per day).
    Annualises to % per year and forward-fills to business days.
    Returns None if the API is unreachable.
    """
    try:
        start_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%d/%m/%Y")
        url = BCB_URL.format(start=start_fmt)
        req = Request(url, headers={"User-Agent": "fx_timing/1.0"})
        raw = json.loads(urlopen(req, timeout=15).read())

        dates, values = [], []
        for r in raw:
            try:
                dates.append(datetime.strptime(r["data"], "%d/%m/%Y"))
                values.append(float(r["valor"].replace(",", ".")))
            except (KeyError, ValueError):
                continue

        if not dates:
            return None

        s = pd.Series(values, index=pd.DatetimeIndex(dates), name="selic_daily")

        # Series 432 returns % per day (e.g. ~0.046).
        # Annualise: (1 + r/100)^252 − 1, then ×100 → % per year.
        if s.mean() < 1.0:
            s = ((1 + s / 100) ** 252 - 1) * 100

        full_range = pd.date_range(s.index.min(), datetime.now(), freq="B")
        return s.reindex(full_range).ffill().dropna()

    except Exception as e:
        print(f"  ⚠  BCB API unavailable ({e}) — carry signal disabled")
        return None


def build_carry_diff(
    selic: Optional[pd.Series],
    us_rate: Optional[pd.Series],
) -> Optional[pd.Series]:
    """
    SELIC (% /yr) − US 3-month T-bill (% /yr).
    ^IRX from yfinance is already in % per year.
    """
    if selic is None or us_rate is None:
        return None

    # Align on common business days
    combined = (
        pd.concat([selic.rename("selic"), us_rate.rename("irx")], axis=1)
        .ffill()
        .dropna()
    )
    if len(combined) < 20:
        return None

    diff = combined["selic"] - combined["irx"]
    return diff.rename("carry_diff")


# ── Probability Model ─────────────────────────────────────────────────────────
def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def probabilities(signals: list[Signal]) -> dict:
    """
    Weighted composite score → sigmoid → three-way prob (exchange_now / split / wait).
    Weights are normalised by the sum of present signals so missing carry doesn't
    silently deflate the composite.
    Signal disagreement (high σ) routes mass into 'split'.
    """
    if not signals:
        return {
            "exchange_now": 1 / 3,
            "split": 1 / 3,
            "wait": 1 / 3,
            "composite": 0.0,
            "agreement": 0.5,
        }

    total_w = sum(s.weight for s in signals)
    composite = sum(s.score * s.weight for s in signals) / total_w
    agreement = max(0.0, 1.0 - float(np.std([s.score for s in signals])))

    p_now_raw = sigmoid(composite * 4.0)
    p_wait_raw = 1.0 - p_now_raw
    p_split = (1.0 - agreement) * 0.4  # up to 40 % at full disagreement

    p_now = p_now_raw * (1.0 - p_split)
    p_wait = p_wait_raw * (1.0 - p_split)
    total = p_now + p_wait + p_split

    return {
        "exchange_now": p_now / total,
        "split": p_split / total,
        "wait": p_wait / total,
        "composite": composite,
        "agreement": agreement,
    }


def apply_regime(probs: dict, regime: float) -> dict:
    """
    ADX regime modifier.
    regime +1 = USD/BRL uptrend (BRL weakening) → suppress NOW, boost WAIT.
    regime -1 = downtrend → suppress WAIT, boost NOW.
    No-op in ranging markets (|regime| < 0.15).
    """
    if abs(regime) < 0.15:
        return {**probs, "regime": regime}

    pn = probs["exchange_now"]
    pw = probs["wait"]
    ps = probs["split"]

    adj = regime * REGIME_STRENGTH
    pn_adj = pn * max(0.05, 1.0 - adj)
    pw_adj = pw * (1.0 + adj)

    total = pn_adj + ps + pw_adj
    return {
        "exchange_now": pn_adj / total,
        "split": ps / total,
        "wait": pw_adj / total,
        "composite": probs["composite"],
        "agreement": probs["agreement"],
        "regime": regime,
    }


def decide(probs: dict) -> str:
    """
    NOW  fires when p_now > 0.51 (high-conviction peak, proven at 52 % accuracy).
    WAIT is the conservative default for all other cases.
    SPLIT is reserved for when p_split dominates (rare).
    """
    pn, ps, pw = probs["exchange_now"], probs["split"], probs["wait"]
    if pn >= pw and pn >= ps and pn > 0.51:
        return "exchange_now"
    return "wait"


# ── Live Rendering ────────────────────────────────────────────────────────────
def pbar(p: float, w: int = 34) -> str:
    n = int(round(p * w))
    return "█" * n + "░" * (w - n)


def sbar(score: float, half: int = 8) -> str:
    n = min(int(abs(score) * half), half)
    if score >= 0:
        return "|" + " " * half + "▶" * n + " " * (half - n) + "|"
    return "|" + " " * (half - n) + "◀" * n + " " * half + "|"


def regime_label(regime: float) -> str:
    if regime > 0.6:
        return "STRONG UPTREND  (BRL weakening — regime filter: suppressing NOW)"
    if regime > 0.15:
        return "mild uptrend  (regime filter: slight WAIT bias)"
    if regime < -0.6:
        return "STRONG DOWNTREND  (BRL strengthening — regime filter: suppressing WAIT)"
    if regime < -0.15:
        return "mild downtrend  (regime filter: slight NOW bias)"
    return "ranging / no trend  (mean-reversion signals active)"


def render_live(signals: list[Signal], probs: dict) -> None:
    rate = next((s.raw for s in signals if s.name == "USD/BRL Level"), None)
    regime = probs.get("regime", 0.0)
    W = 66

    print()
    print("═" * W)
    print("  USD → BRL   EXCHANGE TIMING MODEL")
    print(
        f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}"
        + (f"   ·   R$ {rate:.4f}" if rate else "")
    )
    print("═" * W)

    print()
    print(f"  Trend Regime:  {regime_label(regime)}")
    print()
    print("  SIGNALS" + " " * 22 + "← WAIT   NOW →   score    wt")
    print("  " + "─" * (W - 2))
    print()

    for s in signals:
        v = "NOW " if s.score > 0.15 else "WAIT" if s.score < -0.15 else "FLAT"
        print(f"  {s.name:<18} {sbar(s.score)}  {s.score:+.2f}  {s.weight:.0%}  [{v}]")
        print(f"    {s.note}")
        print()

    comp = probs["composite"]
    agree = probs["agreement"]
    print(
        f"  Composite: {comp:+.3f}   Agreement: {agree:.0%}   Regime adj: {regime:+.2f}"
    )

    print()
    print("  PROBABILITY DISTRIBUTION")
    print("  " + "─" * (W - 2))
    print()
    for label, key in [
        ("Exchange Now", "exchange_now"),
        ("Split 50/50", "split"),
        ("Wait", "wait"),
    ]:
        p = probs[key]
        print(f"  {label:<13}  {p:>5.1%}  [{pbar(p)}]")

    d = decide(probs)
    print()
    print("  " + "─" * (W - 2))
    print()
    if d == "exchange_now":
        print("  ▶  ⚡ EXCHANGE NOW")
        print("     Rate is favorable — signals point to BRL weakening ahead")
    elif d == "wait":
        print("  ▶  ⏳ WAIT")
        print("     USD strengthening or carry attractive — better rate likely ahead")
    else:
        print("  ▶  ⚖  SPLIT  (50 % now · 50 % later)")
        print("     Mixed signals — splitting reduces timing regret")

    print()
    print("  Time horizon override:")
    print("    < 7 days  → execute regardless of model")
    print("    7–30d     → follow model recommendation")
    print("    > 30d     → weight WAIT more aggressively")
    print()
    print("═" * W)
    print()


# ── Backtest Helpers ──────────────────────────────────────────────────────────
def decision_dates(start: str = BACKTEST_START) -> list[date]:
    """All 2nd and 17th of each month from start to two days ago."""
    out: list[date] = []
    sd = datetime.strptime(start, "%Y-%m-%d").date()
    end = date.today() - timedelta(days=2)
    y, m = sd.year, sd.month
    while date(y, m, 1) <= end:
        for day in (2, 17):
            d = date(y, m, day)
            if sd <= d <= end:
                out.append(d)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def nearest_rate(usdbrl: pd.Series, d: date, max_offset: int = 5) -> Optional[float]:
    for offset in range(max_offset + 1):
        ts = pd.Timestamp(d + timedelta(days=offset))
        if ts in usdbrl.index:
            return float(usdbrl.loc[ts])
    return None


def oracle(usdbrl: pd.Series, d: date, dates: list[date]) -> str:
    """
    Next-check-date oracle: compare rate at d to rate at the next 2nd/17th.
    More honest than max-over-14d which penalises brief 1-day spikes that are
    never tradeable at our bi-weekly cadence.
    """
    r0 = nearest_rate(usdbrl, d)
    if r0 is None:
        return "unknown"
    future_dates = [x for x in dates if x > d]
    if not future_dates:
        return "unknown"
    r_next = nearest_rate(usdbrl, future_dates[0])
    if r_next is None:
        return "unknown"
    if r_next > r0 * (1 + ORACLE_THRESH):
        return "wait"
    if r0 > r_next * (1 + ORACLE_THRESH):
        return "exchange_now"
    return "split"


# ── Backtest Core ─────────────────────────────────────────────────────────────
@dataclass
class Row:
    date: date
    model: str
    oracle: str
    correct: bool
    rate: float
    rate_14d: Optional[float]
    rate_30d: Optional[float]
    composite: float
    agreement: float
    regime: float


def run_backtest(
    all_data: dict[str, pd.Series],
    carry_diff: Optional[pd.Series],
) -> list[Row]:
    usdbrl = all_data["usdbrl"]
    dates = decision_dates()
    rows: list[Row] = []

    print(f"\n  Running walk-forward backtest on {len(dates)} decision dates...")

    for d in dates:
        ts = pd.Timestamp(d)
        sliced: dict[str, pd.Series] = {k: v.loc[:ts] for k, v in all_data.items()}
        carry_sliced = carry_diff.loc[:ts] if carry_diff is not None else None

        # Need at least LONG_MA history in every key series
        core_keys = {"usdbrl", "dxy", "brent", "vix"}
        if not core_keys.issubset(sliced) or any(
            len(sliced[k]) < 25 for k in core_keys
        ):
            continue

        sigs = build_signals(sliced, carry_sliced)
        # sigs may be (list, regime) tuple or just list depending on version
        if isinstance(sigs, tuple):
            sigs, regime = sigs
        else:
            regime = 0.0

        probs = probabilities(sigs)
        probs = apply_regime(probs, regime)
        dec = decide(probs)
        orc = oracle(usdbrl, d, dates)
        rate = nearest_rate(usdbrl, d)

        if rate is None or orc == "unknown":
            continue

        rows.append(
            Row(
                date=d,
                model=dec,
                oracle=orc,
                correct=(dec == orc),
                rate=rate,
                rate_14d=nearest_rate(usdbrl, d + timedelta(days=14)),
                rate_30d=nearest_rate(usdbrl, d + timedelta(days=30)),
                composite=probs["composite"],
                agreement=probs["agreement"],
                regime=regime,
            )
        )

    return rows


# ── Sequential Simulation ─────────────────────────────────────────────────────
@dataclass
class Scenario:
    start_date: date
    execute_date: date
    start_rate: float
    execute_rate: float
    periods_waited: int
    brl_model: float
    brl_immediate: float


def sequential_sim(rows: list[Row]) -> list[Scenario]:
    """
    From each 2nd-of-month, follow model decisions until NOW/SPLIT fires
    or MAX_WAIT periods elapse, then compare BRL received vs day-1 exchange.
    """
    row_map = {r.date: r for r in rows}
    all_dates = sorted(row_map)
    scenarios: list[Scenario] = []

    for start_d in (d for d in all_dates if d.day == 2):
        pool = [d for d in all_dates if d >= start_d][: MAX_WAIT + 1]
        if not pool:
            continue

        start_r = row_map[start_d].rate
        execute_d = pool[-1]
        execute_r = row_map[execute_d].rate if execute_d in row_map else None
        periods = len(pool) - 1

        for i, d in enumerate(pool):
            if d not in row_map:
                continue
            if row_map[d].model in ("exchange_now", "split"):
                execute_d = d
                execute_r = row_map[d].rate
                periods = i
                break

        if execute_r is None:
            continue

        scenarios.append(
            Scenario(
                start_date=start_d,
                execute_date=execute_d,
                start_rate=start_r,
                execute_rate=execute_r,
                periods_waited=periods,
                brl_model=AMOUNT * execute_r,
                brl_immediate=AMOUNT * start_r,
            )
        )

    return scenarios


# ── Backtest Rendering ────────────────────────────────────────────────────────
def render_backtest(rows: list[Row], scenarios: list[Scenario]) -> None:
    W = 86

    print("\n" + "═" * W)
    print("  USD → BRL  WALK-FORWARD BACKTEST  (2nd & 17th of each month since 2022)")
    print("  Signals: DXY · Brent · VALE · VIX · IBOV · Carry · Level · RSI · BB%B")
    print("  Regime:  ADX(14) trend filter applied to final probabilities")
    print("═" * W)

    # ── Decision table
    lbl = {
        "exchange_now": "NOW  ",
        "split": "SPLIT",
        "wait": "WAIT ",
        "unknown": "  ?  ",
    }

    print()
    print(
        f"  {'Date':<12} {'Model':<7} {'Oracle':<7} "
        f"{'Rate':>7} {'14d':>7} {'30d':>7} "
        f"{'Δ14d':>6}  {'Δ30d':>6}  {'Rgm':>5}  OK"
    )
    print("  " + "─" * (W - 2))

    for r in rows:
        d14 = f"{r.rate_14d:.3f}" if r.rate_14d else "  n/a"
        d30 = f"{r.rate_30d:.3f}" if r.rate_30d else "  n/a"
        p14 = f"{(r.rate_14d / r.rate - 1) * 100:+.1f}%" if r.rate_14d else "  n/a"
        p30 = f"{(r.rate_30d / r.rate - 1) * 100:+.1f}%" if r.rate_30d else "  n/a"
        rgm = f"{r.regime:+.2f}"
        ok = "✓" if r.correct else "✗"
        print(
            f"  {str(r.date):<12} {lbl[r.model]:<7} {lbl[r.oracle]:<7}"
            f" {r.rate:>7.3f} {d14:>7} {d30:>7}"
            f" {p14:>6}  {p30:>6}  {rgm:>5}  {ok}"
        )

    # ── Accuracy
    total = len(rows)
    n_corr = sum(r.correct for r in rows)

    print()
    print("  " + "─" * (W - 2))
    print(f"\n  ACCURACY  ({total} decisions)")
    print()
    print(f"  Overall              {n_corr / total * 100:>5.1f}%   ({n_corr}/{total})")
    print()

    for label, key in [
        ("Exchange Now", "exchange_now"),
        ("Wait", "wait"),
        ("Split", "split"),
    ]:
        subset = [r for r in rows if r.model == key]
        if subset:
            acc = sum(r.correct for r in subset) / len(subset)
            tag = "  ✓" if key != "split" and acc > 0.51 else ""
            print(f"  {label:<16}   {acc * 100:>5.1f}%   ({len(subset)} calls){tag}")

    # per-year breakdown
    years = sorted({r.date.year for r in rows})
    print()
    print(f"  {'Year':<6}  {'Acc':>6}  {'NOW':>4}  {'WAIT':>4}  {'SPLIT':>5}  {'N':>4}")
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 4}  {'─' * 4}  {'─' * 5}  {'─' * 4}")
    for yr in years:
        yr_rows = [r for r in rows if r.date.year == yr]
        acc = sum(r.correct for r in yr_rows) / len(yr_rows)
        n_now = sum(1 for r in yr_rows if r.model == "exchange_now")
        n_wait = sum(1 for r in yr_rows if r.model == "wait")
        n_split = sum(1 for r in yr_rows if r.model == "split")
        print(
            f"  {yr:<6}  {acc * 100:>5.1f}%"
            f"  {n_now:>4}  {n_wait:>4}  {n_split:>5}  {len(yr_rows):>4}"
        )

    # ── Sequential P&L
    if not scenarios:
        print("\n" + "═" * W + "\n")
        return

    gains = [s.brl_model - s.brl_immediate for s in scenarios]
    pcts = [(s.brl_model / s.brl_immediate - 1) * 100 for s in scenarios]
    wins = sum(1 for g in gains if g > 0)
    losses = sum(1 for g in gains if g < 0)

    print()
    print("  " + "─" * (W - 2))
    print(
        f"\n  SEQUENTIAL P&L  (${AMOUNT:,} per scenario"
        f" · follow model until execute · vs immediate exchange)"
    )
    print()
    print(
        f"  {'Start':<12} {'Execute':<12} {'R@Start':>8} {'R@Exec':>8}"
        f" {'Wait':>5}  {'ΔBRL':>10}  {'Δ%':>7}  Result"
    )
    print("  " + "─" * (W - 2))

    for s in scenarios:
        g = s.brl_model - s.brl_immediate
        pct = (s.brl_model / s.brl_immediate - 1) * 100
        sign = "+" if g >= 0 else ""
        flag = "▲ win" if g > 50 else "▼ loss" if g < -50 else "  tie"
        print(
            f"  {str(s.start_date):<12} {str(s.execute_date):<12}"
            f" {s.start_rate:>8.4f} {s.execute_rate:>8.4f}"
            f" {str(s.periods_waited) + '×':>5}"
            f"  {sign}{g:>8.0f}  {sign}{pct:>5.2f}%  {flag}"
        )

    print()
    print(
        f"  Scenarios: {len(scenarios)}   Wins: {wins}   Losses: {losses}   Ties: {len(scenarios) - wins - losses}"
    )
    print(f"  Avg BRL gain / scenario   R$ {np.mean(gains):>+8.0f}")
    print(f"  Avg return vs immediate       {np.mean(pcts):>+6.2f}%")
    print(f"  Cumulative BRL edge       R$ {sum(gains):>+8.0f}")
    print(f"  Win rate                      {wins / len(scenarios) * 100:>5.1f}%")
    print()
    print("═" * W)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def _print_fetch_status(data: dict[str, pd.Series]) -> None:
    for k, v in data.items():
        if not k.endswith(("_high", "_low")):
            print(f"  ✓  {k:<12} — {len(v)} trading days")


def main() -> None:
    parser = argparse.ArgumentParser(description="USD/BRL Exchange Timing Model")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Walk-forward backtest on 2nd & 17th since 2022",
    )
    args = parser.parse_args()

    if args.backtest:
        warmup = (
            datetime.strptime(BACKTEST_START, "%Y-%m-%d") - timedelta(days=90)
        ).strftime("%Y-%m-%d")
        print(f"\n  Fetching full history from {warmup} ...")
        all_data = fetch(warmup)
        if not all_data:
            print("  ERROR: no data fetched — check connectivity.\n")
            return
        _print_fetch_status(all_data)

        print("  Fetching SELIC from BCB ...")
        selic = fetch_selic(warmup)
        carry_diff = build_carry_diff(selic, all_data.get("us_rate"))

        if carry_diff is not None:
            print(
                f"  ✓  carry_diff   — {len(carry_diff)} days  "
                f"(current {carry_diff.iloc[-1]:.1f} %/yr)"
            )
        else:
            print("  ⚠  carry signal disabled (BCB or ^IRX unavailable)")

        rows = run_backtest(all_data, carry_diff)
        scenarios = sequential_sim(rows)
        render_backtest(rows, scenarios)

    else:
        start = (datetime.now() - timedelta(days=LIVE_FETCH_DAYS)).strftime("%Y-%m-%d")
        print("\n  Fetching market data...")
        data = fetch(start)
        if not data:
            print("  ERROR: no data fetched — check connectivity.\n")
            return
        _print_fetch_status(data)

        print("  Fetching SELIC from BCB ...")
        selic = fetch_selic(start)
        carry_diff = build_carry_diff(selic, data.get("us_rate"))

        result = build_signals(data, carry_diff)
        if isinstance(result, tuple):
            sigs, regime = result
        else:
            sigs, regime = result, 0.0

        probs = probabilities(sigs)
        probs = apply_regime(probs, regime)
        render_live(sigs, probs)


if __name__ == "__main__":
    main()
