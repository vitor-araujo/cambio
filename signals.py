"""
FX Signal Library — USD/BRL Exchange Timing Model
All scores ∈ [-1, 1]:  positive = exchange NOW  (USD/BRL likely to fall)
                        negative = WAIT           (USD/BRL likely to rise)

3-factor structure (academic consensus for EM FX):
  Momentum    ~42% — DXY, Brent, VALE, VIX, IBOV
  Carry       ~20% — SELIC − FFR / realized vol
  Mean-Rev    ~33% — RSI, BB %B (both ADX-conditioned), USD/BRL percentile
"""

import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Weights ───────────────────────────────────────────────────────────────────
WEIGHTS = {
    # momentum (~46%)
    "dxy": 0.14,
    "brent": 0.08,
    "vale": 0.06,
    "vix": 0.10,
    "ibov": 0.08,
    # carry (~5%) — kept low so peak-detection signals can dominate
    "carry": 0.05,
    # mean-reversion / peak-detection (~39%)
    "level": 0.13,
    "rsi": 0.15,
    "bb": 0.11,
    # medium-term USD/BRL trend (~10%)
    "usdbrl_trend": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

SHORT_MA = 5
LONG_MA = 20
ADX_PERIOD = 14


# ── Technical Indicators ──────────────────────────────────────────────────────
def z_momentum(s: pd.Series, short: int = SHORT_MA, long: int = LONG_MA) -> float:
    """MA-crossover spread normalised by its own σ, clipped to [-1, 1]."""
    if len(s) < long + 1:
        return 0.0
    ma_short = s.rolling(short).mean()
    ma_long = s.rolling(long).mean()
    spread = (ma_short - ma_long) / ma_long
    std = spread.std()
    if std < 1e-9 or np.isnan(std):
        return 0.0
    return float(np.clip(float(spread.iloc[-1]) / std / 3.0, -1.0, 1.0))


def _rsi_value(s: pd.Series, period: int = 14) -> float:
    """Raw Wilder RSI in [0, 100]."""
    if len(s) < period + 1:
        return 50.0
    delta = s.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    last_loss = float(avg_loss.iloc[-1])
    if last_loss == 0:
        return 100.0
    rs = float(avg_gain.iloc[-1]) / last_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def rsi_score(s: pd.Series, period: int = 14) -> float:
    """RSI mapped to [-1, 1]: 70 → +1.0 (NOW/overbought), 30 → -1.0 (WAIT/oversold)."""
    return float(np.clip((_rsi_value(s, period) - 50.0) / 20.0, -1.0, 1.0))


def _pct_b(s: pd.Series, period: int = 20, mult: float = 2.0) -> float:
    """Raw Bollinger %B (can exceed [0, 1])."""
    if len(s) < period:
        return 0.5
    ma = s.rolling(period).mean()
    sigma = s.rolling(period).std()
    upper = ma + mult * sigma
    lower = ma - mult * sigma
    band = float(upper.iloc[-1]) - float(lower.iloc[-1])
    if band == 0 or np.isnan(band):
        return 0.5
    return float((float(s.iloc[-1]) - float(lower.iloc[-1])) / band)


def bb_score(s: pd.Series, period: int = 20, mult: float = 2.0) -> float:
    """Bollinger %B mapped to [-1, 1]."""
    return float(np.clip((_pct_b(s, period, mult) - 0.5) * 2.0, -1.0, 1.0))


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[float, float, float]:
    """Wilder ADX. Returns (adx, plus_di, minus_di). Falls back to (0, 50, 50)."""
    if len(close) < period * 2 + 5:
        return (0.0, 50.0, 50.0)

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    dn_move = -low.diff()

    dm_plus = pd.Series(
        np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    dm_minus = pd.Series(
        np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0),
        index=close.index,
    )

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    sm_dm_plus = dm_plus.ewm(alpha=alpha, adjust=False).mean()
    sm_dm_minus = dm_minus.ewm(alpha=alpha, adjust=False).mean()

    safe_atr = atr.replace(0, np.nan)
    di_plus = 100.0 * sm_dm_plus / safe_atr
    di_minus = 100.0 * sm_dm_minus / safe_atr

    denom = (di_plus + di_minus).replace(0, np.nan)
    dx = 100.0 * (di_plus - di_minus).abs() / denom
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return (
        float(adx.iloc[-1]),
        float(di_plus.iloc[-1]),
        float(di_minus.iloc[-1]),
    )


def carry_score(carry_diff: pd.Series) -> float:
    """
    Carry signal using absolute level calibration + trend.
    carry_diff = SELIC − FFR (% per year).

    Absolute calibration anchored to economic reality:
      diff > 8 %  → strongly WAIT  (BRL very attractive, carry trade live)
      diff ~ 4 %  → neutral
      diff < 0 %  → strongly NOW   (carry gone, unwind risk)

    Trend: narrowing differential → NOW, widening → WAIT.
    """
    if len(carry_diff) < LONG_MA:
        return 0.0

    cur = float(carry_diff.iloc[-1])

    # Level: anchor at 3.5 % neutral — any meaningful positive carry is WAIT.
    # 11 % → -1.0 (strongly WAIT)   5.5 % → -0.4 (moderate WAIT)
    #  3.5 % → 0 (neutral)          1 % → +0.5 (mild NOW — carry almost gone)
    level_score = float(np.clip(-(cur - 3.5) / 5.0, -1.0, 1.0))

    # Trend: widening differential → more attractive → WAIT (negate)
    z_trend = z_momentum(carry_diff)

    combined = 0.7 * level_score + 0.3 * (-z_trend)
    return float(np.clip(combined, -1.0, 1.0))


def regime_from_adx(adx: float, plus_di: float, minus_di: float) -> float:
    """
    Trend regime score in [-1, 1].
    0 when ranging (ADX < 20); otherwise signed by DI direction, scaled by strength.
    +1 = strong USD/BRL uptrend (BRL weakening), -1 = strong downtrend.
    """
    if adx < 20:
        return 0.0
    direction = 1.0 if plus_di > minus_di else -1.0
    strength = min(1.0, (adx - 20.0) / 25.0)
    return direction * strength


# ── Signal Dataclass ──────────────────────────────────────────────────────────
@dataclass
class Signal:
    name: str
    raw: float  # display value
    score: float  # [-1,1]: positive=NOW, negative=WAIT
    note: str
    weight: float


# ── Build Signals ─────────────────────────────────────────────────────────────
def build_signals(
    data: dict,
    carry_diff: "pd.Series | None" = None,
) -> "tuple[list[Signal], float]":
    """
    Compute all signals from price series.

    Parameters
    ----------
    data        : dict with keys usdbrl, dxy, brent, vale, vix, ibov
                  (usdbrl_high / usdbrl_low optional — used for ADX)
    carry_diff  : SELIC − FFR daily series (optional)

    Returns
    -------
    (signals, regime_score)
    """
    usdbrl = data["usdbrl"]
    usdbrl_high = data.get("usdbrl_high")
    usdbrl_low = data.get("usdbrl_low")

    # ── Regime detection ──────────────────────────────────────────────────────
    if usdbrl_high is not None and usdbrl_low is not None:
        adx_val, plus_di, minus_di = compute_adx(usdbrl_high, usdbrl_low, usdbrl)
    else:
        adx_val, plus_di, minus_di = 0.0, 50.0, 50.0

    regime = regime_from_adx(adx_val, plus_di, minus_di)
    ranging_w = float(np.clip(1.0 - max(0.0, adx_val - 20.0) / 5.0 * 0.7, 0.3, 1.0))

    signals: list[Signal] = []

    # ── 1. DXY ────────────────────────────────────────────────────────────────
    dxy = data["dxy"]
    dxy_cur = float(dxy.iloc[-1])
    dxy_ma = float(dxy.rolling(LONG_MA).mean().iloc[-1])
    dxy_dir = "↑" if dxy_cur > dxy_ma else ("↓" if dxy_cur < dxy_ma else "→")
    signals.append(
        Signal(
            name="DXY",
            raw=dxy_cur,
            score=-z_momentum(dxy),
            note=f"{dxy_cur:.2f} vs MA20 {dxy_ma:.2f} {dxy_dir}",
            weight=WEIGHTS["dxy"],
        )
    )

    # ── 2. Brent ──────────────────────────────────────────────────────────────
    brent = data["brent"]
    brent_cur = float(brent.iloc[-1])
    brent_ma = float(brent.rolling(LONG_MA).mean().iloc[-1])
    brent_dir = "↑" if brent_cur > brent_ma else ("↓" if brent_cur < brent_ma else "→")
    signals.append(
        Signal(
            name="Brent",
            raw=brent_cur,
            score=z_momentum(brent),
            note=f"${brent_cur:.2f} vs MA20 ${brent_ma:.2f} {brent_dir}",
            weight=WEIGHTS["brent"],
        )
    )

    # ── 3. VALE ───────────────────────────────────────────────────────────────
    vale = data["vale"]
    vale_cur = float(vale.iloc[-1])
    vale_ma = float(vale.rolling(LONG_MA).mean().iloc[-1])
    vale_dir = "↑" if vale_cur > vale_ma else ("↓" if vale_cur < vale_ma else "→")
    signals.append(
        Signal(
            name="VALE",
            raw=vale_cur,
            score=z_momentum(vale),
            note=f"${vale_cur:.2f} vs MA20 ${vale_ma:.2f} {vale_dir}",
            weight=WEIGHTS["vale"],
        )
    )

    # ── 4. VIX ────────────────────────────────────────────────────────────────
    vix = data["vix"]
    vix_cur = float(vix.iloc[-1])
    vix_level = -(vix_cur - 18.0) / 8.0
    vix_trend = z_momentum(vix)
    vix_score = float(np.clip(0.5 * vix_level - 0.5 * vix_trend, -1.0, 1.0))
    vix_label = (
        "panic"
        if vix_cur > 35
        else "fear"
        if vix_cur > 25
        else "elevated"
        if vix_cur > 18
        else "calm"
    )
    signals.append(
        Signal(
            name="VIX",
            raw=vix_cur,
            score=vix_score,
            note=f"{vix_cur:.2f}  {vix_label}",
            weight=WEIGHTS["vix"],
        )
    )

    # ── 5. IBOV ───────────────────────────────────────────────────────────────
    ibov = data["ibov"]
    ibov_cur = float(ibov.iloc[-1])
    ibov_ma = float(ibov.rolling(LONG_MA).mean().iloc[-1])
    ibov_dir = "↑" if ibov_cur > ibov_ma else ("↓" if ibov_cur < ibov_ma else "→")
    signals.append(
        Signal(
            name="IBOV",
            raw=ibov_cur,
            score=z_momentum(ibov),
            note=f"{ibov_cur:.0f} vs MA20 {ibov_ma:.0f} {ibov_dir}",
            weight=WEIGHTS["ibov"],
        )
    )

    # ── 6. Carry (optional) ───────────────────────────────────────────────────
    if carry_diff is not None and len(carry_diff) >= LONG_MA:
        carry_s = carry_score(carry_diff)
        carry_cur = float(carry_diff.iloc[-1])
        carry_tag = (
            "attractive→WAIT"
            if carry_s < -0.2
            else "eroding→NOW"
            if carry_s > 0.2
            else "neutral"
        )
        signals.append(
            Signal(
                name="Carry",
                raw=carry_cur,
                score=carry_s,
                note=f"{carry_cur:.1f}%/yr  {carry_tag}",
                weight=WEIGHTS["carry"],
            )
        )

    # ── 7. USD/BRL Level ──────────────────────────────────────────────────────
    usdbrl_cur = float(usdbrl.iloc[-1])
    pct = float((usdbrl <= usdbrl_cur).mean())
    base_score = float(np.clip((pct - 0.5) * 2.0, -1.0, 1.0))
    signals.append(
        Signal(
            name="USD/BRL Level",
            raw=usdbrl_cur,
            score=base_score * ranging_w,
            note=f"R${usdbrl_cur:.4f}  {pct * 100:.0f}th pct  ADX {adx_val:.1f}",
            weight=WEIGHTS["level"],
        )
    )

    # ── 8. RSI(14) ────────────────────────────────────────────────────────────
    raw_rsi = _rsi_value(usdbrl)
    rsi_state = (
        "overbought" if raw_rsi > 70 else "oversold" if raw_rsi < 30 else "neutral"
    )
    rsi_regime = "ranging" if ranging_w > 0.7 else "trending ↓"
    signals.append(
        Signal(
            name="RSI(14)",
            raw=raw_rsi,
            score=rsi_score(usdbrl) * ranging_w,
            note=f"{raw_rsi:.1f}  {rsi_state}  [{rsi_regime}]",
            weight=WEIGHTS["rsi"],
        )
    )

    # ── 9. Bollinger %B ───────────────────────────────────────────────────────
    raw_bb = _pct_b(usdbrl)
    bb_state = (
        "above upper"
        if raw_bb > 1.0
        else "below lower"
        if raw_bb < 0.0
        else "within bands"
    )
    bb_regime = "ranging" if ranging_w > 0.7 else "trending ↓"
    signals.append(
        Signal(
            name="Bollinger %B",
            raw=raw_bb,
            score=bb_score(usdbrl) * ranging_w,
            note=f"{raw_bb:.2f}  {bb_state}  [{bb_regime}]",
            weight=WEIGHTS["bb"],
        )
    )

    # ── 10. USD/BRL Medium-term Trend (20d / 60d MA crossover) ───────────────
    # Slope of 30d MA over the last 10 trading days.
    # Detects trend reversals ~6 weeks faster than the 20/60d MA crossover:
    # the 20/60d crossover still reads "downtrend" while the rate is already
    # rising (as seen in the Jan-Apr 2024 wrong NOW calls).
    # Rising slope  → WAIT (rate has been climbing, better rate ahead)
    # Falling slope → NOW  (rate declining, lock in while it's still high)
    trend_score = 0.0
    trend_note = "→ flat  [30d MA slope]"
    ma30 = usdbrl.rolling(30).mean().dropna()
    if len(ma30) >= 12:
        ma30_now = float(ma30.iloc[-1])
        ma30_past = float(ma30.iloc[-11])  # 10 trading days ago
        slope = (ma30_now - ma30_past) / ma30_past if ma30_past > 0 else 0.0
        # Normalise: 1 % move in 30d MA over 10 days ≈ strong signal
        trend_score = float(
            np.clip(-slope / 0.01, -1.0, 1.0)
        )  # rising → negative (WAIT)
        slope_pct = slope * 100
        if slope > 0.001:
            trend_note = f"↑ rising slope → WAIT  [{slope_pct:+.2f}% / 10d]"
        elif slope < -0.001:
            trend_note = f"↓ falling slope → NOW  [{slope_pct:+.2f}% / 10d]"
        else:
            trend_note = f"→ flat  [{slope_pct:+.2f}% / 10d]"
    signals.append(
        Signal(
            name="USD/BRL Trend",
            raw=float(usdbrl.iloc[-1]),
            score=trend_score,
            note=trend_note,
            weight=WEIGHTS["usdbrl_trend"],
        )
    )

    return signals, regime
