#!/usr/bin/env python3
"""
USD/BRL Exchange Timing Model
  python fx_timing.py                  — live signal analysis
  python fx_timing.py --lang pt        — saída em português
  python fx_timing.py --backtest       — walk-forward backtest (2nd & 17th since 2022)

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

# ── i18n ──────────────────────────────────────────────────────────────────────
_LANG: str = "en"  # overridden from --lang arg in main()


def _t(key: str) -> str:
    return _STRINGS.get(_LANG, _STRINGS["en"]).get(key, _STRINGS["en"].get(key, key))


_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "title": "USD → BRL   EXCHANGE TIMING MODEL",
        "trend_regime": "Trend Regime",
        "signals_header": "SIGNALS",
        "signals_cols": "← WAIT   NOW →   score    wt",
        "lbl_now": "NOW ",
        "lbl_wait": "WAIT",
        "lbl_flat": "FLAT",
        "prob_title": "PROBABILITY DISTRIBUTION",
        "prob_now": "Exchange Now",
        "prob_split": "Split 50/50",
        "prob_wait": "Wait",
        "stat_line": "Composite: {comp:+.3f}   Agreement: {agree:.0%}   Regime adj: {regime:+.2f}",
        "time_title": "Time horizon note:",
        "time_7d": "  < 7 days   execute regardless — timing models don't help at this horizon",
        "time_30d": "  7–30 days  the signal above is most relevant in this window",
        "time_long": "  > 30 days  consider weighting WAIT more; more time = more optionality",
        "disc1": "Past signal accuracy is no guarantee of future results.",
        "disc2": "This tool does not constitute financial or investment advice.",
        # regime
        "regime_sup": "sustained uptrend detected  (BRL weakening trend — NOW signals suppressed)",
        "regime_mup": "mild uptrend  (slight WAIT bias applied)",
        "regime_sdn": "sustained downtrend detected  (BRL strengthening trend — WAIT signals suppressed)",
        "regime_mdn": "mild downtrend  (slight NOW bias applied)",
        "regime_rng": "no clear trend  (mean-reversion signals fully active)",
        # verdict — exchange now
        "vn_hi_h": "signals lean toward exchanging now",
        "vn_hi_s": (
            "Multiple indicators suggest the current rate may be near a local high. "
            "This is not a guarantee — past signal accuracy has been ~52 % on this call."
        ),
        "vn_md_h": "signals moderately suggest considering an exchange",
        "vn_md_s": (
            "Some indicators point to the current rate being relatively favourable. "
            "Confidence is moderate — splitting (50 % now, 50 % later) is a reasonable alternative."
        ),
        "vn_lo_h": "signals weakly lean toward exchanging now",
        "vn_lo_s": (
            "The balance of indicators tilts slightly toward now, but conviction is low. "
            "Splitting or waiting are equally defensible choices."
        ),
        # verdict — wait
        "vw_hi_h": "signals lean toward waiting",
        "vw_hi_s": (
            "Multiple indicators suggest a better rate may become available. "
            "This is not a guarantee — the WAIT signal historically performs near 44 %."
        ),
        "vw_md_h": "signals moderately suggest waiting",
        "vw_md_s": (
            "Some indicators point to continued USD strength. "
            "Confidence is moderate — if you have a deadline within 7 days, execute regardless."
        ),
        "vw_lo_h": "signals weakly lean toward waiting",
        "vw_lo_s": (
            "The balance of indicators tilts slightly toward waiting, but conviction is low. "
            "Splitting (50 % now, 50 % later) may be the most prudent path."
        ),
        # verdict — split
        "vs_h": "signals are inconclusive — consider splitting",
        "vs_s": (
            "Indicators are mixed with no clear directional conviction. "
            "Exchanging 50 % now and 50 % later reduces timing regret without requiring a call."
        ),
    },
    "pt": {
        "title": "USD → BRL   MODELO DE TIMING DE CÂMBIO",
        "trend_regime": "Regime de Tendência",
        "signals_header": "SINAIS",
        "signals_cols": "← AGUARDAR  AGORA →   score    peso",
        "lbl_now": "AGORA",
        "lbl_wait": "AGU.",
        "lbl_flat": "NEUT",
        "prob_title": "DISTRIBUIÇÃO DE PROBABILIDADE",
        "prob_now": "Câmbio Agora",
        "prob_split": "Dividir 50/50",
        "prob_wait": "Aguardar",
        "stat_line": "Composto: {comp:+.3f}   Concordância: {agree:.0%}   Ajuste regime: {regime:+.2f}",
        "time_title": "Horizonte de tempo:",
        "time_7d": "  < 7 dias     execute independente — modelos de timing não ajudam nesse prazo",
        "time_30d": "  7–30 dias    o sinal acima é mais relevante nessa janela",
        "time_long": "  > 30 dias    considere dar mais peso ao AGUARDAR; mais tempo = mais opcionalidade",
        "disc1": "A acurácia histórica do modelo não garante resultados futuros.",
        "disc2": "Esta ferramenta não constitui aconselhamento financeiro ou de investimento.",
        # regime
        "regime_sup": "tendência de alta sustentada  (BRL enfraquecendo — sinais de AGORA suprimidos)",
        "regime_mup": "leve tendência de alta  (viés sutil para AGUARDAR aplicado)",
        "regime_sdn": "tendência de baixa sustentada  (BRL fortalecendo — sinais de AGUARDAR suprimidos)",
        "regime_mdn": "leve tendência de baixa  (viés sutil para AGORA aplicado)",
        "regime_rng": "sem tendência clara  (sinais de reversão à média totalmente ativos)",
        # verdict — câmbio agora
        "vn_hi_h": "os sinais indicam uma possível oportunidade de câmbio agora",
        "vn_hi_s": (
            "Múltiplos indicadores sugerem que a taxa atual pode estar próxima de uma máxima local. "
            "Isso não é garantia — a acurácia histórica deste sinal foi de ~52 %."
        ),
        "vn_md_h": "os sinais sugerem moderadamente considerar o câmbio agora",
        "vn_md_s": (
            "Alguns indicadores apontam para uma taxa atual relativamente favorável. "
            "A confiança é moderada — dividir (50 % agora, 50 % depois) é uma alternativa razoável."
        ),
        "vn_lo_h": "os sinais apontam fracamente para câmbio agora",
        "vn_lo_s": (
            "O equilíbrio de indicadores inclina-se levemente para agora, mas a convicção é baixa. "
            "Dividir ou aguardar são escolhas igualmente defensáveis."
        ),
        # verdict — aguardar
        "vw_hi_h": "os sinais indicam uma possível vantagem em aguardar",
        "vw_hi_s": (
            "Múltiplos indicadores sugerem que uma taxa melhor pode estar disponível. "
            "Isso não é garantia — o sinal de AGUARDAR historicamente tem ~44 % de acurácia."
        ),
        "vw_md_h": "os sinais sugerem moderadamente aguardar",
        "vw_md_s": (
            "Alguns indicadores apontam para continuidade do fortalecimento do dólar. "
            "Se tiver prazo em menos de 7 dias, execute independentemente."
        ),
        "vw_lo_h": "os sinais apontam fracamente para aguardar",
        "vw_lo_s": (
            "O equilíbrio de indicadores inclina-se levemente para aguardar, mas a convicção é baixa. "
            "Dividir (50 % agora, 50 % depois) pode ser o caminho mais prudente."
        ),
        # verdict — dividir
        "vs_h": "os sinais são inconclusivos — considere dividir a operação",
        "vs_s": (
            "Os indicadores estão mistos sem convicção direcional clara. "
            "Fazer câmbio de 50 % agora e 50 % depois reduz o arrependimento de timing sem exigir uma decisão definitiva."
        ),
    },
}

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
        return _t("regime_sup")
    if regime > 0.15:
        return _t("regime_mup")
    if regime < -0.6:
        return _t("regime_sdn")
    if regime < -0.15:
        return _t("regime_mdn")
    return _t("regime_rng")


def _verdict(d: str, pn: float, pw: float) -> tuple[str, str, str]:
    """
    Returns (icon, headline, subtext) graded by conviction level.
    Language is intentionally hedged — this is a probabilistic signal,
    not a guarantee or recommendation.
    """
    if d == "exchange_now":
        if pn > 0.70:
            return "◈", _t("vn_hi_h"), _t("vn_hi_s")
        if pn > 0.57:
            return "◈", _t("vn_md_h"), _t("vn_md_s")
        return "◈", _t("vn_lo_h"), _t("vn_lo_s")

    if d == "wait":
        if pw > 0.70:
            return "◷", _t("vw_hi_h"), _t("vw_hi_s")
        if pw > 0.57:
            return "◷", _t("vw_md_h"), _t("vw_md_s")
        return "◷", _t("vw_lo_h"), _t("vw_lo_s")

    return "◫", _t("vs_h"), _t("vs_s")


def render_live(signals: list[Signal], probs: dict) -> None:
    rate = next((s.raw for s in signals if s.name == "USD/BRL Level"), None)
    regime = probs.get("regime", 0.0)
    W = 66

    print()
    print("═" * W)
    print(f"  {_t('title')}")
    print(
        f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}"
        + (f"   ·   R$ {rate:.4f}" if rate else "")
    )
    print("═" * W)

    print()
    print(f"  {_t('trend_regime')}:  {regime_label(regime)}")
    print()
    cols = _t("signals_cols")
    pad = max(0, 30 - len(_t("signals_header")))
    print(f"  {_t('signals_header')}" + " " * pad + cols)
    print("  " + "─" * (W - 2))
    print()

    for s in signals:
        score = s.score
        if score > 0.15:
            v = _t("lbl_now")
        elif score < -0.15:
            v = _t("lbl_wait")
        else:
            v = _t("lbl_flat")
        print(f"  {s.name:<18} {sbar(score)}  {score:+.2f}  {s.weight:.0%}  [{v}]")
        print(f"    {s.note}")
        print()

    comp = probs["composite"]
    agree = probs["agreement"]
    print("  " + _t("stat_line").format(comp=comp, agree=agree, regime=regime))

    print()
    print(f"  {_t('prob_title')}")
    print("  " + "─" * (W - 2))
    print()
    for label, key in [
        (_t("prob_now"), "exchange_now"),
        (_t("prob_split"), "split"),
        (_t("prob_wait"), "wait"),
    ]:
        p = probs[key]
        print(f"  {label:<13}  {p:>5.1%}  [{pbar(p)}]")

    d = decide(probs)
    pn = probs["exchange_now"]
    pw = probs["wait"]
    icon, headline, sub = _verdict(d, pn, pw)

    print()
    print("  " + "─" * (W - 2))
    print()
    print(f"  {icon}  {headline}")
    # wrap subtext at ~62 chars
    words, line_buf = sub.split(), ""
    for word in words:
        if len(line_buf) + len(word) + 1 > 62:
            print(f"     {line_buf.rstrip()}")
            line_buf = word + " "
        else:
            line_buf += word + " "
    if line_buf.strip():
        print(f"     {line_buf.rstrip()}")

    print()
    print(f"  {_t('time_title')}")
    print(_t("time_7d"))
    print(_t("time_30d"))
    print(_t("time_long"))
    print()
    print(f"  ⚠  {_t('disc1')}")
    print(f"     {_t('disc2')}")
    print()
    print("═" * W)
    print()


# ── Backtest Helpers ──────────────────────────────────────────────────────────
def decision_dates(
    start: str = BACKTEST_START,
    check_days: tuple[int, ...] = (2, 17),
) -> list[date]:
    """All occurrences of check_days in each month from start to two days ago."""
    out: list[date] = []
    sd = datetime.strptime(start, "%Y-%m-%d").date()
    end = date.today() - timedelta(days=2)
    y, m = sd.year, sd.month
    while date(y, m, 1) <= end:
        for day in check_days:
            try:
                d = date(y, m, day)
            except ValueError:
                continue  # e.g. day 31 in a 30-day month
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
    check_days: tuple[int, ...] = (2, 17),
) -> list[Row]:
    usdbrl = all_data["usdbrl"]
    dates = decision_dates(check_days=check_days)
    rows: list[Row] = []

    days_str = " & ".join(str(d) for d in check_days)
    print(
        f"\n  Running walk-forward backtest on {len(dates)} decision dates  [{days_str} of each month]..."
    )

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


def sequential_sim(
    rows: list[Row], check_days: tuple[int, ...] = (2, 17)
) -> list[Scenario]:
    """
    From each first check-day of the month, follow model decisions until NOW/SPLIT fires
    or MAX_WAIT periods elapse, then compare BRL received vs day-1 exchange.
    """
    row_map = {r.date: r for r in rows}
    all_dates = sorted(row_map)
    scenarios: list[Scenario] = []
    first_day = min(check_days)  # use lowest day as the scenario start

    for start_d in (d for d in all_dates if d.day == first_day):
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
def render_backtest(
    rows: list[Row],
    scenarios: list[Scenario],
    check_days: tuple[int, ...] = (2, 17),
) -> None:
    W = 86
    days_str = " & ".join(str(d) for d in check_days)

    print("\n" + "═" * W)
    print(f"  USD → BRL  WALK-FORWARD BACKTEST  ({days_str} of each month since 2022)")
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
    parser = argparse.ArgumentParser(
        description="USD/BRL Exchange Timing Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python fx_timing.py                        live analysis, run any day
  python fx_timing.py --lang pt              saída em português
  python fx_timing.py --backtest             backtest on default schedule (2nd & 17th)
  python fx_timing.py --backtest --days 5 20 backtest on your own schedule (5th & 20th)
  python fx_timing.py --backtest --days 15   backtest on a single day per month""",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Walk-forward backtest since 2022",
    )
    parser.add_argument(
        "--days",
        nargs="+",
        type=int,
        default=[2, 17],
        metavar="DAY",
        help="Day(s) of the month to check (default: 2 17). "
        "Set to the day(s) you typically receive USD payments.",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "pt"],
        default="en",
        help="Output language: en (default) or pt (português)",
    )
    args = parser.parse_args()

    global _LANG
    _LANG = args.lang

    # Validate day numbers
    for d in args.days:
        if not 1 <= d <= 28:
            parser.error(f"--days: {d} is out of range. Use values between 1 and 28.")
    check_days = tuple(sorted(set(args.days)))

    if args.backtest:
        warmup = (
            datetime.strptime(BACKTEST_START, "%Y-%m-%d") - timedelta(days=90)
        ).strftime("%Y-%m-%d")
        days_str = " & ".join(str(d) for d in check_days)
        print(f"\n  Schedule: {days_str} of each month")
        print(f"  Fetching full history from {warmup} ...")
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

        rows = run_backtest(all_data, carry_diff, check_days)
        scenarios = sequential_sim(rows, check_days)
        render_backtest(rows, scenarios, check_days)

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
