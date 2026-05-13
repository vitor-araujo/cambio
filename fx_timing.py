#!/usr/bin/env python3
"""
USD/BRL Exchange Timing Model
  python fx_timing.py                  — live signal analysis
  python fx_timing.py --lang pt        — saída em português
  python fx_timing.py --notify         — HTML alert in the browser on flip-to-NOW
  python fx_timing.py --watch --notify — background mode with auto-alerts
  python fx_timing.py --backtest       — walk-forward backtest (2nd & 17th since 2022)

pip install yfinance pandas numpy
"""

__version__ = "0.4.3"

import argparse
import io
import json
import os
import time
import warnings
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

import journal
import notify
import rate_alert
from notifiers import DEFAULT_PROVIDER, ENV_VAR, available, get_notifier
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
DEFAULT_DEADLINE_DAYS = 15  # forced execution window — 2 decisions/month avg
AMOUNT = 10_000  # USD per scenario
REGIME_STRENGTH = 0.55  # how hard the regime filter shifts probabilities
DEFAULT_SPREAD_BPS = 50  # 0.50 % effective spread (Wise-class fintech, pre-IOF)
DEFAULT_USER_NAME = "Vitor"  # used in the HTML alert headline
WATCH_INTERVAL_MIN = 5  # minutes between checks in --watch mode
NOTIFY_THRESHOLD = 0.55  # min p_now to trigger browser alert
NOTIFY_COOLDOWN_HOURS = 6  # don't re-open the alert within this window

# ── Vanguard discipline (DCA + CMH) ───────────────────────────────────────────────────────
DEFAULT_DCA_FLOOR = 0.25  # minimum fraction to convert every cycle (regret cap)
DEFAULT_DCA_CEILING = 1.00  # maximum fraction (only fires on high-conviction NOW)
SIZE_PIVOT_LOW = 0.30  # p_now ≤ this → floor only
SIZE_PIVOT_HIGH = 0.70  # p_now ≥ this → ceiling
CMH_SAFETY_MULTIPLIER = 2.0  # edge must clear 2× spread to be "real" (Bogle)

TICKERS = {
    # usdbrl is intentionally omitted — fetched from BCB PTAX instead of Yahoo
    "dxy": "DX-Y.NYB",
    "brent": "BZ=F",
    "vale": "VALE",
    "vix": "^VIX",
    "ibov": "^BVSP",
    "us_rate": "^IRX",  # 13-week T-bill (FFR proxy, % per year)
    "six_l": "6L=F",  # CME BRL/USD futures — institutional positioning signal
}

BCB_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"
    "?formato=json&dataInicial={start}"
)

# Tracks whether the current run is using BCB PTAX (True) or Yahoo BRL=X (False)
_PTAX_SOURCE: bool = False
# Date of the last available PTAX bulletin (may be yesterday outside trading hours)
_PTAX_DATE: Optional[date] = None

# ── CFTC COT (Commitment of Traders) ─────────────────────────────────────────
COT_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
COT_CACHE = ".cot_cache.csv"  # local weekly cache — listed in .gitignore
COT_MARKET = "EURO FX - CHICAGO MERCANTILE EXCHANGE"  # 57 % of DXY; best free USD proxy

BCB_PTAX_URL = (
    "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
    "CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
    "?@dataInicial='{start}'&@dataFinalCotacao='{end}'"
    "&$top=15000&$format=json&$select=cotacaoVenda,dataHoraCotacao"
)


# ── Data Fetching ─────────────────────────────────────────────────────────────
def fetch_ptax(start: str, end: Optional[datetime] = None) -> Optional[pd.Series]:
    """
    Fetch USD/BRL PTAX (cotação de venda) from BCB PTAX API.

    PTAX is the official commercial rate published by the Banco Central do Brasil
    several times per day. This is the rate used for bank transfers and fintechs
    (Wise, Remessa Online, etc.) — NOT the tourism rate (dólar turismo), which
    is typically 8–15 % higher.

    Returns one rate per business day (last bulletin of each day).
    Falls back to None if unavailable.
    """
    try:
        end = end or datetime.now()
        start_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%m-%d-%Y")
        end_fmt = end.strftime("%m-%d-%Y")

        url = BCB_PTAX_URL.format(start=start_fmt, end=end_fmt)
        req = Request(url, headers={"User-Agent": "cambio/1.0"})
        raw = json.loads(urlopen(req, timeout=15).read())
        records = raw.get("value", [])

        if not records:
            return None

        # Multiple bulletins per day — keep the last one (most recent PTAX)
        by_date: dict[str, float] = {}
        for r in records:
            date_str = r["dataHoraCotacao"][:10]  # "2022-01-03"
            by_date[date_str] = float(r["cotacaoVenda"])

        # CotacaoDolarPeriodo lags by 1 day — patch with today via CotacaoDolarDia
        # (only returns data during BCB trading hours: 10h–14h Brasília)
        try:
            today_str = date.today().strftime("%m-%d-%Y")
            today_url = (
                "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
                f"CotacaoDolarDia(dataCotacao=@dataCotacao)?@dataCotacao='{today_str}'"
                "&$top=10&$format=json&$select=cotacaoVenda,dataHoraCotacao"
            )
            today_raw = json.loads(
                urlopen(
                    Request(today_url, headers={"User-Agent": "cambio/1.0"}), timeout=10
                ).read()
            )
            for r in today_raw.get("value", []):
                by_date[r["dataHoraCotacao"][:10]] = float(r["cotacaoVenda"])
        except Exception:
            pass  # market closed / outside hours — period data is sufficient

        dates = [pd.Timestamp(d) for d in sorted(by_date)]
        values = [by_date[d.strftime("%Y-%m-%d")] for d in dates]

        s = pd.Series(values, index=pd.DatetimeIndex(dates), name="ptax").dropna()
        # Store the date of the last available bulletin for display
        global _PTAX_DATE
        if not s.empty:
            _PTAX_DATE = s.index[-1].date()
        return s

    except Exception as e:
        print(f"  ⚠  PTAX API unavailable ({e}) — falling back to Yahoo BRL=X")
        return None


def fetch(start: str, end: Optional[datetime] = None) -> dict[str, pd.Series]:
    """
    Fetch Close series for macro/technical tickers from Yahoo Finance.
    USD/BRL is sourced from BCB PTAX (commercial rate) with Yahoo BRL=X fallback.
    PTAX does not provide H/L data — ADX will return zero (ranging mode).
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

        except Exception:
            pass

    # USD/BRL: prefer PTAX (commercial rate) over Yahoo BRL=X (includes spread/markup)
    global _PTAX_SOURCE
    ptax = fetch_ptax(start, end)
    if ptax is not None and len(ptax) > 20:
        out["usdbrl"] = ptax
        _PTAX_SOURCE = True
        # No H/L from PTAX — ADX defaults to zero (ranging, no regime filter)
    else:
        _PTAX_SOURCE = False
        # Fallback: Yahoo BRL=X with OHLC for ADX
        try:
            df = yf.download(
                "BRL=X",
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            )
            if not df.empty:
                out["usdbrl"] = df["Close"].squeeze().dropna()
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


def _load_cot_cache() -> pd.DataFrame:
    """Load COT cache if it exists and is less than 7 days old."""
    if not os.path.exists(COT_CACHE):
        return pd.DataFrame()
    if (datetime.now().timestamp() - os.path.getmtime(COT_CACHE)) > 7 * 86400:
        return pd.DataFrame()
    try:
        return pd.read_csv(COT_CACHE, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()


def _fetch_cot_year(year: int) -> pd.DataFrame:
    """Download and parse one year of CFTC financial futures COT data."""
    url = COT_URL.format(year=year)
    req = Request(url, headers={"User-Agent": "cambio/1.0"})
    zip_bytes = urlopen(req, timeout=45).read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        txt = next(n for n in zf.namelist() if n.lower().endswith((".txt", ".csv")))
        with zf.open(txt) as f:
            return pd.read_csv(f, low_memory=False)


def fetch_cot_eur(start: str) -> Optional[pd.Series]:
    """
    Fetch CFTC Commitment of Traders — leveraged-money net position in EUR/USD futures.

    EUR/USD is the best free proxy for broad USD sentiment: EUR is 57 % of DXY and the
    CME EUR futures are the most liquid FX contract with COT data.

    Interpretation (from a USD/BRL perspective):
      Net LONG  EUR → hedge funds short USD → USD likely to weaken → USD/BRL down → NOW
      Net SHORT EUR → hedge funds long  USD → USD likely to strengthen → USD/BRL up → WAIT

    Data: weekly (published every Tuesday for the previous Friday).
    Cached locally for 7 days in COT_CACHE to avoid redundant downloads.
    Forward-filled to business-day frequency for walk-forward slicing.
    """
    cache = _load_cot_cache()

    if cache.empty:
        start_year = int(start[:4])
        current_year = datetime.now().year
        frames: list[pd.DataFrame] = []

        print("  Fetching CFTC COT data (USD sentiment, ~30 s)...")
        for year in range(start_year, current_year + 1):
            try:
                df = _fetch_cot_year(year)
                eur = df[
                    df["Market_and_Exchange_Names"]
                    .astype(str)
                    .str.contains(COT_MARKET, na=False)
                ].copy()
                if eur.empty:
                    continue

                eur = eur.sort_values("Report_Date_as_YYYY-MM-DD")
                eur["net"] = pd.to_numeric(
                    eur["Lev_Money_Positions_Long_All"], errors="coerce"
                ) - pd.to_numeric(eur["Lev_Money_Positions_Short_All"], errors="coerce")
                eur["oi"] = pd.to_numeric(eur["Open_Interest_All"], errors="coerce")
                eur.index = pd.DatetimeIndex(eur["Report_Date_as_YYYY-MM-DD"])
                frames.append(eur[["net", "oi"]].dropna())
                print(f"  ✓  COT {year}: {len(eur)} weeks")
            except Exception as exc:
                print(f"  ⚠  COT {year}: {exc}")

        if not frames:
            print("  ⚠  COT unavailable — USD sentiment signal disabled")
            return None

        cache = pd.concat(frames)
        cache = cache[~cache.index.duplicated(keep="last")].sort_index()
        try:
            cache.to_csv(COT_CACHE)
        except Exception:
            pass  # non-fatal if we can't write cache

    if cache.empty or "net" not in cache.columns:
        return None

    net = cache["net"].dropna()
    # Forward-fill weekly reports to business days for walk-forward slicing
    full_range = pd.date_range(net.index.min(), datetime.now(), freq="B")
    return net.reindex(full_range).ffill().dropna().rename("cot_eur_net")


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


def size(
    probs: dict,
    floor: float = DEFAULT_DCA_FLOOR,
    ceiling: float = DEFAULT_DCA_CEILING,
) -> float:
    """
    Vanguard-DCA sizing: never zero, never blindly all-in.

    Maps p_now linearly into [floor, ceiling] across the SIZE_PIVOT_LOW …
    SIZE_PIVOT_HIGH band. Outside the band the result is clipped, so even a
    strong WAIT still converts at the floor and a strong NOW caps at the
    ceiling. This minimises worst-case regret variance — the actual
    objective when your life depends on the outcome.
    """
    pn = float(probs["exchange_now"])
    span = SIZE_PIVOT_HIGH - SIZE_PIVOT_LOW
    if span <= 0:
        return float(np.clip(floor, 0.0, 1.0))
    norm = (pn - SIZE_PIVOT_LOW) / span
    norm = max(0.0, min(1.0, norm))
    return float(np.clip(floor + (ceiling - floor) * norm, 0.0, 1.0))


def decide(
    probs: dict,
    floor: float = DEFAULT_DCA_FLOOR,
    ceiling: float = DEFAULT_DCA_CEILING,
) -> str:
    """
    Categorical label derived from the sizing fraction:
      NOW   — size ≥ 0.70 (high conviction → convert most/all)
      SPLIT — 0.40 ≤ size < 0.70 (mixed signals, partial)
      WAIT  — size < 0.40 (floor-only, mandatory DCA)
    """
    s = size(probs, floor, ceiling)
    if s >= 0.70:
        return "exchange_now"
    if s >= 0.40:
        return "split"
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


def _apply_intraday(
    data: dict[str, pd.Series],
    live_fx: Optional[tuple[float, str]],
) -> dict[str, pd.Series]:
    """
    Replace / append today's USD/BRL bar with the live intraday tick so
    technical signals (Level, RSI, BB, Trend) reflect the rate the user is
    actually seeing on Higlobe right now — not yesterday's PTAX close.

    Idempotent: safe to call repeatedly inside the --watch loop.
    """
    if live_fx is None or "usdbrl" not in data:
        return data
    rate, _ = live_fx
    if rate <= 0 or not np.isfinite(rate):
        return data

    series = data["usdbrl"].copy()
    today = pd.Timestamp(date.today())
    if not series.empty and series.index[-1].normalize() == today:
        series.iloc[-1] = float(rate)
    else:
        series.loc[today] = float(rate)
        series = series.sort_index()
    data = {**data, "usdbrl": series}
    return data


def _fetch_live_fx() -> Optional[tuple[float, str]]:
    """
    Fetch the current intraday USD/BRL commercial rate.

    Priority:
      1. AwesomeAPI (economia.awesomeapi.com.br) — aggregates B3 broker feeds,
         ~0.06 % spread, updates every ~1 min during market hours. No API key.
      2. Yahoo Finance BRL=X — fallback, carries a tourist-rate markup (~0.5–1 %).

    Returns (rate, source_label) or None on total failure.
    """
    # 1 — AwesomeAPI (best free source for dólar comercial)
    try:
        url = "https://economia.awesomeapi.com.br/json/last/USD-BRL"
        req = Request(url, headers={"User-Agent": "cambio/1.0"})
        raw = json.loads(urlopen(req, timeout=8).read())
        bid = float(raw["USDBRL"]["bid"])
        updated = raw["USDBRL"].get("create_date", "")[:16]  # "2026-04-10 10:37"
        label = f"mercado {updated[11:]}" if updated else "mercado"  # show HH:MM
        return bid, label
    except Exception:
        pass

    # 2 — Yahoo Finance fallback
    try:
        df = yf.download("BRL=X", period="2d", progress=False, auto_adjust=True)
        if not df.empty:
            return float(df["Close"].iloc[-1]), "Yahoo FX"
    except Exception:
        pass

    return None


def render_live(
    signals: list[Signal],
    probs: dict,
    live_fx: Optional[tuple[float, str]] = None,
) -> None:
    rate = next((s.raw for s in signals if s.name == "USD/BRL Level"), None)
    regime = probs.get("regime", 0.0)
    W = 66

    # Build the rate line: live market rate (AwesomeAPI) + PTAX reference
    if live_fx:
        live_val, live_label = live_fx
        fx_str = f"R$ {live_val:.4f} ({live_label})"
        if _PTAX_SOURCE and rate:
            if _PTAX_DATE and _PTAX_DATE < date.today():
                ptax_str = f"PTAX {_PTAX_DATE.strftime('%d/%m')}: R$ {rate:.4f}"
            else:
                ptax_str = f"PTAX: R$ {rate:.4f}"
            rate_line = f"{fx_str}  ·  {ptax_str}"
        else:
            rate_line = fx_str
    elif rate:
        if _PTAX_SOURCE:
            if _PTAX_DATE and _PTAX_DATE < date.today():
                rate_line = f"R$ {rate:.4f}  (PTAX {_PTAX_DATE.strftime('%d/%m')} · fora do horário BCB)"
            else:
                rate_line = f"R$ {rate:.4f}  (PTAX comercial)"
        else:
            rate_line = f"R$ {rate:.4f}  (Yahoo FX)"
    else:
        rate_line = ""

    print()
    print("═" * W)
    print(f"  {_t('title')}")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}   ·   {rate_line}")
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


# ── Backtest Core ──────────────────────────────────────────────────────────────────────
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
    size_frac: float = DEFAULT_DCA_FLOOR


def run_backtest(
    all_data: dict[str, pd.Series],
    carry_diff: Optional[pd.Series],
    check_days: tuple[int, ...] = (2, 17),
    dca_floor: float = DEFAULT_DCA_FLOOR,
    dca_ceiling: float = DEFAULT_DCA_CEILING,
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
        dec = decide(probs, dca_floor, dca_ceiling)
        size_frac = size(probs, dca_floor, dca_ceiling)
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
                size_frac=size_frac,
            )
        )

    return rows


# ── Sequential Simulation ────────────────────────────────────────────────────────────
@dataclass
class Scenario:
    start_date: date
    execute_date: date  # date of the LAST partial conversion
    start_rate: float
    effective_rate: float  # USD-weighted average rate the model achieved
    periods_used: int  # number of partial conversions executed
    brl_model: float
    brl_immediate: float


def _max_wait_periods(check_days: tuple[int, ...], deadline_days: int) -> int:
    """
    Map a real-world deadline (in calendar days) to the number of subsequent
    check-points the model is allowed to wait. With the default 2nd & 17th
    schedule (period ≈ 15 days) and a 15-day deadline, this yields max_wait=1
    — i.e. 2 decisions per month on average.
    """
    period_days = max(1, 30 // max(1, len(check_days)))
    return max(1, deadline_days // period_days)


def sequential_sim(
    rows: list[Row],
    check_days: tuple[int, ...] = (2, 17),
    deadline_days: int = DEFAULT_DEADLINE_DAYS,
    spread_bps: int = DEFAULT_SPREAD_BPS,
) -> list[Scenario]:
    """
    Vanguard-DCA simulation. From each scenario start, walk the schedule and
    convert `size_frac` of the **remaining** USD at every check date. Whatever
    is left at the deadline is force-converted.

    Compares the resulting effective BRL against the lump-sum baseline
    (convert 100 % at start). Both legs pay the same per-USD spread, so the
    win/loss is purely the timing edge net of conversion friction.
    """
    max_wait = _max_wait_periods(check_days, deadline_days)
    fee = 1.0 - spread_bps / 10_000.0

    row_map = {r.date: r for r in rows}
    all_dates = sorted(row_map)
    scenarios: list[Scenario] = []
    first_day = min(check_days)  # use lowest day as the scenario start

    for start_d in (d for d in all_dates if d.day == first_day):
        pool = [d for d in all_dates if d >= start_d][: max_wait + 1]
        if not pool:
            continue

        start_r = row_map[start_d].rate
        remaining = float(AMOUNT)
        brl_received = 0.0
        usd_converted = 0.0
        last_exec_d = pool[0]
        periods_used = 0

        for i, d in enumerate(pool):
            r = row_map.get(d)
            if r is None:
                continue
            is_last = i == len(pool) - 1
            frac = 1.0 if is_last else r.size_frac
            chunk = remaining * frac
            if chunk <= 1e-6:
                continue
            brl_received += chunk * r.rate * fee
            usd_converted += chunk
            remaining -= chunk
            last_exec_d = d
            periods_used += 1
            if remaining <= 1e-6:
                break

        if usd_converted <= 0:
            continue

        # USD-weighted effective rate (pre-fee, for comparability with start_rate)
        effective_rate = brl_received / (usd_converted * fee)

        scenarios.append(
            Scenario(
                start_date=start_d,
                execute_date=last_exec_d,
                start_rate=start_r,
                effective_rate=effective_rate,
                periods_used=periods_used,
                brl_model=brl_received,
                brl_immediate=AMOUNT * start_r * fee,
            )
        )

    return scenarios


# ── Backtest Rendering ────────────────────────────────────────────────────────
def render_backtest(
    rows: list[Row],
    scenarios: list[Scenario],
    check_days: tuple[int, ...] = (2, 17),
    deadline_days: int = DEFAULT_DEADLINE_DAYS,
    spread_bps: int = DEFAULT_SPREAD_BPS,
) -> None:
    W = 86
    days_str = " & ".join(str(d) for d in check_days)
    max_wait = _max_wait_periods(check_days, deadline_days)

    print("\n" + "═" * W)
    print(f"  USD → BRL  WALK-FORWARD BACKTEST  ({days_str} of each month since 2022)")
    print("  Signals: DXY · Brent · VALE · VIX · IBOV · Carry · Level · RSI · BB%B")
    print("  Regime:  ADX(14) trend filter applied to final probabilities")
    print(
        f"  Deadline: {deadline_days}d (max {max_wait} wait period{'s' if max_wait != 1 else ''})  ·  Spread: {spread_bps / 100:.2f}%"
    )
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
        f"  {'Date':<12} {'Model':<7} {'Sz':>4} {'Oracle':<7} "
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
        sz = f"{r.size_frac:.0%}"
        ok = "✓" if r.correct else "✗"
        print(
            f"  {str(r.date):<12} {lbl[r.model]:<7} {sz:>4} {lbl[r.oracle]:<7}"
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
        f"  {'Start':<12} {'Last Exec':<12} {'R@Start':>8} {'R@Eff':>8}"
        f" {'Cuts':>5}  {'ΔBRL':>10}  {'Δ%':>7}  Result"
    )
    print("  " + "─" * (W - 2))

    for s in scenarios:
        g = s.brl_model - s.brl_immediate
        pct = (s.brl_model / s.brl_immediate - 1) * 100
        sign = "+" if g >= 0 else ""
        flag = "▲ win" if g > 50 else "▼ loss" if g < -50 else "  tie"
        print(
            f"  {str(s.start_date):<12} {str(s.execute_date):<12}"
            f" {s.start_rate:>8.4f} {s.effective_rate:>8.4f}"
            f" {str(s.periods_used) + '×':>5}"
            f"  {sign}{g:>8.0f}  {sign}{pct:>5.2f}%  {flag}"
        )

    print()
    print(
        f"  Scenarios: {len(scenarios)}   Wins: {wins}   Losses: {losses}   Ties: {len(scenarios) - wins - losses}"
    )
    avg_gain_pct = float(np.mean(pcts))
    print(f"  Avg BRL gain / scenario   R$ {np.mean(gains):>+8.0f}")
    print(f"  Avg return vs immediate       {avg_gain_pct:>+6.2f}%")
    print(f"  Cumulative BRL edge       R$ {sum(gains):>+8.0f}")
    print(f"  Win rate                      {wins / len(scenarios) * 100:>5.1f}%")

    # ── Cost-Matters Hypothesis (Bogle/Vanguard) edge validation ─────────────────────────
    spread_pct = spread_bps / 100.0
    safety_threshold = CMH_SAFETY_MULTIPLIER * spread_pct
    safety_margin = avg_gain_pct - safety_threshold
    real_edge = safety_margin > 0
    verdict = "REAL" if real_edge else "INSIDE THE SPREAD"
    badge = "✓" if real_edge else "⚠"

    print()
    print("  " + "─" * (W - 2))
    print(
        f"  COST-MATTERS HYPOTHESIS  ·  margin-of-safety vs {CMH_SAFETY_MULTIPLIER:g}× spread"
    )
    print("  " + "─" * (W - 2))
    print(f"  Avg model edge                  {avg_gain_pct:>+6.2f} %")
    print(
        f"  Cost threshold ({CMH_SAFETY_MULTIPLIER:g}× spread)        {safety_threshold:>6.2f} %"
    )
    print(
        f"  Margin of safety              {safety_margin:>+8.2f} %  {badge} edge is {verdict}"
    )
    if not real_edge:
        print(
            "  \u26a0  the alleged timing edge sits inside conversion friction \u2014 "
            "treat the model as decoration, not signal."
        )
    print()
    print("═" * W)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def _print_fetch_status(data: dict[str, pd.Series]) -> None:
    for k, v in data.items():
        if not k.endswith(("_high", "_low")):
            print(f"  ✓  {k:<12} — {len(v)} trading days")


def _build_journal_entry(
    probs: dict,
    decision: str,
    size_frac: float,
    rate_signal: float,
    rate_live: Optional[float],
    notified: bool,
) -> journal.Entry:
    return journal.Entry(
        ts=datetime.now(),
        rate_signal=rate_signal,
        rate_live=rate_live,
        decision=decision,
        p_now=probs["exchange_now"],
        p_split=probs["split"],
        p_wait=probs["wait"],
        composite=probs["composite"],
        agreement=probs["agreement"],
        regime=probs.get("regime", 0.0),
        notified=notified,
        size=size_frac,
    )


def _maybe_notify(
    sigs: list[Signal],
    probs: dict,
    decision: str,
    size_frac: float,
    deadline_remaining: Optional[int],
    live_fx: Optional[tuple[float, str]],
    user_name: str,
) -> bool:
    """Fire the browser alert when conviction is high and cooldown elapsed."""
    if not journal.should_notify(
        decision,
        probs["exchange_now"],
        threshold=NOTIFY_THRESHOLD,
        cooldown_hours=NOTIFY_COOLDOWN_HOURS,
    ):
        return False

    rate_for_alert = (
        live_fx[0]
        if live_fx
        else next((s.raw for s in sigs if s.name == "USD/BRL Level"), 0.0)
    )
    top = sorted(sigs, key=lambda s: abs(s.score) * s.weight, reverse=True)[:5]
    top_signals = [(s.name, s.score) for s in top]

    return notify.alert(
        name=user_name,
        rate_live=float(rate_for_alert),
        p_now=probs["exchange_now"],
        composite=probs["composite"],
        agreement=probs["agreement"],
        regime=probs.get("regime", 0.0),
        top_signals=top_signals,
        convert_pct=size_frac,
        deadline_remaining=deadline_remaining,
    )


def _run_live_cycle(
    args,
    *,
    render: bool = True,
    refetch: bool = True,
    cache: Optional[dict] = None,
) -> dict:
    """
    Single live-analysis pass. Returns a dict with the cycle outcome so the
    --watch loop can reuse fetched data on subsequent ticks.
    """
    if refetch or cache is None:
        start = (datetime.now() - timedelta(days=LIVE_FETCH_DAYS)).strftime("%Y-%m-%d")
        if render:
            print("\n  Fetching market data...")
        data = fetch(start)
        if not data:
            print("  ERROR: no data fetched — check connectivity.\n")
            return {"ok": False}
        if render:
            _print_fetch_status(data)
            print("  Fetching SELIC from BCB ...")
        selic = fetch_selic(start)
        carry_diff = build_carry_diff(selic, data.get("us_rate"))
        cot_eur = fetch_cot_eur(start)
        if cot_eur is not None:
            data["cot_eur"] = cot_eur
        cache = {"data": data, "carry_diff": carry_diff}
    else:
        data = cache["data"]
        carry_diff = cache["carry_diff"]

    # Always refresh the live tick — cheap, ~50 ms
    live_fx = _fetch_live_fx()
    data_live = _apply_intraday(data, live_fx)

    result = build_signals(data_live, carry_diff)
    sigs, regime = result if isinstance(result, tuple) else (result, 0.0)

    probs = probabilities(sigs)
    probs = apply_regime(probs, regime)
    floor = getattr(args, "dca_floor", DEFAULT_DCA_FLOOR)
    ceiling = getattr(args, "dca_ceiling", DEFAULT_DCA_CEILING)
    decision = decide(probs, floor, ceiling)
    size_frac = size(probs, floor, ceiling)

    rate_signal = float(data_live["usdbrl"].iloc[-1])
    rate_live = float(live_fx[0]) if live_fx else None

    deadline_days = getattr(args, "deadline_days", DEFAULT_DEADLINE_DAYS)
    deadline_remaining = journal.days_until_deadline(deadline_days)

    if render:
        prev = journal.last_entry()
        summary = journal.render_summary(prev, rate_live or rate_signal)
        if summary:
            print(f"\n  {summary}")
        if deadline_remaining is not None:
            usd_chunk = AMOUNT * size_frac
            print(
                f"  prazo: {deadline_remaining} dias até execução forçada  ·  "
                f"sugestão agora: converter {size_frac:.0%} (≈ ${usd_chunk:,.0f})"
            )
        else:
            print(
                f"  sugestão agora: converter {size_frac:.0%}  ·  "
                "use --mark-executed após o câmbio para ativar o cronômetro de prazo"
            )
        render_live(sigs, probs, live_fx=live_fx)

    notified = False
    if args.notify:
        notified = _maybe_notify(
            sigs, probs, decision, size_frac, deadline_remaining, live_fx, args.name
        )
        if notified and render:
            print("  ◈  alerta aberto no navegador — abre Higlobe e converte agora.\n")

    alert_result: Optional[dict] = None
    if getattr(args, "phone_alerts", False) and rate_live is not None:
        try:
            notifier = get_notifier(args.alert_provider)
        except ValueError as e:
            if render:
                print(f"  ⚠  provedor inválido: {e}")
            notifier = None
        if notifier is not None:
            alert_result = rate_alert.maybe_alert_on_rise(
                rate_live,
                notifier=notifier,
                threshold_pct=args.alert_threshold,
                cooldown_min=args.alert_cooldown,
            )
            if alert_result["fired"] and render:
                print(
                    f"  📱  alerta enviado via {alert_result['provider']} · "
                    f"R$ {rate_live:.4f} ({alert_result['delta_pct']:+.2f}% vs âncora)\n"
                )
            elif alert_result.get("error") and render:
                print(f"  ⚠  alerta falhou: {alert_result['error']}")

    journal.append(
        _build_journal_entry(
            probs, decision, size_frac, rate_signal, rate_live, notified
        )
    )

    return {
        "ok": True,
        "cache": cache,
        "decision": decision,
        "size": size_frac,
        "probs": probs,
        "rate_live": rate_live,
        "notified": notified,
        "deadline_remaining": deadline_remaining,
        "alert": alert_result,
    }


def _watch_loop(args) -> None:
    """Background mode: refresh signals on a schedule, fire alerts on flip-to-NOW."""
    interval = max(5, args.watch_interval) * 60  # seconds
    print(
        f"\n  ◉  watch mode · a cada {args.watch_interval} min · "
        f"alerta abre Higlobe quando p(agora) ≥ {NOTIFY_THRESHOLD:.0%}"
    )
    if getattr(args, "phone_alerts", False):
        try:
            notifier = get_notifier(args.alert_provider)
            if notifier.is_configured():
                print(
                    f"  📱  {notifier.name} ON · dispara em +{args.alert_threshold:.2f}% · "
                    f"cooldown {args.alert_cooldown} min"
                )
            else:
                print(
                    f"  ⚠  --phone-alerts ({notifier.name}) faltam vars: "
                    f"{', '.join(notifier.missing_keys())}. Rode `python configure.py`."
                )
        except ValueError as e:
            print(f"  ⚠  provedor inválido: {e}")
    print("  Ctrl+C para parar.\n")
    cache: Optional[dict] = None
    cycle = 0
    while True:
        try:
            cycle += 1
            refetch = cycle % 6 == 1  # full refetch ~hourly when interval=10min
            res = _run_live_cycle(args, render=False, refetch=refetch, cache=cache)
            if res["ok"]:
                cache = res["cache"]
                ts = datetime.now().strftime("%H:%M")
                d = res["decision"]
                pn = res["probs"]["exchange_now"]
                rate = res["rate_live"] or 0.0
                sz = res["size"]
                rem = res["deadline_remaining"]
                rem_str = f"{rem}d" if rem is not None else "—"
                tags = []
                if res["notified"]:
                    tags.append("◈ ALERT")
                a = res.get("alert")
                if a and a.get("fired"):
                    tags.append(f"📱 {a['provider']} {a['delta_pct']:+.2f}%")
                tag = "  ".join(tags) if tags else "·"
                print(
                    f"  [{ts}] {d:<13} p_now={pn:.2f}  size={sz:.0%}  "
                    f"R$ {rate:.4f}  prazo={rem_str}  {tag}"
                )
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  watch interrompido.\n")
            return
        except Exception as e:
            print(f"  ⚠  ciclo falhou ({e}) — retentando em {args.watch_interval} min")
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="USD/BRL Exchange Timing Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python fx_timing.py                        live analysis, run any day
  python fx_timing.py --lang pt              saída em português
  python fx_timing.py --notify               opens HTML alert when p(now) is high
  python fx_timing.py --watch --notify       run in background, alert on flip
  python fx_timing.py --backtest             backtest on default schedule (2nd & 17th)
  python fx_timing.py --backtest --days 5 20 backtest on your own schedule (5th & 20th)
  python fx_timing.py --backtest --deadline-days 15  enforce 15-day deadline""",
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
    parser.add_argument(
        "--deadline-days",
        type=int,
        default=DEFAULT_DEADLINE_DAYS,
        metavar="N",
        help=f"Forced execution window in days (default: {DEFAULT_DEADLINE_DAYS}). "
        "Backtest waits at most this long before exchanging anyway.",
    )
    parser.add_argument(
        "--spread-bps",
        type=int,
        default=DEFAULT_SPREAD_BPS,
        metavar="BPS",
        help=f"Effective FX spread in basis points (default: {DEFAULT_SPREAD_BPS} = 0.50%%).",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Open an HTML alert in the default browser when p(now) is high.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Background mode: re-run on a schedule and notify on flip.",
    )
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=WATCH_INTERVAL_MIN,
        metavar="MIN",
        help=f"Minutes between checks in --watch mode (default: {WATCH_INTERVAL_MIN}).",
    )
    parser.add_argument(
        "--phone-alerts",
        action="store_true",
        help="Send phone alerts when USD/BRL rises past --alert-threshold. "
        "Run `python configure.py` first.",
    )
    parser.add_argument(
        "--alert-provider",
        choices=available(),
        default=None,
        help="Messaging backend (default: NOTIFIER_PROVIDER from .env, "
        f"else '{DEFAULT_PROVIDER}').",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        metavar="PCT",
        help="%% rise vs anchor that triggers an alert (default: 1.0, "
        "or FX_ALERT_THRESHOLD_PCT from .env).",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=int,
        default=None,
        metavar="MIN",
        help="Minutes between consecutive alerts (default: 60, "
        "or FX_ALERT_COOLDOWN_MIN from .env).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_USER_NAME,
        help=f"Name shown in the browser alert (default: {DEFAULT_USER_NAME}).",
    )
    parser.add_argument(
        "--dca-floor",
        type=float,
        default=DEFAULT_DCA_FLOOR,
        metavar="FRAC",
        help=f"Minimum fraction to convert each cycle (default: {DEFAULT_DCA_FLOOR:.2f}). "
        "Vanguard-DCA discipline — caps worst-case regret variance.",
    )
    parser.add_argument(
        "--dca-ceiling",
        type=float,
        default=DEFAULT_DCA_CEILING,
        metavar="FRAC",
        help=f"Maximum fraction to convert when conviction is high (default: {DEFAULT_DCA_CEILING:.2f}).",
    )
    parser.add_argument(
        "--mark-executed",
        action="store_true",
        help="Mark the most recent journal entry as executed (anchors the deadline countdown).",
    )
    parser.add_argument(
        "--audit",
        type=int,
        nargs="?",
        const=30,
        default=None,
        metavar="DAYS",
        help="Print behavior-gap audit for the last N days (default: 30) and exit.",
    )
    args = parser.parse_args()

    global _LANG
    _LANG = args.lang

    # Load .env (silently no-op if absent) before reading alert config
    rate_alert.load_env()
    if args.alert_threshold is None:
        args.alert_threshold = float(
            os.getenv("FX_ALERT_THRESHOLD_PCT", rate_alert.DEFAULT_THRESHOLD_PCT)
        )
    # Cooldown defaults to watch-interval (single source of truth) unless the
    # user explicitly overrides via flag or .env. This keeps the two values in
    # sync: every check that crosses the threshold can fire an alert.
    if args.alert_cooldown is None:
        env_cooldown = os.getenv("FX_ALERT_COOLDOWN_MIN")
        args.alert_cooldown = int(env_cooldown) if env_cooldown else args.watch_interval
    if args.alert_provider is None:
        args.alert_provider = os.getenv(ENV_VAR, DEFAULT_PROVIDER)

    # Validate day numbers
    for d in args.days:
        if not 1 <= d <= 28:
            parser.error(f"--days: {d} is out of range. Use values between 1 and 28.")
    check_days = tuple(sorted(set(args.days)))

    # Validate DCA bounds
    if not 0.0 <= args.dca_floor <= args.dca_ceiling <= 1.0:
        parser.error(
            "--dca-floor and --dca-ceiling must satisfy 0 ≤ floor ≤ ceiling ≤ 1."
        )

    # ── Standalone subcommands ──────────────────────────────────────────────────────
    if args.mark_executed:
        entry = journal.mark_executed()
        if entry is None:
            print("\n  ⚠  diário vazio — rode uma análise antes de marcar execução.\n")
            return
        print(
            f"\n  ✓  marcado como executado: {entry.ts.strftime('%d/%m/%Y %H:%M')}  "
            f"· {entry.decision.upper()} @ R$ {entry.rate_signal:.4f} · size {entry.size:.0%}\n"
        )
        return

    if args.audit is not None:
        a = journal.audit_summary(days=args.audit)
        print()
        print(f"  AUDITORIA · últimos {a['window_days']} dias")
        print("  " + "─" * 60)
        print(f"  Execuções do script   {a['total_runs']:>4}")
        print(f"  Alertas disparados   {a['alerts']:>4}")
        print(f"  Câmbios marcados     {a['executed']:>4}")
        print(f"  Alertas ignorados    {a['missed_alerts']:>4}")
        if a["alerts"]:
            print(
                f"  Taxa de override    {a['override_rate']:>5.0%}  "
                "← quanto você ignorou o próprio modelo"
            )
        if a["missed_entries"]:
            print()
            print("  Alertas não executados:")
            for e in a["missed_entries"][-10:]:
                print(
                    f"    {e.ts.strftime('%d/%m %H:%M')}  "
                    f"NOW p={e.p_now:.2f} @ R$ {e.rate_signal:.4f}"
                )
        print()
        return

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

        cot_eur = fetch_cot_eur(warmup)
        if cot_eur is not None:
            all_data["cot_eur"] = cot_eur
            print(
                f"  ✓  cot_eur      — {len(cot_eur)} days  "
                f"(latest net {cot_eur.iloc[-1]:+,.0f} contracts)"
            )
        else:
            print("  ⚠  COT signal disabled (CFTC unavailable)")

        rows = run_backtest(
            all_data,
            carry_diff,
            check_days,
            dca_floor=args.dca_floor,
            dca_ceiling=args.dca_ceiling,
        )
        scenarios = sequential_sim(
            rows, check_days, args.deadline_days, args.spread_bps
        )
        render_backtest(
            rows, scenarios, check_days, args.deadline_days, args.spread_bps
        )

    elif args.watch:
        _watch_loop(args)

    else:
        _run_live_cycle(args)


if __name__ == "__main__":
    main()
