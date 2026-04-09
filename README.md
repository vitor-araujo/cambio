<p align="center">
  <h1 align="center">cambio</h1>
  <p align="center">quantitative timing for USD → BRL exchanges</p>
  <p align="center">
    <a href="https://github.com/vitor-araujo/cambio/blob/main/LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square"></a>
    <img alt="Python" src="https://img.shields.io/badge/python-3.10+-4B8BBE?style=flat-square&logo=python&logoColor=white">
    <img alt="no API key" src="https://img.shields.io/badge/data-free%20·%20no%20API%20key-f59e0b?style=flat-square">
    <img alt="en · pt" src="https://img.shields.io/badge/lang-en%20·%20pt--BR-6366f1?style=flat-square">
  </p>
</p>

You receive a USD payment. You need BRL. Do you convert today, or wait?

**cambio** fetches live market data, runs 10 quant signals across macro, technical, and carry dimensions, and gives you a probability-graded answer — calibrated to *your* payment schedule.

---

```
══════════════════════════════════════════════════════════════════
  USD → BRL   EXCHANGE TIMING MODEL
  2025-06-02   ·   R$ 5.7208
══════════════════════════════════════════════════════════════════

  Trend Regime:  no clear trend  (mean-reversion signals fully active)

  SIGNALS                      ← WAIT   NOW →   score    wt
  ────────────────────────────────────────────────────────────────
  DXY                |                |  -0.12  14%  [WAIT]
  Brent              |        ▶▶▶     |  +0.41   8%  [NOW ]
  VALE               |        ▶▶▶     |  +0.50   6%  [NOW ]
  VIX                |                |  -0.08  10%  [FLAT]
  IBOV               |        ▶▶▶     |  +0.44   8%  [NOW ]
  Carry (SELIC−FFR)  |    ◀◀◀         |  -0.52   5%  [WAIT]
  USD/BRL Level      |  ◀◀◀           |  -0.40  13%  [WAIT]
  RSI(14)            |        ▶▶▶▶    |  +0.55  15%  [NOW ]
  Bollinger %B       |        ▶▶▶▶    |  +0.64  11%  [NOW ]
  USD/BRL Trend      |                |  +0.06  10%  [FLAT]

  Composite: +0.178   Agreement: 60%   Regime adj: +0.00

  PROBABILITY DISTRIBUTION
  ────────────────────────────────────────────────────────────────
  Exchange Now   58.2%  [████████████████████░░░░░░░░░░░░░░]
  Split 50/50    12.9%  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  Wait           28.9%  [██████████░░░░░░░░░░░░░░░░░░░░░░░░]
  ────────────────────────────────────────────────────────────────

  ◈  signals lean toward exchanging now
     Multiple indicators suggest the current rate may be near a
     local high. This is not a guarantee — past signal accuracy
     has been ~58% on this call.

  Time horizon note:
    < 7 days   execute regardless — timing models don't help at this horizon
    7–30 days  the signal above is most relevant in this window
    > 30 days  consider weighting WAIT more; more time = more optionality

  ⚠  Past signal accuracy is no guarantee of future results.
     This tool does not constitute financial or investment advice.
══════════════════════════════════════════════════════════════════
```

---

## quick start

```bash
git clone https://github.com/vitor-araujo/cambio.git && cd cambio
python3 -m venv .venv && .venv/bin/pip install -q yfinance pandas numpy
.venv/bin/python fx_timing.py
```

No API keys. Pulls live data from Yahoo Finance and the [BCB open API](https://dadosabertos.bcb.gov.br/).

---

## features

- **10 signals** across three factor families — momentum, carry, and mean-reversion
- **ADX-conditioned** RSI and Bollinger %B: mean-reversion signals auto-suppressed in trending markets
- **Probability-graded verdicts** — language scales with confidence, never overcommits
- **Walk-forward backtest** (`--backtest`) against your own payment schedule with `--days`
- **Português** output with `--lang pt` — made for Brazilians

---

## signals

| Factor | Signal | Weight | Logic |
|---|---|---|---|
| **Momentum** | DXY Index | 14% | Rising → USD stronger → wait |
| | Brent Crude | 8% | Rising → commodity boost → now |
| | VALE (iron ore) | 6% | Rising → Brazil trade → now |
| | VIX | 10% | Elevated + rising → risk-off → wait |
| | IBOVESPA | 8% | Rising → Brazil sentiment → now |
| **Carry** | SELIC − FFR | 5% | High differential → BRL attractive → wait |
| **Mean-Rev** | RSI(14) | 15% | >70 overbought → now · <30 oversold → wait |
| | Bollinger %B | 11% | Near upper band → now · near lower → wait |
| | USD/BRL percentile | 13% | Multi-year high → now |
| | 30d MA slope | 10% | Rising → wait · falling → now |

RSI and Bollinger %B are weighted down by up to 70% when ADX detects a strong trend — avoiding mean-reversion calls against running momentum.

---

## backtest

Walk-forward, no look-ahead. Oracle = rate at the **next scheduled check date** — not a theoretical intraday price you couldn't act on.

| Schedule | NOW accuracy | WAIT accuracy | Calls |
|---|---|---|---|
| 2nd & 17th (default) | 58.3 % | 44.4 % | 102 decisions |
| 5th & 20th | **75.0 %** | 42.2 % | 103 decisions |

**Accuracy varies by schedule.** Run the backtest on your actual payment dates before relying on the model.

```bash
# test your own schedule
.venv/bin/python fx_timing.py --backtest --days 5 20
```

---

## options

```
fx_timing.py [--backtest] [--days DAY ...] [--lang {en,pt}]

  --backtest           walk-forward backtest since 2022
  --days 5 20          which day(s) of month you typically decide (default: 2 17)
  --lang pt            output in Portuguese / saída em português
```

---

## contributing

PRs welcome. High-value directions:

- Brazil 5Y CDS or EMBI+ spread as sovereign risk signal
- COPOM calendar as a volatility event filter  
- Cross-sectional EM FX momentum (ZAR, MXN, CLP)
- HMM 2-state regime classifier to replace ADX

Include a backtest accuracy diff in any signal PR.

---

> ⚠️ **Disclaimer.** This tool provides probabilistic analysis of publicly available market signals for informational purposes only. It is not financial advice, investment advice, or a solicitation to buy or sell any currency or asset. Past model performance does not guarantee future results. Always consult a licensed financial professional before making currency exchange decisions. Use at your own risk. See [LICENSE](LICENSE).
