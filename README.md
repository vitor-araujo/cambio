# timing-the-real

> A quantitative signal model for timing USD → BRL currency exchanges.

**Not investment advice. Not financial advice. A tool to reduce emotional timing — nothing more.**

---

## What it does

Fetches live market data, runs 10 quantitative signals across macro, technical, and carry dimensions, and outputs a probability distribution: should you exchange USD to BRL **now**, **wait**, or **split**?

Includes a full **walk-forward backtest** (2022 → present) on every 2nd and 17th of each month so you can see exactly how the model would have performed on historical data before trusting it with real money.

---

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- It is **not** financial advice, investment advice, or a recommendation to buy, sell, or hold any currency or asset.
- Past model accuracy does **not** guarantee future results.
- FX markets are unpredictable. The model's backtest shows ~52 % accuracy on "exchange now" calls — barely above a coin flip.
- **Always consult a licensed financial advisor before making currency exchange decisions.**
- The authors accept no liability for any financial losses arising from use of this tool.

---

## Signals

The model computes a weighted composite from 10 signals across three factor categories:

### Momentum (~42%)
| Signal | Ticker | Logic |
|---|---|---|
| DXY | `DX-Y.NYB` | Dollar strength → USD/BRL up → wait |
| Brent Crude | `BZ=F` | Commodities up → BRL stronger → exchange now |
| VALE | `VALE` | Iron ore proxy → Brazil trade → exchange now |
| VIX | `^VIX` | Risk-off spike → EM selloff → wait |
| IBOVESPA | `^BVSP` | Brazil equity sentiment → exchange now |

### Carry (~5%)
| Signal | Source | Logic |
|---|---|---|
| SELIC − FFR | BCB API + `^IRX` | High positive carry → BRL attractive → wait |

### Mean-Reversion / Peak Detection (~38%)
| Signal | Logic |
|---|---|
| RSI(14) on USD/BRL | Overbought → exchange now before rate falls |
| Bollinger %B on USD/BRL | Near upper band → exchange now |
| USD/BRL Historical Percentile | Rate at multi-year high → exchange now |
| 30d MA Slope of USD/BRL | Rising trend → wait; falling → exchange now |

All momentum signals are z-score normalised (±3σ → ±1) via a 5/20d MA crossover. An ADX(14) regime filter applies a residual nudge when a strong directional trend is confirmed.

---

## Backtest results (2022 – present)

Walk-forward, no look-ahead. Evaluated on the **next check date** (the actual next 2nd or 17th you would have traded on) — not a theoretical maximum that you couldn't have acted on.

```
Exchange Now    52.4 %   (21 calls)
Wait            44.4 %   (81 calls)
```

| Year | Accuracy | NOW | WAIT |
|------|----------|-----|------|
| 2022 | 54.2 % | 6 | 18 |
| 2023 | 33.3 % | 2 | 22 |
| 2024 | 50.0 % | 6 | 18 |
| 2025 | 54.2 % | 7 | 17 |

**Honest read:** The model's real value is as a **NOW filter** — when it fires "exchange now", the current rate has a >50 % chance of being a local peak. The WAIT signal is closer to a coin flip and should be treated as a directional lean, not a confident prediction.

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/vitor-araujo/timing-the-real.git
cd timing-the-real

python3 -m venv .venv
.venv/bin/pip install yfinance pandas numpy
```

---

## Usage

### Live analysis
```bash
.venv/bin/python fx_timing.py
```

Fetches live data, runs all signals, outputs the current recommendation.

```
══════════════════════════════════════════════════════════════════
  USD → BRL   EXCHANGE TIMING MODEL
  2025-06-02   ·   R$ 5.7208

  Trend Regime:  ranging / no trend

  SIGNALS               ← WAIT   NOW →   score   wt
  ─────────────────────────────────────────────────────────

  DXY                |                |  -0.12  14%  [WAIT]
  Brent              |        ▶▶▶     |  +0.41   8%  [NOW ]
  ...

  ▶  ⚡ EXCHANGE NOW
```

### Walk-forward backtest
```bash
.venv/bin/python fx_timing.py --backtest
```

Runs the full historical simulation (~2 min). Shows a decision table, accuracy breakdown by year, and a sequential P&L simulation of following the model vs immediate exchange.

---

## How the decision works

```
composite score = Σ(signal_score × weight) / Σ(weights)

p_now  = sigmoid(composite × 4)
p_wait = 1 − p_now
(signal disagreement routes mass into "split")

if p_now > 0.51  →  EXCHANGE NOW
else             →  WAIT
```

The 0.51 threshold was calibrated on the walk-forward backtest to maximise NOW precision (>51%) while keeping call volume meaningful (~20 calls over 3 years).

---

## Time horizon override

Regardless of what the model says:

| Horizon | Action |
|---|---|
| < 7 days | Execute regardless |
| 7 – 30 days | Follow the model |
| > 30 days | Weight WAIT more aggressively |

---

## Project structure

```
timing-the-real/
├── fx_timing.py   # entry point — run this
├── signals.py     # signal library (indicators, build_signals)
├── LICENSE
└── README.md
```

---

## Data sources

All data is fetched at runtime from free public sources:

- **Yahoo Finance** via [`yfinance`](https://github.com/ranaroussi/yfinance) — prices, indices, ETFs
- **Banco Central do Brasil (BCB) open API** — SELIC rate (series 432, no key required)

No API keys needed.

---

## Contributing

PRs welcome. Useful directions:

- Better Brazil sovereign risk proxy (EMBI+ spread, CDS)
- COPOM meeting calendar as a volatility event signal
- Peer EM FX cross-sectional momentum (ZAR, MXN, CLP)
- Regime detection via HMM (2-state)

Please keep the core script runnable as a single command and add a backtest result to any signal PR.

---

## License

MIT — see [LICENSE](LICENSE).

Use freely. No warranty. Not financial advice.