<p align="center">
  <h1 align="center">cambio</h1>
  <p align="center">a quantitative timing model for USD → BRL exchanges</p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-4B8BBE?style=flat-square&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" />
    <img src="https://img.shields.io/badge/data-free%20%2F%20no%20API%20key-f59e0b?style=flat-square" />
    <img src="https://img.shields.io/badge/NOW%20accuracy-52–75%25-6366f1?style=flat-square" />
  </p>
</p>

---

> **Not financial advice. Not investment advice.**  
> This is a decision-support tool — it removes emotion from timing, it does not predict the future.  
> Past accuracy does not guarantee future results. Use at your own risk.

---

## what it does

You receive a USD payment. You need to convert it to BRL at some point.  
Do you exchange today, or wait for a better rate?

Run `cambio` on the day you're deciding. It fetches live market data, runs 10 quantitative signals across macro, technical, and carry dimensions, and outputs a probability-weighted recommendation.

You can also **backtest it against your own payment schedule** — the day you typically receive funds — to see how the model would have performed historically on your specific dates.

---

## demo

```
══════════════════════════════════════════════════════════════════
  USD → BRL   EXCHANGE TIMING MODEL
  2025-06-02   ·   R$ 5.7208
══════════════════════════════════════════════════════════════════

  Trend Regime:  ranging / no trend  (mean-reversion signals active)

  SIGNALS                      ← WAIT   NOW →   score    wt
  ────────────────────────────────────────────────────────────────

  DXY                |                |  -0.12  14%  [WAIT]
    103.4  vs MA20 104.1  ↓

  Brent              |        ▶▶▶     |  +0.41   8%  [NOW ]
    $82.1  vs MA20 $79.3  ↑

  VALE               |        ▶▶▶     |  +0.50   6%  [NOW ]
    $14.2  vs MA20 $13.1  ↑

  VIX                |                |  -0.08  10%  [FLAT]
    18.3  calm

  IBOV               |        ▶▶▶     |  +0.44   8%  [NOW ]
    128540 vs MA20 123800  ↑

  Carry              |    ◀◀◀         |  -0.52   5%  [WAIT]
    8.7%/yr  attractive→WAIT

  USD/BRL Level      |  ◀◀◀           |  -0.40  13%  [WAIT]
    R$5.7208  72th pct  ADX 21.4

  RSI(14)            |        ▶▶▶▶    |  +0.55  15%  [NOW ]
    63.8  neutral  [ranging]

  Bollinger %B       |        ▶▶▶▶    |  +0.64  11%  [NOW ]
    0.82  within bands  [ranging]

  USD/BRL Trend      |                |  +0.06  10%  [FLAT]
    ↑ rising slope  [+0.12% / 10d]

  Composite: +0.178   Agreement: 60%   Regime adj: +0.00

  PROBABILITY DISTRIBUTION
  ────────────────────────────────────────────────────────────────

  Exchange Now   58.2%  [████████████████████░░░░░░░░░░░░░░]
  Split 50/50    12.9%  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  Wait           28.9%  [██████████░░░░░░░░░░░░░░░░░░░░░░░░]

  ────────────────────────────────────────────────────────────────

  ▶  ⚡ EXCHANGE NOW
     Rate is favorable — signals point to BRL weakening ahead

══════════════════════════════════════════════════════════════════
```

---

## install

```bash
git clone https://github.com/vitor-araujo/cambio.git
cd cambio

python3 -m venv .venv
.venv/bin/pip install yfinance pandas numpy
```

No API keys required. Data is fetched live from Yahoo Finance and the BCB open API.

---

## usage

### live — run any day

```bash
# English (default)
.venv/bin/python fx_timing.py

# Português
.venv/bin/python fx_timing.py --lang pt
```

Fetches current market data, outputs a recommendation in ~5 seconds. Run it whenever you have USD to convert and are deciding whether to act now or hold.

### backtest — calibrate to your payment schedule

```bash
# default: tests the 2nd and 17th of each month (2022 → present)
.venv/bin/python fx_timing.py --backtest

# your salary lands on the 5th? test that
.venv/bin/python fx_timing.py --backtest --days 5

# freelance, two paydays a month?
.venv/bin/python fx_timing.py --backtest --days 10 25

# combine with language flag
.venv/bin/python fx_timing.py --lang pt --backtest --days 5
```

The `--days` flag sets which days of the month to treat as decision points. **Accuracy varies by schedule** — backtest your own dates before relying on the model.

Example: on a **5th & 20th** schedule, "exchange now" calls showed **75 % accuracy** historically. Default 2nd & 17th gives 52 %. Same model, different cadence.

---

## signals

Three factor families, 10 signals total — all computed walk-forward with no look-ahead.

### momentum  `~42% weight`

| Signal | Source | Direction |
|---|---|---|
| DXY (Dollar Index) | `DX-Y.NYB` | Rising → USD stronger → **wait** |
| Brent Crude | `BZ=F` | Rising → commodity boost for BRL → **now** |
| VALE (iron ore proxy) | `VALE` | Rising → Brazil trade surplus → **now** |
| VIX (risk sentiment) | `^VIX` | Elevated + rising → risk-off, EM sells → **wait** |
| IBOVESPA | `^BVSP` | Rising → Brazil sentiment improving → **now** |

### carry  `~5% weight`

| Signal | Source | Direction |
|---|---|---|
| SELIC − Fed Funds Rate | BCB open API + `^IRX` | High differential → BRL carry attractive → **wait** |

### mean-reversion / peak detection  `~38% weight`

| Signal | Direction |
|---|---|
| RSI(14) on USD/BRL | > 70 overbought → **now**; < 30 oversold → **wait** |
| Bollinger %B on USD/BRL | Near upper band → **now**; near lower → **wait** |
| Historical percentile of USD/BRL | Rate at multi-year high → **now** |
| 30d MA slope of USD/BRL | Rising → **wait**; falling → **now** |

RSI and Bollinger %B are **ADX-conditioned**: in a strong trend (ADX > 25) their weights are suppressed to avoid calling mean-reversion against a running trend.

---

## backtest results

Walk-forward, no look-ahead. Oracle = rate at your **next scheduled check date** (not a theoretical maximum you couldn't trade at).

Default schedule (2nd & 17th):

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

**Run `--backtest --days <your-days>` to see results for your own schedule.**

---

## time horizon

| You need BRL in… | Action |
|---|---|
| < 7 days | Exchange regardless of signal |
| 7 – 30 days | Follow the model |
| > 30 days | Weight WAIT more aggressively |

---

## structure

```
cambio/
├── fx_timing.py    ← entry point
├── signals.py      ← indicator library
├── LICENSE
└── README.md
```

---

## contributing

PRs welcome. High-value directions:

- Brazil 5Y CDS as sovereign risk signal
- COPOM meeting dates as volatility event filter
- Cross-sectional EM FX momentum (ZAR, MXN, CLP)
- HMM 2-state regime classifier to replace ADX

Keep the single-command entrypoint. Include backtest diff in any signal PR.

---

## license

MIT — see [LICENSE](LICENSE).  
Free to use. No warranty. **Not financial advice.**
