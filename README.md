<p align="center">
  <h1 align="center">cambio</h1>
  <p align="center">a quantitative timing model for USD → BRL exchanges</p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-4B8BBE?style=flat-square&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" />
    <img src="https://img.shields.io/badge/data-free%20%2F%20no%20API%20key-f59e0b?style=flat-square" />
    <img src="https://img.shields.io/badge/NOW%20accuracy-52.4%25-6366f1?style=flat-square" />
  </p>
</p>

---

> **Not financial advice. Not investment advice.**  
> This is a decision-support tool — it removes emotion from timing, it does not predict the future.  
> Past accuracy does not guarantee future results. Use at your own risk.

---

## what it does

On every **2nd and 17th of the month** — the cadence that matters for most retail FX decisions — run:

```
python fx_timing.py
```

You get a probability distribution across three outcomes, a composite signal score, and a clear recommendation backed by 10 quantitative indicators across macro, technical, and carry dimensions.

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

  Time horizon override:
    < 7 days  → execute regardless of model
    7–30d     → follow model recommendation
    > 30d     → weight WAIT more aggressively

══════════════════════════════════════════════════════════════════
```

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

Carry weight is intentionally low: historically high SELIC masked by z-score drift caused consistent miscalibration. Kept as a weak WAIT signal only.

### mean-reversion / peak detection  `~38% weight`

| Signal | Direction |
|---|---|
| RSI(14) on USD/BRL | > 70 overbought → **now**; < 30 oversold → **wait** |
| Bollinger %B on USD/BRL | Near upper band → **now**; near lower → **wait** |
| Historical percentile of USD/BRL | Rate at multi-year high → **now** |
| 30d MA slope of USD/BRL | Rising → **wait**; falling → **now** |

RSI and Bollinger %B are **ADX-conditioned**: in a strong trend (ADX > 25), their weights are suppressed by up to 70% to avoid calling mean-reversion against a running trend.

---

## backtest

Walk-forward on every **2nd and 17th** from January 2022 to present. Oracle = rate at **next check date** (not the theoretical intraday maximum — that's a rate you could never actually trade at).

```
Exchange Now    52.4 %   (21 calls, 11 correct)
Wait            44.4 %   (81 calls)
Overall         46.1 %   (102 decisions)
```

| Year | Accuracy | NOW calls | WAIT calls |
|------|----------|-----------|------------|
| 2022 | 54.2 % | 6 | 18 |
| 2023 | 33.3 % | 2 | 22 |
| 2024 | 50.0 % | 6 | 18 |
| 2025 | 54.2 % | 7 | 17 |

**Honest read:** The model's primary value is as a **NOW filter**. At 52 % precision, when it fires "exchange now" the current rate is more likely a local peak than not. The WAIT signal is a directional lean, not a confident prediction — treat it accordingly.

Run it yourself:

```bash
python fx_timing.py --backtest
```

---

## install

```bash
git clone https://github.com/vitor-araujo/cambio.git
cd cambio

python3 -m venv .venv
.venv/bin/pip install yfinance pandas numpy
```

No API keys required. All data is fetched live from Yahoo Finance and the BCB open API.

---

## usage

```bash
# live recommendation (run any time, ~5 seconds)
.venv/bin/python fx_timing.py

# full walk-forward backtest (~2 minutes)
.venv/bin/python fx_timing.py --backtest
```

---

## time horizon override

The model assumes you have 7–30 days of flexibility. If you don't:

| Horizon | Action |
|---|---|
| < 7 days | Execute regardless of model signal |
| 7 – 30 days | Follow the recommendation |
| > 30 days | Weight WAIT more aggressively |

---

## structure

```
cambio/
├── fx_timing.py    ← run this
├── signals.py      ← signal library, indicators
├── LICENSE
└── README.md
```

---

## contributing

PRs welcome. High-value directions:

- Brazil 5Y CDS as sovereign risk signal
- COPOM meeting dates as volatility event filter
- Cross-sectional EM FX peer momentum (ZAR, MXN, CLP)
- HMM 2-state regime classifier to replace ADX

Keep the single-file entrypoint. Include backtest diff in any signal PR.

---

## license

MIT. See [LICENSE](LICENSE).  
Free to use. No warranty. **Not financial advice.**
