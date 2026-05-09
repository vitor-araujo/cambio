<p align="center">
  <h1 align="center">cambio</h1>
  <p align="center"><em>Timing quantitativo para câmbio USD → BRL.</em></p>
  <p align="center">
    <a href="https://github.com/vitor-araujo/cambio/blob/main/LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square"></a>
    <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-4B8BBE?style=flat-square&logo=python&logoColor=white">
    <img alt="versão 0.2.0" src="https://img.shields.io/badge/version-0.2.0-fbbf24?style=flat-square">
    <img alt="sem API key" src="https://img.shields.io/badge/dados-grátis%20·%20sem%20API%20key-f59e0b?style=flat-square">
    <img alt="pt · en" src="https://img.shields.io/badge/lang-pt--BR%20·%20en-6366f1?style=flat-square">
  </p>
</p>

---

**cambio** é um modelo probabilístico que decide se você converte dólar agora ou espera. Roda 12 sinais quantitativos sobre dados públicos, calibra o resultado pro seu calendário de pagamentos, e te avisa no navegador quando o dólar parecer que vai cair.

Documentação dos sinais: [`signals.py`](signals.py) · Código-fonte: [github.com/vitor-araujo/cambio](https://github.com/vitor-araujo/cambio)

---

## Recursos

* 🎯 **Decisão em três vias** — AGORA · DIVIDIR · AGUARDAR, com linguagem graduada por convicção.
* 📡 **Dados ao vivo** — PTAX comercial (BCB), AwesomeAPI (intraday), Yahoo (DXY/Brent/VIX/etc), CFTC (COT), SELIC. Zero API keys.
* 🔄 **Ajuste intraday** — a última cotação substitui a barra do dia antes do cálculo dos sinais.
* 🛎️ **Alerta no navegador** — abre uma página HTML com link direto pra Higlobe quando p(AGORA) ≥ 55 %.
* 👁️ **Modo background** — fica rodando, te avisa só quando o sinal vira.
* 📓 **Diário automático** — toda decisão fica em `.fx_journal.csv`. A próxima execução te mostra se a anterior estava certa.
* ⏱️ **Backtest com prazo real** — `--deadline-days` força execução depois do seu cutoff.
* 💰 **Backtest com custo** — `--spread-bps` desconta o spread da corretora dos resultados.
* 🇧🇷 **Saída em português** — `--lang pt`.

---

## Requisitos

* Python **3.10+**
* `yfinance`, `pandas`, `numpy`
* Conexão com a internet (os dados são buscados em tempo real)

---

## Instalação

```bash
git clone https://github.com/vitor-araujo/cambio.git && cd cambio
python3 -m venv .venv
.venv/bin/pip install -q yfinance pandas numpy
```

> **Windows:** troque `.venv/bin/` por `.venv\Scripts\` em todos os comandos.

---

## Uso

### Análise pontual

```bash
.venv/bin/python fx_timing.py --lang pt
```

```
══════════════════════════════════════════════════════════════════
  USD → BRL   MODELO DE TIMING DE CÂMBIO
  2025-06-02   ·   R$ 5.7208
══════════════════════════════════════════════════════════════════

  Regime de Tendência:  sem tendência clara

  SINAIS                        ← AGUARDAR  AGORA →   score    peso
  ────────────────────────────────────────────────────────────────
  DXY                |                |  -0.12  11%  [AGU.]
  Brent              |        ▶▶▶     |  +0.41   6%  [AGORA]
  VALE               |        ▶▶▶     |  +0.50   5%  [AGORA]
  VIX                |                |  -0.08   9%  [NEUT]
  IBOV               |        ▶▶▶     |  +0.44   7%  [AGORA]
  Carry              |    ◀◀◀         |  -0.52   5%  [AGU.]
  USD/BRL Level      |  ◀◀◀           |  -0.40  10%  [AGU.]
  RSI(14)            |        ▶▶▶▶    |  +0.55  12%  [AGORA]
  Bollinger %B       |        ▶▶▶▶    |  +0.64   9%  [AGORA]
  USD/BRL Trend      |                |  +0.06   9%  [NEUT]
  BRL Futures (6L)   |        ▶       |  +0.18   8%  [AGORA]
  COT USD            |                |  +0.03   9%  [NEUT]

  DISTRIBUIÇÃO DE PROBABILIDADE
  Câmbio Agora   58.2%  [████████████████████░░░░░░░░░░░░░░]
  Dividir 50/50  12.9%  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  Aguardar       28.9%  [██████████░░░░░░░░░░░░░░░░░░░░░░░░]

  ◈  os sinais indicam uma possível oportunidade de câmbio agora
══════════════════════════════════════════════════════════════════
```

### Modo background

Deixe rodando em uma aba do terminal — alerta automático quando o sinal vira:

```bash
.venv/bin/python fx_timing.py --lang pt --watch --notify
```

Re-roda a cada 60 minutos, registra no diário, abre o alerta no navegador na primeira virada para AGORA com p ≥ 55 %. `Ctrl+C` para parar.

### Backtest realista

```bash
.venv/bin/python fx_timing.py --backtest --days 5 20 --deadline-days 15 --spread-bps 50
```

Walk-forward desde 2022, sem look-ahead, no *seu* calendário de recebimentos.

---

## Sinais

Doze sinais em três famílias. Pesos somam 100 %.

| Família | Sinal | Peso | Lógica |
|---|---|---|---|
| **Momentum** | DXY | 11 % | sobe → dólar forte → aguardar |
| | Brent | 6 % | sobe → boost de commodities → agora |
| | VALE | 5 % | sobe → balança comercial → agora |
| | VIX | 9 % | alto + subindo → risk-off → aguardar |
| | IBOV | 7 % | sobe → otimismo Brasil → agora |
| **Carry** | SELIC − FFR | 5 % | diferencial alto → BRL atrativo → aguardar |
| **Reversão** | USD/BRL Level | 10 % | percentil multi-anual alto → agora |
| | RSI(14) | 12 % | >70 sobrecomprado → agora · <30 → aguardar |
| | Bollinger %B | 9 % | banda superior → agora · inferior → aguardar |
| | USD/BRL Trend | 9 % | inclinação MA 30d → reversão de tendência |
| **Posicionamento** | BRL Futures (6L) | 8 % | momentum dos futuros CME — fluxo institucional |
| | COT USD | 9 % | net position EUR/USD — sentimento de hedge funds |

**Filtro de regime:** o ADX(14) detecta tendência forte e reduz o peso dos sinais de reversão (RSI, Bollinger, Level) em até 70 % — para evitar chamar reversão contra um momentum em andamento.

---

## Backtest

Oracle = taxa no **próximo dia de checagem agendado** (não preço intraday teórico).

| Calendário | AGORA | AGUARDAR | Decisões |
|---|---|---|---|
| Dias 2 e 17 (padrão) | 58,3 % | 44,4 % | 102 |
| Dias 5 e 20 | **75,0 %** | 42,2 % | 103 |

**A acurácia varia com o calendário.** Sempre rode no seu antes de confiar.

---

## Referência da CLI

```
fx_timing.py [--backtest] [--days DIA ...] [--lang {en,pt}]
             [--deadline-days N] [--spread-bps BPS]
             [--notify] [--watch] [--watch-interval MIN] [--name NOME]
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--backtest` | — | Roda walk-forward desde 2022 |
| `--days` | `2 17` | Dia(s) do mês em que você decide |
| `--lang` | `en` | Idioma da saída (`en` ou `pt`) |
| `--deadline-days` | `15` | Prazo limite em dias para forçar execução |
| `--spread-bps` | `50` | Spread efetivo da corretora em basis points |
| `--notify` | — | Abre alerta HTML no navegador em flip-to-AGORA |
| `--watch` | — | Modo background — re-roda em loop |
| `--watch-interval` | `60` | Minutos entre checagens em `--watch` |
| `--name` | `Vitor` | Nome no cabeçalho do alerta no navegador |

---

## Estrutura do projeto

Quatro módulos. Cada um com responsabilidade única.

### `fx_timing.py` — orquestrador

Ponto de entrada. Junta tudo:

* parsing de CLI e dispatch (`main`)
* download dos dados (`fetch`, `fetch_ptax`, `fetch_selic`, `fetch_cot_eur`, `_fetch_live_fx`)
* aplicação do tick intraday (`_apply_intraday`)
* probabilidades e regime (`probabilities`, `apply_regime`, `decide`)
* renderização do terminal (`render_live`, `render_backtest`)
* loop walk-forward e simulação P&L (`run_backtest`, `sequential_sim`)
* loop background (`_run_live_cycle`, `_watch_loop`)
* integração com journal e notify

### `signals.py` — biblioteca de sinais

Compute puro, sem efeitos colaterais. Funções principais:

* `z_momentum` — spread de MAs normalizado por σ, clipado em [-1, 1]
* `_rsi_value` / `rsi_score` — Wilder RSI(14) e mapeamento para score
* `_pct_b` / `bb_score` — Bollinger %B e mapeamento para score
* `compute_adx` — Wilder ADX(14) com DI±
* `regime_from_adx` — converte ADX em regime de tendência ∈ [-1, 1]
* `carry_score` — calibração absoluta do diferencial SELIC−FFR + tendência
* `build_signals` — agrega tudo em `list[Signal]` + regime score

### `journal.py` — diário de decisões

Append-only CSV (`.fx_journal.csv`). Sem dependências além da stdlib.

* `append(entry)` — adiciona linha
* `last_entry()` — última decisão registrada
* `last_notified()` — última vez que o alerta foi disparado
* `render_summary(prev, cur_rate)` — gera a linha "último sinal há Xh: WAIT @ R$ 5.08 → agora R$ 5.13 (+1.04 %) ✓"
* `should_notify(decision, p_now)` — aplica threshold + cooldown de 6 h

### `notify.py` — alerta HTML

Renderiza `.fx_alert.html` e abre no navegador padrão.

* Tipografia: **Fraunces** (serif) no headline, **JetBrains Mono** nos números
* Paleta: verde-floresta + âmbar (Brasil), gradientes em camadas, grid sutil
* Animações CSS-only — fade-in escalonado, pulse no CTA
* `render_alert(...)` — escreve o HTML
* `open_in_browser(path)` — abre via `webbrowser.open`
* `alert(...)` — atalho que faz os dois

---

## Diário e alertas

A cada execução o cambio adiciona uma linha no CSV local:

```csv
ts,rate_signal,rate_live,decision,p_now,p_split,p_wait,composite,agreement,regime,notified
2025-06-02T14:32:00,5.0810,5.0825,wait,0.31,0.18,0.51,-0.18,0.62,+0.05,0
```

Na próxima execução, aparece no topo:

```
último sinal há 6h: WAIT @ R$ 5.0810  →  agora R$ 5.1340  (+1.04 %) ✓
```

Quando `--notify` está ativo e `p_now ≥ 0.55`, abre `.fx_alert.html` no navegador. Cooldown de 6 horas evita spam. Os dois arquivos ficam locais — não vão pro Git.

---

## Guia completo para leigos 🇧🇷

Nunca usou terminal? Veja o [guia passo-a-passo](docs/GUIDE-PT.md) — instalação do Python, primeiro câmbio, modo background, diário, backtest. Zero conhecimento prévio assumido.

> **Quick links:** [instalar Python](docs/GUIDE-PT.md#passo-1) · [primeiro uso](docs/GUIDE-PT.md#passo-5) · [alerta no navegador](docs/GUIDE-PT.md#passo-6) · [modo background](docs/GUIDE-PT.md#passo-7) · [troubleshooting](docs/GUIDE-PT.md#troubleshooting)

---

## Contribuindo

PRs são bem-vindos. Direções de alto valor:

* CDS 5Y do Brasil ou EMBI+ como sinal de risco soberano
* Calendário do COPOM como filtro de eventos de volatilidade
* Momentum cross-sectional de FX em emergentes (ZAR, MXN, CLP)
* Classificador HMM de 2 estados para substituir o ADX

Inclua um diff de acurácia do backtest em qualquer PR de sinal.

---

## Licença

[MIT](LICENSE).

> ⚠️ **Aviso legal.** Análise probabilística de dados de mercado públicos, apenas para fins informativos. Não é aconselhamento financeiro. O desempenho histórico não garante resultados futuros. Use por sua conta e risco.

---
---

## English

**cambio** is a probabilistic model that decides whether to convert USD now or wait. It runs 12 quant signals over public data, calibrates the answer to your payment schedule, and pings you in the browser when the dollar looks ready to fall.

### Features

* 🎯 **Three-way decision** — NOW · SPLIT · WAIT, probability-graded language.
* 📡 **Live data** — BCB PTAX, AwesomeAPI (intraday), Yahoo, CFTC (COT), SELIC. No API keys.
* 🔄 **Intraday-aware** — last bar replaced with the live tick before signals run.
* 🛎️ **Browser alerts** (`--notify`) — HTML page with one-click Higlobe link when p(NOW) ≥ 55 %.
* 👁️ **Background mode** (`--watch`) — leave it running, alert on flip.
* 📓 **Auto journal** — every call logged to `.fx_journal.csv`; next run audits the previous one.
* ⏱️ **Deadline-aware backtest** (`--deadline-days`).
* 💰 **Cost-aware backtest** (`--spread-bps`).

### Installation

```bash
git clone https://github.com/vitor-araujo/cambio.git && cd cambio
python3 -m venv .venv
.venv/bin/pip install -q yfinance pandas numpy
```

### Usage

```bash
# one-shot analysis
.venv/bin/python fx_timing.py

# background with browser alerts
.venv/bin/python fx_timing.py --watch --notify

# backtest on your real schedule
.venv/bin/python fx_timing.py --backtest --days 5 20 --deadline-days 15 --spread-bps 50
```

### CLI reference

```
fx_timing.py [--backtest] [--days DAY ...] [--lang {en,pt}]
             [--deadline-days N] [--spread-bps BPS]
             [--notify] [--watch] [--watch-interval MIN] [--name NAME]
```

| Flag | Default | Description |
|---|---|---|
| `--backtest` | — | Walk-forward backtest since 2022 |
| `--days` | `2 17` | Day(s) of month you typically decide |
| `--lang` | `en` | Output language (`en` or `pt`) |
| `--deadline-days` | `15` | Forced-execution window |
| `--spread-bps` | `50` | Effective FX spread in basis points |
| `--notify` | — | Open HTML alert on flip-to-NOW |
| `--watch` | — | Background mode — re-run on a schedule |
| `--watch-interval` | `60` | Minutes between checks in `--watch` |
| `--name` | `Vitor` | Name shown in the browser alert |

### Project layout

| Module | Role |
|---|---|
| `fx_timing.py` | Entry point — CLI, data fetch, signal pipeline, render, journal, notify, watch loop |
| `signals.py` | Pure signal library — RSI, Bollinger %B, ADX, carry score, `build_signals` |
| `journal.py` | Append-only CSV log of decisions, last-call summary, notify cooldown |
| `notify.py` | HTML alert renderer (Fraunces + JetBrains Mono) and browser opener |

### Backtest

Walk-forward, no look-ahead. Oracle = rate at the next scheduled check date.

| Schedule | NOW | WAIT | Calls |
|---|---|---|---|
| 2nd & 17th (default) | 58.3 % | 44.4 % | 102 |
| 5th & 20th | **75.0 %** | 42.2 % | 103 |

### Contributing

PRs welcome. High-value directions:

* Brazil 5Y CDS or EMBI+ spread as sovereign risk signal
* COPOM calendar as a volatility event filter
* Cross-sectional EM FX momentum (ZAR, MXN, CLP)
* HMM 2-state regime classifier to replace ADX

Include a backtest accuracy diff in any signal PR.

### License

[MIT](LICENSE).

> ⚠️ **Disclaimer.** Probabilistic analysis of public market data, for informational purposes only. Not financial advice. Past performance does not guarantee future results. Use at your own risk.
