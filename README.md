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

## como usar — guia para leigos 🇧🇷

Nunca mexeu com programação? Sem problema. Esse guia assume que você parte do zero.

O **cambio** é um script Python — um programinha de terminal que roda no seu computador, busca os dados na internet e te diz o resultado. Não tem site, não tem app, não tem instalador. Você vai digitar alguns comandos, e só.

---

### antes de começar — o que é o terminal?

O terminal (também chamado de "linha de comando") é uma janela onde você digita instruções pro computador em vez de clicar em botões. Parece intimidador no começo, mas você vai usar no máximo 5 comandos diferentes.

**Como abrir:**
- **Mac:** pressione `Cmd + Espaço`, digite `Terminal`, aperte Enter
- **Windows:** pressione a tecla Windows, digite `PowerShell`, aperte Enter

Quando o terminal abrir, você vai ver uma linha com um cursor piscando esperando você digitar. É ali que você vai colar os comandos abaixo.

---

### passo 1 — instale o Python

Python é a linguagem em que o cambio foi escrito. Você precisa dela instalada no computador pra rodar o script.

1. Acesse [python.org/downloads](https://python.org/downloads)
2. Clique no botão amarelo de download (ele detecta seu sistema automaticamente)
3. Abra o instalador baixado

> ⚠️ **Windows: passo crítico.** Na primeira tela do instalador, antes de clicar em qualquer coisa, **marque a caixinha que diz "Add Python to PATH"**. Se não marcar isso, nada vai funcionar e você vai ter que reinstalar.

Depois de instalar, confirme que deu certo abrindo o terminal e digitando:

```
python3 --version
```

Aperte Enter. Deve aparecer algo como `Python 3.12.4`. Se aparecer, está pronto. Se aparecer "comando não encontrado", tente `python --version` (sem o 3). Se ainda não funcionar, reinstale o Python e não esqueça do PATH.

---

### passo 2 — baixe o projeto

Agora você precisa copiar os arquivos do cambio pro seu computador.

**Opção A — pelo terminal (recomendado):**

Cole esse comando no terminal e aperte Enter:
```bash
git clone https://github.com/vitor-araujo/cambio.git
```

Isso vai criar uma pasta chamada `cambio` no local atual (normalmente sua pasta de usuário). Se o terminal disser que `git` não existe, baixe em [git-scm.com](https://git-scm.com), instale com as opções padrão, feche e reabra o terminal, e tente de novo.

**Opção B — pelo navegador:**

Acesse [github.com/vitor-araujo/cambio](https://github.com/vitor-araujo/cambio), clique no botão verde **"Code"** e depois em **"Download ZIP"**. Descompacte o ZIP em algum lugar fácil de achar, como a sua Área de Trabalho.

---

### passo 3 — entre na pasta do projeto

No terminal, navegue até a pasta que acabou de criar:

```bash
cd cambio
```

`cd` significa "change directory" — entrar numa pasta. Se você baixou o ZIP e descompactou em outro lugar, substitua `cambio` pelo caminho completo até a pasta. No Mac você pode arrastar a pasta pro terminal após digitar `cd ` (com espaço) que ele preenche o caminho automaticamente.

Confirme que está no lugar certo digitando:
```bash
ls
```
(Mac/Linux) ou:
```bash
dir
```
(Windows). Deve aparecer os arquivos `fx_timing.py`, `signals.py`, `README.md` etc.

---

### passo 4 — crie o ambiente e instale as bibliotecas

O cambio usa três bibliotecas externas (pacotes com funcionalidades prontas). Antes de instalá-las, criamos um "ambiente virtual" — uma pastinha isolada que guarda essas bibliotecas só pro cambio, sem bagunçar o resto do seu computador.

Execute esses dois comandos, **um de cada vez**, esperando cada um terminar antes de digitar o próximo:

```bash
python3 -m venv .venv
```

Esse comando cria o ambiente virtual (pode demorar alguns segundos, sem saída visível).

```bash
.venv/bin/pip install yfinance pandas numpy
```

Esse instala as três bibliotecas. Você vai ver um monte de texto rolando — é normal. Quando voltar pro cursor, está pronto.

> **Windows:** nos dois comandos acima, substitua `.venv/bin/` por `.venv\Scripts\`. Fica assim:
> ```
> .venv\Scripts\pip install yfinance pandas numpy
> ```

Esse passo só precisa ser feito **uma única vez**. Da próxima vez que quiser usar o cambio, pode pular direto pro passo 5.

---

### passo 5 — rode

```bash
.venv/bin/python fx_timing.py --lang pt
```

Aguarde uns 5–10 segundos. O programa vai buscar dados de câmbio, Ibovespa, dólar, petróleo e juros, processar os sinais, e imprimir o resultado direto no terminal.

> **Windows:**
> ```
> .venv\Scripts\python fx_timing.py --lang pt
> ```

---

### passo 6 — backtest no seu calendário (opcional mas recomendado)

O backtest mostra como o modelo teria se comportado nos últimos anos nos *seus* dias específicos de decisão — porque a acurácia muda dependendo do dia do mês.

Se você recebe dólares todo dia 5, por exemplo:

```bash
.venv/bin/python fx_timing.py --lang pt --backtest --days 5
```

Se recebe no dia 10 e no dia 25:

```bash
.venv/bin/python fx_timing.py --lang pt --backtest --days 10 25
```

O backtest demora cerca de 2 minutos e mostra uma tabela completa com cada decisão desde 2022 e o resultado real.

---

### da próxima vez

Você não precisa repetir toda a configuração. Na próxima vez que quiser consultar, basta:

1. Abrir o terminal
2. Entrar na pasta: `cd cambio`
3. Rodar: `.venv/bin/python fx_timing.py --lang pt`

---

### resumo dos comandos

| O que fazer | Mac / Linux | Windows |
|---|---|---|
| Análise de hoje | `.venv/bin/python fx_timing.py --lang pt` | `.venv\Scripts\python fx_timing.py --lang pt` |
| Backtest padrão | `... --backtest` | igual |
| Backtest no seu dia | `... --backtest --days 10` | igual |
| Dois dias por mês | `... --backtest --days 5 20` | igual |

---

### algo deu errado?

| Erro | O que fazer |
|---|---|
| `python3: command not found` | Python não foi instalado corretamente. Reinstale e marque "Add to PATH" no Windows |
| `No module named pip` | Tente `python3 -m ensurepip` e repita o passo 4 |
| `ModuleNotFoundError: No module named 'yfinance'` | Você rodou o Python sem ativar o ambiente. Use `.venv/bin/python`, não só `python` |
| Tela em branco ou trava | Verifique sua conexão com a internet — o script busca dados ao vivo |
| Erro no Windows com `\` vs `/` | No Windows, sempre use `\` nos caminhos: `.venv\Scripts\python` |

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
