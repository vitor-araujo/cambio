<p align="center">
  <h1 align="center">cambio</h1>
  <p align="center">timing quantitativo para câmbio USD → BRL</p>
  <p align="center">
    <a href="https://github.com/vitor-araujo/cambio/blob/main/LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square"></a>
    <img alt="Python" src="https://img.shields.io/badge/python-3.10+-4B8BBE?style=flat-square&logo=python&logoColor=white">
    <img alt="sem API key" src="https://img.shields.io/badge/dados-grátis%20·%20sem%20API%20key-f59e0b?style=flat-square">
    <img alt="pt · en" src="https://img.shields.io/badge/lang-pt--BR%20·%20en-6366f1?style=flat-square">
  </p>
</p>

Você recebeu um pagamento em dólar. Precisa de real. Converte hoje, ou espera?

O **cambio** busca dados de mercado ao vivo, roda 10+ sinais quantitativos (momentum, carry, reversão à média) e te dá uma resposta probabilística — calibrada pro *seu* calendário de pagamentos, *seu* spread e *seu* prazo real. Pode ficar rodando em segundo plano e te avisar no navegador no momento em que o dólar parecer que vai cair.

> **v0.2.0** — modo background (`--watch`), alertas no navegador (`--notify`), diário local com auto-feedback, sinais com ajuste intraday, backtest ciente de prazo e spread, decisão em três vias (AGORA / DIVIDIR / AGUARDAR).

---

```
══════════════════════════════════════════════════════════════════
  USD → BRL   MODELO DE TIMING DE CÂMBIO
  2025-06-02   ·   R$ 5.7208
══════════════════════════════════════════════════════════════════

  Regime de Tendência:  sem tendência clara  (sinais de reversão à média ativos)

  SINAIS                       ← AGUARDAR  AGORA →   score    peso
  ────────────────────────────────────────────────────────────────
  DXY                |                |  -0.12  14%  [AGU.]
  Brent              |        ▶▶▶     |  +0.41   8%  [AGORA]
  VALE               |        ▶▶▶     |  +0.50   6%  [AGORA]
  VIX                |                |  -0.08  10%  [NEUT]
  IBOV               |        ▶▶▶     |  +0.44   8%  [AGORA]
  Carry (SELIC−FFR)  |    ◀◀◀         |  -0.52   5%  [AGU.]
  USD/BRL Level      |  ◀◀◀           |  -0.40  13%  [AGU.]
  RSI(14)            |        ▶▶▶▶    |  +0.55  15%  [AGORA]
  Bollinger %B       |        ▶▶▶▶    |  +0.64  11%  [AGORA]
  USD/BRL Trend      |                |  +0.06  10%  [NEUT]

  Composto: +0.178   Concordância: 60%   Ajuste regime: +0.00

  DISTRIBUIÇÃO DE PROBABILIDADE
  ────────────────────────────────────────────────────────────────
  Câmbio Agora   58.2%  [████████████████████░░░░░░░░░░░░░░]
  Dividir 50/50  12.9%  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  Aguardar       28.9%  [██████████░░░░░░░░░░░░░░░░░░░░░░░░]
  ────────────────────────────────────────────────────────────────

  ◈  os sinais indicam uma possível oportunidade de câmbio agora
     Múltiplos indicadores sugerem que a taxa atual pode estar
     próxima de uma máxima local. Isso não é garantia — a acurácia
     histórica deste sinal foi de ~58 %.

  ⚠  A acurácia histórica do modelo não garante resultados futuros.
     Esta ferramenta não constitui aconselhamento financeiro.
══════════════════════════════════════════════════════════════════
```

---

## início rápido

```bash
git clone https://github.com/vitor-araujo/cambio.git && cd cambio
python3 -m venv .venv && .venv/bin/pip install -q yfinance pandas numpy
.venv/bin/python fx_timing.py --lang pt
```

Sem API keys. Os dados vêm do Yahoo Finance e da [API aberta do Banco Central](https://dadosabertos.bcb.gov.br/).

---

## funcionalidades

- **10+ sinais** em três famílias — momentum, carry e reversão à média
- **ADX-condicionado** — RSI e Bollinger %B reduzem peso automaticamente em mercado de tendência forte
- **Três decisões possíveis** — AGORA, DIVIDIR (50/50) ou AGUARDAR, com linguagem graduada por convicção
- **Ajuste intraday** — a última barra é substituída pela cotação ao vivo, então RSI / BB / Level refletem o preço atual e não o fechamento PTAX de ontem
- **Backtest com prazo real** (`--deadline-days`) — força o câmbio depois do seu prazo limite
- **Backtest com custo** (`--spread-bps`) — os valores em BRL saem líquidos do spread da sua corretora
- **Alerta no navegador** (`--notify`) — uma página HTML com tipografia serifada e link direto pra Higlobe abre quando a p(AGORA) está alta
- **Modo background** (`--watch`) — deixa rodando e recebe um ping quando o sinal vira
- **Diário local** — toda execução fica registrada em `.fx_journal.csv` e a próxima rodada te mostra se a chamada anterior estava certa
- **Backtest walk-forward** (`--backtest`) no *seu* calendário de pagamentos via `--days`
- Saída em **português** com `--lang pt`

---

## sinais

| Família | Sinal | Peso | Lógica |
|---|---|---|---|
| **Momentum** | Índice DXY | 14% | subindo → dólar mais forte → aguardar |
| | Brent (petróleo) | 8% | subindo → boost de commodities → agora |
| | VALE (minério) | 6% | subindo → balança comercial Brasil → agora |
| | VIX | 10% | elevado + subindo → risk-off → aguardar |
| | IBOVESPA | 8% | subindo → otimismo Brasil → agora |
| **Carry** | SELIC − FFR | 5% | diferencial alto → BRL atrativo → aguardar |
| **Reversão** | RSI(14) | 15% | >70 sobrecomprado → agora · <30 sobrevendido → aguardar |
| | Bollinger %B | 11% | perto da banda superior → agora · inferior → aguardar |
| | USD/BRL percentil | 13% | máxima de vários anos → agora |
| | Inclinação MA 30d | 10% | subindo → aguardar · caindo → agora |

O RSI e o Bollinger %B têm o peso reduzido em até 70 % quando o ADX detecta tendência forte — pra evitar chamar reversão contra um momentum em andamento.

---

## backtest

Walk-forward, sem look-ahead. O *oracle* compara a taxa de hoje com a taxa do **próximo dia de checagem agendado** — não com um preço intraday teórico que você não conseguiria executar.

| Calendário | Acurácia AGORA | Acurácia AGUARDAR | Decisões |
|---|---|---|---|
| Dias 2 e 17 (padrão) | 58,3 % | 44,4 % | 102 |
| Dias 5 e 20 | **75,0 %** | 42,2 % | 103 |

**A acurácia muda dependendo do calendário.** Rode o backtest nos *seus* dias reais de recebimento antes de confiar no modelo.

```bash
# testa o seu calendário, com prazo e spread realistas
.venv/bin/python fx_timing.py --lang pt --backtest --days 5 20 --deadline-days 15 --spread-bps 50
```

---

## opções

```
fx_timing.py [--backtest] [--days DIA ...] [--lang {en,pt}]
             [--deadline-days N] [--spread-bps BPS]
             [--notify] [--watch] [--watch-interval MIN] [--name NOME]

  --backtest             backtest walk-forward desde 2022
  --days 5 20            dia(s) do mês em que você decide (padrão: 2 17)
  --lang pt              saída em português / output in Portuguese
  --deadline-days 15     prazo limite em dias (padrão: 15)
  --spread-bps 50        spread efetivo em basis points (padrão: 50 = 0,50 %)
  --notify               abre alerta HTML no navegador quando p(AGORA) é alta
  --watch                modo background — re-roda em loop e alerta no flip
  --watch-interval 60    minutos entre checagens em --watch (padrão: 60)
  --name Vitor           nome no cabeçalho do alerta no navegador
```

---

## guia completo para leigos 🇧🇷

Nunca mexeu com programação? Sem problema. Esse guia parte do zero.

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

### passo 6 — alerta no navegador quando o dólar cair

Em vez de ficar checando o terminal toda hora, você pode pedir pro cambio te avisar visualmente quando ele detectar uma boa janela pra converter:

```bash
.venv/bin/python fx_timing.py --lang pt --notify
```

Quando os sinais convergirem com convicção (probabilidade de "agora" ≥ 55 %), uma página HTML abre no seu navegador padrão com:

- a taxa atual ao vivo
- os 5 sinais mais relevantes do momento
- um botão grande pra abrir o **Higlobe** e fechar o câmbio em um clique

Se já abriu uma vez, ele não reabre nas próximas 6 horas (cooldown automático, evita spam).

O arquivo gerado fica em `.fx_alert.html` na pasta do projeto e não vai pro Git — é só seu.

---

### passo 7 — deixar rodando em segundo plano

A forma mais prática de usar o cambio no dia-a-dia: deixa ele rodando numa janela de terminal e ele te avisa quando precisar agir.

```bash
.venv/bin/python fx_timing.py --lang pt --watch --notify
```

O que acontece:

- A cada 60 minutos (configurável com `--watch-interval`), ele rebusca os dados, recalcula tudo, registra no diário (`.fx_journal.csv`)
- Quando o sinal vira pra **AGORA** com convicção, abre o alerta no navegador automaticamente
- Imprime uma linha por ciclo no terminal pra você ver que está vivo:
  ```
  [14:32] wait          p_now=0.31  R$ 5.0810  ·
  [15:32] exchange_now  p_now=0.62  R$ 5.1340  ◈ ALERT
  ```
- Pra parar: `Ctrl+C`

Quer checar a cada 15 minutos em vez de 60?

```bash
.venv/bin/python fx_timing.py --lang pt --watch --notify --watch-interval 15
```

Quer trocar o nome que aparece no alerta?

```bash
.venv/bin/python fx_timing.py --lang pt --watch --notify --name "Maria"
```

> **Dica:** rode dentro do `tmux` ou de uma aba dedicada do Terminal pra não ocupar uma janela aberta o tempo todo.

---

### passo 8 — diário: o que o modelo disse ontem?

A cada execução (live ou watch), o cambio anota a decisão num arquivo CSV local: `.fx_journal.csv`. Na próxima rodada, ele te mostra uma linha no topo dizendo se a chamada anterior estava certa:

```
último sinal há 6h: WAIT @ R$ 5.0810  → agora R$ 5.1340  (+1.04%) ✓
```

Isso te dá feedback constante — você vê com seus próprios olhos quando o modelo acerta e quando erra, sem precisar abrir planilha nenhuma. O histórico fica todo lá, em um CSV simples que você pode abrir no Excel ou no Numbers.

O arquivo não vai pro Git (já está no `.gitignore`) — é só do seu computador.

---

### passo 9 — backtest no seu calendário (opcional mas recomendado)

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

**Dois ajustes que valem ouro pra ser realista:**

- `--deadline-days 15` — você não pode esperar pra sempre. Esse parâmetro força o câmbio depois de N dias mesmo se o sinal estiver dizendo pra aguardar (padrão: 15, que é ≈ 2 decisões por mês).
- `--spread-bps 50` — sua corretora cobra um spread (Wise, Higlobe, Remessa Online … todas tiram um pedacinho). 50 = 0,50 %. O backtest desconta isso pra mostrar quanto BRL você **realmente** receberia.

```bash
.venv/bin/python fx_timing.py --lang pt --backtest --days 5 20 --deadline-days 15 --spread-bps 50
```

---

### da próxima vez

Você não precisa repetir toda a configuração. Na próxima vez que quiser consultar, basta:

1. Abrir o terminal
2. Entrar na pasta: `cd cambio`
3. Rodar: `.venv/bin/python fx_timing.py --lang pt`

---

### resumo dos comandos

| O que fazer | Comando |
|---|---|
| Análise de hoje | `... fx_timing.py --lang pt` |
| Análise + alerta no navegador | `... fx_timing.py --lang pt --notify` |
| Background, alerta automático | `... fx_timing.py --lang pt --watch --notify` |
| Background a cada 15 min | `... fx_timing.py --lang pt --watch --notify --watch-interval 15` |
| Backtest padrão | `... fx_timing.py --backtest` |
| Backtest no seu dia | `... fx_timing.py --backtest --days 10` |
| Backtest com deadline e spread | `... fx_timing.py --backtest --deadline-days 15 --spread-bps 50` |

> No Mac/Linux substitua `...` por `.venv/bin/python`. No Windows, `.venv\Scripts\python`.

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

## contribuindo

PRs são bem-vindos. Direções de alto valor:

- CDS 5Y do Brasil ou spread do EMBI+ como sinal de risco soberano
- Calendário do COPOM como filtro de eventos de volatilidade
- Momentum cross-sectional de FX em emergentes (ZAR, MXN, CLP)
- Classificador de regime HMM de 2 estados pra substituir o ADX

Inclua um diff de acurácia do backtest em qualquer PR de sinal.

---

> ⚠️ **Aviso legal.** Esta ferramenta fornece análise probabilística de sinais de mercado publicamente disponíveis, apenas para fins informativos. Não é aconselhamento financeiro, recomendação de investimento, nem solicitação para comprar ou vender qualquer moeda ou ativo. O desempenho histórico do modelo não garante resultados futuros. Sempre consulte um profissional financeiro licenciado antes de tomar decisões de câmbio. Use por sua conta e risco. Veja [LICENSE](LICENSE).

---
---

## English

You receive a USD payment. You need BRL. Do you convert today, or wait?

**cambio** fetches live market data, runs 10+ quant signals across macro, technical, and carry dimensions, and gives you a probability-graded answer — calibrated to *your* payment schedule, your spread, and your real-world deadline. It can also sit in the background and ping you in the browser the moment the dollar looks ready to fall.

> **v0.2.0** — background mode (`--watch`), browser alerts (`--notify`), local journal with self-feedback, intraday-aware signals, deadline/spread-aware backtest, three-way decision (NOW / SPLIT / WAIT).

### quick start

```bash
git clone https://github.com/vitor-araujo/cambio.git && cd cambio
python3 -m venv .venv && .venv/bin/pip install -q yfinance pandas numpy
.venv/bin/python fx_timing.py
```

No API keys. Live data from Yahoo Finance and the [BCB open API](https://dadosabertos.bcb.gov.br/).

### features

- **10+ signals** — momentum, carry, mean-reversion
- **ADX-conditioned** RSI and Bollinger %B (auto-suppressed in trending markets)
- **Three-way verdict** — NOW / SPLIT / WAIT, probability-graded language
- **Intraday-aware** — last bar replaced with the live tick before signals run
- **Deadline-aware backtest** (`--deadline-days`) — forces execution at your real-world cutoff
- **Cost-aware backtest** (`--spread-bps`) — BRL figures net of fintech spread
- **Browser alerts** (`--notify`) — Fraunces/JetBrains-Mono HTML page with one-click link to Higlobe
- **Background mode** (`--watch`) — leave it running, get pinged on flip-to-NOW
- **Journal** — every call logged to `.fx_journal.csv`; next run audits the previous one
- **Walk-forward backtest** (`--backtest`) on your own payment schedule via `--days`
- **Português** output via `--lang pt`

### signals

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

### backtest

Walk-forward, no look-ahead. Oracle = rate at the **next scheduled check date** — not a theoretical intraday price you couldn't act on.

| Schedule | NOW accuracy | WAIT accuracy | Calls |
|---|---|---|---|
| 2nd & 17th (default) | 58.3 % | 44.4 % | 102 |
| 5th & 20th | **75.0 %** | 42.2 % | 103 |

**Accuracy varies by schedule.** Run the backtest on your actual payment dates first.

```bash
.venv/bin/python fx_timing.py --backtest --days 5 20 --deadline-days 15 --spread-bps 50
```

### options

```
fx_timing.py [--backtest] [--days DAY ...] [--lang {en,pt}]
             [--deadline-days N] [--spread-bps BPS]
             [--notify] [--watch] [--watch-interval MIN] [--name NAME]

  --backtest             walk-forward backtest since 2022
  --days 5 20            day(s) of month you typically decide (default: 2 17)
  --lang pt              output in Portuguese
  --deadline-days 15     forced execution window in days (default: 15)
  --spread-bps 50        effective FX spread in basis points (default: 50 = 0.50 %)
  --notify               open an HTML alert in the browser when p(NOW) is high
  --watch                background mode — re-run on a schedule, alert on flip
  --watch-interval 60    minutes between checks in --watch mode
  --name Vitor           name shown in the browser alert headline
```

### contributing

PRs welcome. High-value directions:

- Brazil 5Y CDS or EMBI+ spread as sovereign risk signal
- COPOM calendar as a volatility event filter
- Cross-sectional EM FX momentum (ZAR, MXN, CLP)
- HMM 2-state regime classifier to replace ADX

Include a backtest accuracy diff in any signal PR.

> ⚠️ **Disclaimer.** This tool provides probabilistic analysis of publicly available market signals for informational purposes only. It is not financial advice, investment advice, or a solicitation to buy or sell any currency or asset. Past model performance does not guarantee future results. Always consult a licensed financial professional before making currency exchange decisions. Use at your own risk. See [LICENSE](LICENSE).
