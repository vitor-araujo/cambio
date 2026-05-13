<p align="center">
  <h1 align="center">cambio</h1>
  <p align="center"><em>Timing quantitativo para câmbio USD → BRL.</em></p>
  <p align="center">
    <a href="https://github.com/vitor-araujo/cambio/blob/main/LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square"></a>
    <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-4B8BBE?style=flat-square&logo=python&logoColor=white">
    <img alt="versão 0.4.4" src="https://img.shields.io/badge/version-0.4.4-fbbf24?style=flat-square">
    <img alt="sem API key" src="https://img.shields.io/badge/dados-grátis%20·%20sem%20API%20key-f59e0b?style=flat-square">
    <img alt="pt · en" src="https://img.shields.io/badge/lang-pt--BR%20·%20en-6366f1?style=flat-square">
  </p>
</p>

---

**cambio** é um modelo probabilístico que decide **quanto** você converte de dólar a cada ciclo. Roda 12 sinais quantitativos sobre dados públicos, aplica disciplina Vanguard-DCA (nunca zero, nunca 100 % no escuro), e te avisa no navegador quando a convicção estiver alta o suficiente para subir o tamanho da conversão.

Documentação dos sinais: [`signals.py`](signals.py) · Código-fonte: [github.com/vitor-araujo/cambio](https://github.com/vitor-araujo/cambio)

---

## Recursos

* 🎯 **Sizing Vanguard-DCA** — cada ciclo converte uma fração entre `--dca-floor` (25 %) e `--dca-ceiling` (100 %), proporcional à convicção. Nunca zero, nunca tudo no escuro — minimiza variância de arrependimento.
* ⚖️ **Validador Cost-Matters** — o backtest declara se a vantagem do modelo sobrepassa **2× spread** (margem de segurança Bogle). Caso contrário, sinaliza "INSIDE THE SPREAD".
* ⏱️ **Cronômetro de prazo** — o `--watch` mostra quantos dias faltam até a execução forçada, ancorado em `--mark-executed`.
* 📊 **Auditoria de comportamento** — `--audit` mostra quantos alertas você ignorou nos últimos 30 dias (sua *behavior gap*).
* 📡 **Dados ao vivo** — PTAX comercial (BCB), AwesomeAPI (intraday), Yahoo (DXY/Brent/VIX/etc), CFTC (COT), SELIC. Zero API keys.
* 🔄 **Ajuste intraday** — a última cotação substitui a barra do dia antes do cálculo dos sinais.
* 🛎️ **Alerta no navegador** — página HTML com cartão de tamanho de conversão, prazo restante e link direto pra Higlobe.
* 📱 **Alertas no celular** — mensagem automática quando USD/BRL sobe além do seu limite (`--phone-alerts`). Telegram por padrão, WhatsApp opcional via Twilio. Configure com `python configure.py`.
* 👁️ **Modo background** — fica rodando, te avisa só quando o sinal vira.
* 📓 **Diário automático** — toda decisão fica em `.fx_journal.csv` com `size`, `notified` e `executed`.
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

Deixe rodando em uma aba do terminal — alerta automático quando a convicção subir:

```bash
.venv/bin/python fx_timing.py --lang pt --watch --notify
```

Cada ciclo imprime tamanho sugerido + dias até prazo:

```
[14:32] wait          p_now=0.31  size=25%  R$ 5.0810  prazo=11d  ·
[15:32] exchange_now  p_now=0.62  size=81%  R$ 5.1340  prazo=10d  ◈ ALERT
```

### Alertas no celular (`--phone-alerts`)

Receba uma mensagem sempre que o USD/BRL subir além do limite. Provedor padrão: **Telegram** (grátis, ilimitado, oficial). WhatsApp via Twilio disponível como opção.

```bash
python configure.py                       # CLI interativa (escolhe o provedor)
python configure.py --provider telegram   # pula a pergunta de provedor
python configure.py --test                # envia mensagem de teste
python configure.py --reset               # zera âncora e cooldown
```

O CLI cria um `.env` local (modo 600, no `.gitignore` — nunca vai pro repo). Depois é só rodar com `--phone-alerts`:

```bash
.venv/bin/python fx_timing.py --lang pt --watch --notify --phone-alerts
```

#### Setup Telegram (2 min, grátis)

1. No Telegram, abra a conversa com `@BotFather`
2. Envie `/newbot` → nome qualquer → username terminando em `bot`
3. O BotFather te dá um **token** (`1234567890:AAH...`)
4. Rode `python configure.py`, cole o token, mande qualquer mensagem pro bot quando pedido
5. O CLI descobre seu **chat_id** automaticamente via `getUpdates`
6. Aceita o teste — chega uma mensagem na hora

O `.env` fica assim:

```env
NOTIFIER_PROVIDER=telegram
TELEGRAM_BOT_TOKEN=1234567890:AAH...
TELEGRAM_CHAT_ID=123456789
FX_ALERT_THRESHOLD_PCT=1.0
FX_ALERT_COOLDOWN_MIN=5
```

#### Setup WhatsApp (opcional, via Twilio)

WhatsApp Cloud API direto da Meta exige uma conta business com verificação que atualmente bloqueia muitos devs solo sem CNPJ. O caminho viável é Twilio:

1. Conta em [twilio.com/try-twilio](https://www.twilio.com/try-twilio)
2. **Messaging → Try it out → Send a WhatsApp message** → ativa o sandbox
3. Do seu WhatsApp, manda `join <duas-palavras>` para `+1 415 523 8886`
4. Pega **Account SID** e **Auth Token** no console
5. Rode `python configure.py --provider whatsapp` e cole os valores

#### Como funciona a âncora

No primeiro tick, o preço atual vira âncora. Cada novo tick é comparado contra ela. Se cair, a âncora desce junto (rastreia a mínima local). Se subir além do `threshold` e o cooldown já expirou, dispara o alerta e a âncora se move pro preço atual — o próximo alerta exige outra alta de `+threshold` da nova base.

Estado fica em `.fx_alert.state` (gitignored) e sobrevive a restarts. Use `--reset` após um câmbio manual.

### Marcar câmbio executado

Depois que você fecha o câmbio na corretora, ancora o cronômetro de prazo:

```bash
.venv/bin/python fx_timing.py --mark-executed
```

### Auditoria do comportamento

Quantos alertas você ignorou? (a *behavior gap* que destrói mais valor que más previsões)

```bash
.venv/bin/python fx_timing.py --audit 30
```

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
             [--dca-floor FRAC] [--dca-ceiling FRAC]
             [--notify] [--watch] [--watch-interval MIN] [--name NOME]
             [--phone-alerts] [--alert-provider {telegram,whatsapp}]
             [--alert-threshold PCT] [--alert-cooldown MIN]
             [--mark-executed] [--audit [DAYS]]
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--backtest` | — | Roda walk-forward desde 2022 |
| `--days` | `2 17` | Dia(s) do mês em que você decide |
| `--lang` | `en` | Idioma da saída (`en` ou `pt`) |
| `--deadline-days` | `15` | Prazo limite em dias para forçar execução |
| `--spread-bps` | `50` | Spread efetivo da corretora em basis points |
| `--dca-floor` | `0.25` | Fração mínima a converter a cada ciclo (Vanguard-DCA) |
| `--dca-ceiling` | `1.00` | Fração máxima quando a convicção é alta |
| `--notify` | — | Abre alerta HTML no navegador em flip-to-AGORA |
| `--watch` | — | Modo background — re-roda em loop |
| `--watch-interval` | `5` | Minutos entre checagens em `--watch` |
| `--name` | `Vitor` | Nome no cabeçalho do alerta no navegador |
| `--phone-alerts` | — | Habilita alertas no celular (Telegram/WhatsApp) |
| `--alert-provider` | `telegram` | Provedor: `telegram` ou `whatsapp` (override do `.env`) |
| `--alert-threshold` | `1.0` | % de alta vs âncora que dispara o alerta |
| `--alert-cooldown` | = `--watch-interval` | Minutos mínimos entre alertas (casa com a interval por padrão) |
| `--mark-executed` | — | Marca última entrada do diário como executada (ancora o cronômetro) |
| `--audit` | `30` | Imprime auditoria de *behavior gap* dos últimos N dias |

---

## Estrutura do projeto

Quatro módulos. Cada um com responsabilidade única.

### `fx_timing.py` — orquestrador

Ponto de entrada. Junta tudo:

* parsing de CLI e dispatch (`main`) — inclui `--mark-executed` e `--audit` como subcomandos
* download dos dados (`fetch`, `fetch_ptax`, `fetch_selic`, `fetch_cot_eur`, `_fetch_live_fx`)
* aplicação do tick intraday (`_apply_intraday`)
* probabilidades e regime (`probabilities`, `apply_regime`)
* **sizing Vanguard-DCA** (`size`, `decide`) — fração de conversão ancorada em piso/teto
* renderização do terminal (`render_live`, `render_backtest`) — inclui seção Cost-Matters Hypothesis
* simulação de conversões parciais (`run_backtest`, `sequential_sim`) com `size_frac` por checkpoint
* loop background (`_run_live_cycle`, `_watch_loop`) — mostra cronômetro de prazo a cada ciclo
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
* `last_entry()` / `last_notified()` / `last_executed()` — acessadores de cauda
* `mark_executed(when=None)` — flipa `executed=True` na entrada (mais recente ou pelo timestamp)
* `render_summary(prev, cur_rate)` — linha "último sinal há Xh: WAIT @ R$ 5.08 → agora R$ 5.13 (+1.04 %) ✓"
* `should_notify(decision, p_now)` — aplica threshold + cooldown de 6 h
* `days_until_deadline(deadline_days)` — dias restantes ancorados na última execução
* `audit_summary(days)` — alertas vs execuções vs taxa de override (a *behavior gap* de Morningstar)

### `notify.py` — alerta HTML

Renderiza `.fx_alert.html` e abre no navegador padrão.

* Tipografia: **Fraunces** (serif) no headline, **JetBrains Mono** nos números
* Paleta: verde-floresta + âmbar (Brasil), gradientes em camadas, grid sutil
* Animações CSS-only — fade-in escalonado, pulse no CTA
* `render_alert(...)` — escreve o HTML
* `open_in_browser(path)` — abre via `webbrowser.open`
* `alert(...)` — atalho que faz os dois

### `notifiers/` — backends de mensageria (pluggable)

Arquitetura SOLID: um Protocol pequeno + uma fábrica. Adicionar provedor novo é escrever um arquivo e registrar no `__init__.py` — nada mais muda no projeto.

* `notifiers/base.py` — Protocol `Notifier` com `send`, `is_configured`, `missing_keys`
* `notifiers/telegram.py` — `TelegramNotifier` + `discover_chat_id` (auto-detecção via `getUpdates`)
* `notifiers/whatsapp.py` — `WhatsAppNotifier` (Twilio) — mantido pro dia que WhatsApp ficar viável
* `notifiers/__init__.py` — `get_notifier(provider)` + registry

### `rate_alert.py` — disparador de alertas (provider-agnostic)

Depende apenas do Protocol `Notifier`, não de classes concretas (DIP).

* `load_env(path)` — popula `os.environ` a partir do `.env`
* `maybe_alert_on_rise(rate, *, notifier, ...)` — âncora + threshold + cooldown
* `reset_state()` — limpa `.fx_alert.state`

### `configure.py` — setup CLI

CLI interativa que gera o `.env` (modo 600, gitignored).

* Pergunta o provedor primeiro (Telegram default, WhatsApp opcional)
* Telegram: auto-descobre o `chat_id` via `getUpdates` após o usuário mandar uma msg pro bot
* WhatsApp: pede telefone, SID, token (com `getpass`)
* `--test` envia mensagem, `--reset` limpa âncora, `--provider` pula a pergunta
* Nada sensível **nunca** entra no repositório

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

Quando `--notify` está ativo e `p_now ≥ 0.40` (calibrado a partir de 2 meses de uso), abre `.fx_alert.html` no navegador. Cooldown de 6 horas evita spam. Os dois arquivos ficam locais — não vão pro Git.

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

**cambio** is a probabilistic model that decides **how much** USD to convert each cycle. It runs 12 quant signals over public data, applies Vanguard-DCA discipline (never zero, never blindly all-in), and pings you in the browser when conviction is high enough to scale up the conversion size.

### Features

* 🎯 **Vanguard-DCA sizing** — every cycle converts a fraction in `[--dca-floor, --dca-ceiling]` proportional to conviction. Never zero, never blindly all-in.
* ⚖️ **Cost-Matters validator** — the backtest declares whether the model edge clears **2× spread** (Bogle margin of safety). Otherwise it flags "INSIDE THE SPREAD".
* ⏱️ **Deadline countdown** — `--watch` shows days remaining until forced execution, anchored on `--mark-executed`.
* 📊 **Behavior-gap audit** (`--audit`) — how many alerts you ignored in the last N days.
* 📡 **Live data** — BCB PTAX, AwesomeAPI (intraday), Yahoo, CFTC (COT), SELIC. No API keys.
* 🔄 **Intraday-aware** — last bar replaced with the live tick before signals run.
* 🛎️ **Browser alerts** (`--notify`) — HTML page with size card, deadline countdown and one-click Higlobe link.
* 📱 **Phone alerts** (`--phone-alerts`) — ping your phone when USD/BRL rises past your threshold. Telegram by default, WhatsApp via Twilio optional. Setup via `python configure.py`.
* 👁️ **Background mode** (`--watch`).
* 📓 **Auto journal** — every call logged to `.fx_journal.csv` with `size`, `notified`, `executed`.

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

# add phone alerts on rate spikes (Telegram by default, setup once)
python configure.py
.venv/bin/python fx_timing.py --watch --notify --phone-alerts

# backtest on your real schedule
.venv/bin/python fx_timing.py --backtest --days 5 20 --deadline-days 15 --spread-bps 50
```

### Phone alerts setup

The `configure.py` CLI asks which provider you want and walks through the setup. Credentials are saved to a local `.env` (mode 600, gitignored — **nothing sensitive ever enters the repo**).

```bash
python configure.py                       # interactive (picks provider)
python configure.py --provider telegram   # skip the provider prompt
python configure.py --test                # send a test message
python configure.py --reset               # wipe anchor + cooldown
```

#### Telegram (default, 2 min, free)

1. In Telegram, open `@BotFather`
2. Send `/newbot` → any name → username ending in `bot`
3. BotFather gives you a **token** like `1234567890:AAH...`
4. Run `python configure.py`, paste the token, send any message to your bot when asked
5. The CLI auto-discovers your **chat_id** via `getUpdates`
6. Accept the test — you'll get a message instantly

#### WhatsApp (optional, via Twilio)

Meta's WhatsApp Cloud API requires business verification that currently blocks many indie devs without a registered company. The viable path is Twilio:

1. Account at [twilio.com/try-twilio](https://www.twilio.com/try-twilio)
2. **Messaging → Try it out → Send a WhatsApp message** → enable sandbox
3. From your WhatsApp, send `join <two-words>` to `+1 415 523 8886`
4. Grab **Account SID** and **Auth Token** from the console
5. Run `python configure.py --provider whatsapp` and paste them in

#### Anchor logic

The first observed rate becomes the anchor. Each tick is compared to it. If the rate falls, the anchor follows down (tracks the local low). If it rises beyond `--alert-threshold` and the cooldown has elapsed, an alert fires and the anchor jumps to the current rate — so the next alert requires another `+threshold` rally from there.

State is persisted in `.fx_alert.state` (gitignored) and survives restarts. Use `--reset` after a manual exchange to clear it.

### CLI reference

```
fx_timing.py [--backtest] [--days DAY ...] [--lang {en,pt}]
             [--deadline-days N] [--spread-bps BPS]
             [--dca-floor FRAC] [--dca-ceiling FRAC]
             [--notify] [--watch] [--watch-interval MIN] [--name NAME]
             [--phone-alerts] [--alert-provider {telegram,whatsapp}]
             [--alert-threshold PCT] [--alert-cooldown MIN]
             [--mark-executed] [--audit [DAYS]]
```

| Flag | Default | Description |
|---|---|---|
| `--backtest` | — | Walk-forward backtest since 2022 |
| `--days` | `2 17` | Day(s) of month you typically decide |
| `--lang` | `en` | Output language (`en` or `pt`) |
| `--deadline-days` | `15` | Forced-execution window |
| `--spread-bps` | `50` | Effective FX spread in basis points |
| `--dca-floor` | `0.25` | Minimum fraction to convert each cycle (Vanguard-DCA) |
| `--dca-ceiling` | `1.00` | Maximum fraction when conviction is high |
| `--notify` | — | Open HTML alert on flip-to-NOW |
| `--watch` | — | Background mode — re-run on a schedule |
| `--watch-interval` | `5` | Minutes between checks in `--watch` |
| `--name` | `Vitor` | Name shown in the browser alert |
| `--phone-alerts` | — | Enable phone alerts (Telegram or WhatsApp) |
| `--alert-provider` | `telegram` | Backend: `telegram` or `whatsapp` (overrides `.env`) |
| `--alert-threshold` | `1.0` | % rise vs anchor that fires an alert |
| `--alert-cooldown` | = `--watch-interval` | Minutes between consecutive alerts (matches the watch interval by default) |
| `--mark-executed` | — | Mark the most recent journal entry as executed |
| `--audit` | `30` | Print behavior-gap audit for the last N days |

### Project layout

| Module | Role |
|---|---|
| `fx_timing.py` | Entry point — CLI, data fetch, signal pipeline, render, journal, notify, watch loop |
| `signals.py` | Pure signal library — RSI, Bollinger %B, ADX, carry score, `build_signals` |
| `journal.py` | Append-only CSV log of decisions, last-call summary, notify cooldown |
| `notify.py` | HTML alert renderer (Fraunces + JetBrains Mono) and browser opener |
| `notifiers/` | Pluggable messaging backends: `Notifier` Protocol + Telegram + WhatsApp |
| `rate_alert.py` | Provider-agnostic spike alert (anchor/threshold/cooldown) |
| `configure.py` | Interactive CLI — picks provider, writes `.env` (gitignored, mode 600) |

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
