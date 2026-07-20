<p align="center">
  <img src="docs/brand/cambio-mark.png" alt="Símbolo do cambio" width="148" />
</p>

<h1 align="center">cambio</h1>

<p align="center">
  <strong>Disciplina de execução para quem vende USD e recebe BRL.</strong><br />
  Uma mesa pessoal que transforma sinais de mercado em tranches pequenas, recorrentes e auditáveis.
</p>

<p align="center">
  <a href="CHANGELOG.md"><img alt="versão 0.8.0" src="https://img.shields.io/badge/version-0.8.0-D6924B?style=for-the-badge"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10+-3195FF?style=for-the-badge&logo=python&logoColor=white">
  <img alt="sem API key" src="https://img.shields.io/badge/dados-públicos-10283C?style=for-the-badge">
  <a href="LICENSE"><img alt="licença MIT" src="https://img.shields.io/badge/license-MIT-53D6A0?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="#comece-em-um-minuto">Começar</a> ·
  <a href="#como-a-execução-funciona">Como funciona</a> ·
  <a href="#linha-de-comando">CLI</a> ·
  <a href="#api-local">API</a> ·
  <a href="#english">English</a>
</p>

---

## O problema que o cambio resolve

Quem recebe em dólar quase sempre enfrenta a mesma dúvida: converter agora ou
esperar um preço melhor? Um modelo que apenas responde “espere” parece prudente,
mas pode paralisar a execução por semanas.

O **cambio** separa previsão de disciplina:

- os sinais de médio prazo definem **quanto** converter;
- a qualidade recente do USD/BRL ajuda a escolher **quando**, dentro de uma
  janela curta;
- um relógio de execução impede que a busca pelo preço perfeito vire inércia.

O resultado não é uma promessa de acertar o topo. É um processo repetível para
trazer dólares para reais em lotes pequenos, registrar cada decisão e limitar o
arrependimento de timing.

<table>
  <tr>
    <td width="25%"><strong>⏱ 3–4 dias</strong><br />janela limitada entre tranches</td>
    <td width="25%"><strong>◫ 25–50%</strong><br />tamanho ajustado à convicção</td>
    <td width="25%"><strong>⌁ 12 sinais</strong><br />mercado, carry e posicionamento</td>
    <td width="25%"><strong>✓ Auditável</strong><br />ledger local em SQLite</td>
  </tr>
</table>

## Comece em um minuto

Você precisa de Python 3.10+, Node.js e npm.

```bash
git clone https://github.com/vitor-araujo/cambio.git
cd cambio

python3 -m venv .venv
source .venv/bin/activate
pip install yfinance pandas numpy

cd ui && npm install && cd ..
python server.py --dev
```

Abra [localhost:5173](http://localhost:5173). O servidor busca os dados
imediatamente, mantém a cotação viva e atualiza o ledger em segundo plano.

> No Windows, ative o ambiente com `.venv\Scripts\activate`.

### Sem interface gráfica

```bash
python fx_timing.py --watch --notify --phone-alerts
```

O modo contínuo mantém uma linha compacta no terminal e só imprime novamente
quando o estado muda ou quando chega o heartbeat:

```text
11:42:08  EXECUTE  R$ 5.1003 · quality 52% / hurdle 40% · p(now) 31% · tranche 26%
```

## Como a execução funciona

```text
12 sinais de médio prazo ──► p(agora) ─────────► tamanho da tranche
                                                       │
percentil USD/BRL em 20 sessões ──► qualidade ────────┤
                                                       ▼
relógio desde a última execução ──► janela 3–4 dias ─► instrução + ledger
```

### 1. Direção dimensiona a tranche

O modelo combina momentum, carry, reversão e posicionamento. A probabilidade
`p(agora)` não decide sozinha se uma operação pode acontecer; ela ajusta o lote
entre **25% e 50%** do saldo disponível.

### 2. Qualidade escolhe o ponto da janela

Para uma venda de USD, uma taxa USD/BRL mais alta é mais favorável. O seletor
calcula o percentil da cotação atual nas últimas 20 sessões e o compara com um
limiar que diminui à medida que o prazo se esgota.

### 3. A cadência impede espera infinita

| Momento | Política |
|---|---|
| Após executar | preserva opcionalidade e monitora o mercado |
| Terceiro dia | abre a janela; executa se a qualidade superar o limiar |
| Quarto dia | executa a tranche programada, mesmo sem sinal forte |
| Após marcar | registra a operação e reinicia o relógio |

Se o processo ficar desligado, o intervalo real pode ultrapassar quatro dias.
Ao voltar, uma tranche vencida aparece como ação imediata.

## A mesa de execução

O dashboard foi desenhado como uma pequena mesa institucional, não como um
painel genérico de investimentos.

| Superfície | O que entrega |
|---|---|
| **Ticket** | ação atual, percentual, saldo USD, BRL estimado e corretora |
| **Relógio** | abertura da janela, prazo restante e motivo da instrução |
| **Mercado** | USD/BRL, IBOV, âncora do dia e histórico recente |
| **Gráfico** | preço, qualidade de execução, limiar e `p(agora)` |
| **Ledger** | sinais, tranches, alertas, gatilhos e execuções registradas |
| **Controles** | mandato de risco validado e persistido atomicamente |
| **CLI** | comandos prontos para copiar e prévia do modo contínuo |

Marque uma tranche pelo ticket ou pelo terminal:

```bash
python fx_timing.py --mark-executed
```

Se o registro foi acidental, a interface oferece **Desfazer** imediatamente.

## Evidência, sem falsa precisão

O backtest walk-forward mantém cada decisão no tempo e separa o modelo
direcional da política de cadência.

Snapshot de **20/07/2026**, com 284 janelas históricas de três a quatro dias:

| Métrica | Resultado |
|---|---:|
| Execuções no terceiro dia | 61,6% |
| Escolha do melhor dia entre 3 e 4 | 56,7% |
| Diferença vs. sempre esperar o dia 4 | +6,1 bps |
| Diferença vs. sempre executar no dia 3 | −1,4 bps |
| Holdout 2025+ vs. dia 4 | +16,4 bps |
| Bootstrap 95% do holdout vs. dia 4 | +9,0 a +24,8 bps |

Isso sustenta a política contra **sempre esperar até o quarto dia**. Não prova
alfa, não garante o melhor câmbio e não elimina spread, IOF ou risco de mercado.

Rode a avaliação no seu próprio calendário:

```bash
python fx_timing.py --backtest --days 5 20
```

## Dados e sinais

O projeto usa dados públicos e não exige chave de API.

| Fonte | Uso |
|---|---|
| BCB PTAX | referência diária USD/BRL |
| AwesomeAPI | cotação intradiária |
| Yahoo Finance | DXY, Brent, VIX, IBOV e VALE |
| CFTC COT | posicionamento em dólar |
| BCB / mercado americano | SELIC, T-bill e carry |

Os 12 sinais pertencem a quatro famílias:

| Família | Sinais |
|---|---|
| Momentum | DXY, Brent, VALE, VIX e IBOV |
| Carry | diferencial SELIC − juros americanos |
| Reversão | nível USD/BRL, RSI, Bollinger e tendência |
| Posicionamento | futuros de BRL e COT USD |

Um filtro de regime baseado em ADX reduz o peso de reversão quando a tendência
está forte.

## Alertas e diário

Cada ciclo é salvo em `.fx_journal.db`, incluindo cotação, instrução, tamanho,
qualidade, limiar, gatilho, alerta e status de execução. O arquivo é local e
ignorado pelo Git.

Para receber alertas no Telegram:

```bash
python configure.py
python configure.py --test
```

O alerta intradiário compara a cotação viva com a primeira taxa observada no
dia. A âncora permanece fixa, evitando que alertas anteriores distorçam a base
de comparação.

## Controles

Todos os limites podem ser alterados na aba **Controles** ou pela API. O servidor
valida as faixas antes de persistir a configuração.

| Nome | Padrão | Significado |
|---|---:|---|
| `watch_interval_min` | 5 | minutos entre coletas |
| `notify_threshold` | 0.40 | `p(agora)` mínima para alerta no navegador |
| `notify_cooldown_hours` | 6 | intervalo entre alertas do navegador |
| `alert_threshold_pct` | 1.0 | alta intradiária para alerta no celular |
| `alert_cooldown_min` | 5 | intervalo entre alertas no celular |
| `dca_floor` | 0.25 | tamanho mínimo da tranche |
| `dca_ceiling` | 0.50 | tamanho máximo da tranche |
| `cadence_days` | 4 | limite entre tranches |
| `spread_bps` | 50 | spread efetivo usado na avaliação |
| `deadline_days` | 15 | janela legada para análises agendadas |

## Linha de comando

<details>
<summary><strong>Ver comandos e opções</strong></summary>

```bash
# Análise pontual em português
python fx_timing.py --lang pt

# Monitor contínuo a cada minuto
python fx_timing.py --watch --watch-interval 1

# Monitor + navegador + celular
python fx_timing.py --watch --notify --phone-alerts

# Auditoria de comportamento dos últimos 30 dias
python fx_timing.py --audit 30

# Backtest no seu calendário de recebimentos
python fx_timing.py --backtest --days 5 20
```

| Opção | Padrão | Função |
|---|---:|---|
| `--lang {en,pt}` | `en` | idioma da saída |
| `--watch` | — | modo contínuo |
| `--watch-interval MIN` | `5` | minutos entre verificações |
| `--notify` | — | alerta no navegador |
| `--phone-alerts` | — | alertas Telegram/WhatsApp |
| `--dca-floor FRAC` | `0.25` | tranche mínima |
| `--dca-ceiling FRAC` | `0.50` | tranche máxima |
| `--cadence-days N` | `4` | limite da cadência ao vivo |
| `--spread-bps BPS` | `50` | custo efetivo para avaliação |
| `--mark-executed` | — | ancora o próximo ciclo |
| `--audit [DAYS]` | `30` | auditoria de comportamento |

</details>

## API local

<details>
<summary><strong>Ver endpoints</strong></summary>

Base: `http://127.0.0.1:8765/api`

| Endpoint | Método | Uso |
|---|---|---|
| `/dashboard` | GET | resumo e sinais recentes |
| `/journal?limit=1000` | GET | ledger paginado, até 5.000 entradas |
| `/state` | GET | âncora e cooldown dos alertas |
| `/thresholds` | GET / POST | consultar ou atualizar controles |
| `/notifier` | GET / POST | estado e configuração do Telegram |
| `/notifier/test` | POST | enviar alerta de teste |
| `/executions` | POST | marcar a última tranche |
| `/executions/undo` | POST | desfazer a marcação |
| `/health` | GET | saúde, uptime e idade do último sinal |

</details>

## Verificação

```bash
# Política de execução
python test_execution_policy.py

# API — execute com o servidor rodando
python test_server.py

# TypeScript + build de produção
cd ui && npm run build
```

## Estrutura

| Arquivo | Responsabilidade |
|---|---|
| `fx_timing.py` | dados, sinais, política, backtest e CLI |
| `terminal_ui.py` | interface contínua do terminal |
| `server.py` | API, coleta em background e Vite em desenvolvimento |
| `signals.py` | biblioteca de sinais quantitativos |
| `journal_db.py` | ledger SQLite e auditoria |
| `notify.py` | alerta HTML do navegador |
| `notifiers/` | integrações Telegram e WhatsApp |
| `ui/` | mesa de execução em React + TypeScript |

Para uma instalação guiada, veja o
[guia em português](docs/GUIDE-PT.md). Mudanças por versão estão no
[changelog](CHANGELOG.md).

## Publicar como portfolio

O modo portfolio não usa Python, banco de dados, chaves ou serviços externos.
Ele carrega um snapshot sintético e mantém as interações somente no navegador.

```bash
cd ui
npm ci
npm run build:portfolio
```

Publique a pasta `ui/dist` em qualquer host estático. O repositório também inclui
um workflow pronto para GitHub Pages. Veja o [guia de lançamento](PORTFOLIO.md)
para GitHub Pages, Netlify, Cloudflare Pages e Vercel.

## Contribuindo

Contribuições são bem-vindas. Para novos sinais ou alterações na política,
inclua testes e um comparativo walk-forward. Boas próximas frentes incluem risco
soberano (CDS/EMBI+), calendário do COPOM, momentum de moedas emergentes e um
classificador de regime mais robusto.

## English

<details>
<summary><strong>Read the English overview</strong></summary>

**cambio** is a personal USD/BRL execution desk for people who earn dollars and
convert them into Brazilian reais. It separates medium-term direction from
short-window execution: twelve public-data signals size each tranche, a
20-session USD/BRL percentile ranks the current price, and a bounded cadence
prevents indefinite waiting.

The default policy opens an opportunity window on day three and makes the
scheduled tranche due on day four. Tranche size stays between 25% and 50% of the
available USD balance. Every instruction and execution marker is recorded in a
local SQLite ledger.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install yfinance pandas numpy
cd ui && npm install && cd ..
python server.py --dev
```

Open [localhost:5173](http://localhost:5173), or run the terminal-only workflow:

```bash
python fx_timing.py --watch --notify
```

Historical evaluation supports the cadence relative to always waiting until
day four; it does not establish guaranteed alpha or promise the best available
rate. Include real broker spread, taxes, and operational constraints in any
decision.

</details>

## Licença e aviso

[MIT](LICENSE). Este projeto analisa dados públicos para fins informativos. Não
é recomendação de investimento. Resultados passados não garantem resultados
futuros.
