# Guia para leigos 🇧🇷

Nunca mexeu com programação? Esse guia parte do zero. O **cambio** é um script Python — um programinha de terminal que roda no seu computador, busca dados na internet e te diz o resultado. Não tem site, app ou instalador.

> Você vai usar no máximo 5 comandos diferentes. Calma que dá.

---

## O terminal

Janela onde você digita instruções pro computador em vez de clicar em botões.

* **Mac:** `Cmd + Espaço` → digite `Terminal` → Enter
* **Windows:** tecla Windows → digite `PowerShell` → Enter

---

## Passo 1

### Instalar o Python

1. Acesse [python.org/downloads](https://python.org/downloads)
2. Clique no botão amarelo de download
3. Abra o instalador

> ⚠️ **Windows:** na primeira tela, **marque "Add Python to PATH"** antes de continuar. Sem isso, nada funciona.

Confirme no terminal:

```bash
python3 --version
```

Deve aparecer `Python 3.12.x` ou similar.

---

## Passo 2

### Baixar o projeto

```bash
git clone https://github.com/vitor-araujo/cambio.git
```

Cria uma pasta `cambio` no seu diretório atual. Sem `git` instalado? Baixe em [git-scm.com](https://git-scm.com), ou use o botão **Code → Download ZIP** no [GitHub](https://github.com/vitor-araujo/cambio).

---

## Passo 3

### Entrar na pasta

```bash
cd cambio
```

Confirme que está no lugar certo:

```bash
ls          # Mac/Linux
dir         # Windows
```

Deve listar `fx_timing.py`, `signals.py`, `journal.py`, `notify.py`, `README.md`.

---

## Passo 4

### Criar o ambiente e instalar dependências

```bash
python3 -m venv .venv
.venv/bin/pip install yfinance pandas numpy
```

> **Windows:** troque `.venv/bin/` por `.venv\Scripts\`.

Esse passo é feito **uma única vez**. Próxima execução, pula direto pro passo 5.

---

## Passo 5

### Primeira análise

```bash
.venv/bin/python fx_timing.py --lang pt
```

5–10 segundos depois, aparece a tabela de sinais e a recomendação. Pronto.

---

## Passo 6

### Alerta no navegador

```bash
.venv/bin/python fx_timing.py --lang pt --notify
```

Quando os sinais convergirem com convicção (`p(AGORA) ≥ 55 %`), abre uma página HTML no navegador com:

* a taxa atual ao vivo
* os 5 sinais mais relevantes
* botão direto pra abrir o Higlobe

Cooldown de 6 horas — não reabre durante esse período.

---

## Passo 7

### Modo background

A forma mais prática no dia-a-dia. Deixa rodando, ele te avisa.

```bash
.venv/bin/python fx_timing.py --lang pt --watch --notify
```

* Re-roda a cada 60 minutos (use `--watch-interval 15` pra checar a cada 15 min)
* Registra cada ciclo no diário
* Abre o alerta no navegador quando o sinal vira pra **AGORA**
* `Ctrl+C` pra parar

```
[14:32] wait          p_now=0.31  R$ 5.0810  ·
[15:32] exchange_now  p_now=0.62  R$ 5.1340  ◈ ALERT
```

> **Dica:** rode dentro de `tmux` ou em uma aba dedicada do Terminal.

---

## Passo 8

### Diário + auditoria de comportamento

Toda execução adiciona uma linha em `.fx_journal.csv`. Na rodada seguinte, aparece no topo:

```
último sinal há 6h: WAIT @ R$ 5.0810  →  agora R$ 5.1340  (+1.04%) ✓
```

Depois que você fecha um câmbio na corretora, marca a entrada para o cronômetro de prazo funcionar:

```bash
.venv/bin/python fx_timing.py --mark-executed
```

Quer ver quantos alertas você ignorou? (a *behavior gap* que destrói mais valor que más previsões — Morningstar, *Mind the Gap*)

```bash
.venv/bin/python fx_timing.py --audit 30
```

Mostra: alertas disparados, câmbios marcados, alertas ignorados, taxa de override.

O CSV abre direto no Excel ou Numbers se quiser ver o histórico completo. O arquivo não vai pro Git — é só seu.

---

## Passo 9

### Backtest no seu calendário

Mostra como o modelo teria performado nos *seus* dias específicos de decisão. A acurácia varia muito com o calendário.

```bash
# recebe todo dia 5
.venv/bin/python fx_timing.py --lang pt --backtest --days 5

# recebe nos dias 10 e 25
.venv/bin/python fx_timing.py --lang pt --backtest --days 10 25
```

**Três ajustes essenciais para realismo:**

* `--deadline-days 15` — força o câmbio depois de N dias mesmo se o sinal disser pra aguardar.
* `--spread-bps 50` — desconta o spread da corretora (Wise, Higlobe, etc). 50 = 0,50 %.
* `--dca-floor 0.25 --dca-ceiling 1.00` — disciplina Vanguard-DCA. Cada checkpoint converte uma fração entre 25 % e 100 % do que ainda falta, baseada na convicção do modelo. Nunca zero, nunca tudo no escuro.

```bash
.venv/bin/python fx_timing.py --lang pt --backtest --days 5 20 --deadline-days 15 --spread-bps 50
```

No final do backtest, repare na seção **COST-MATTERS HYPOTHESIS** — ela compara a vantagem do modelo com 2× o spread. Se a margem for negativa, o relatório diz `INSIDE THE SPREAD`: a vantagem do modelo é menor que o custo da corretora, então estatisticamente é ruído. Princípio Bogle/Vanguard: sem margem de segurança, não é vantagem real.

---

## Da próxima vez

Não precisa repetir a configuração. Só:

```bash
cd cambio
.venv/bin/python fx_timing.py --lang pt
```

Ou, mais útil, deixa o background ligado:

```bash
.venv/bin/python fx_timing.py --lang pt --watch --notify
```

---

## Resumo dos comandos

| Tarefa | Comando |
|---|---|
| Análise pontual | `... --lang pt` |
| Análise + alerta | `... --lang pt --notify` |
| Background com alerta | `... --lang pt --watch --notify` |
| Background a cada 15 min | `... --lang pt --watch --notify --watch-interval 15` |
| Backtest padrão | `... --backtest` |
| Backtest no seu dia | `... --backtest --days 10` |
| Backtest com deadline e spread | `... --backtest --deadline-days 15 --spread-bps 50` |

> Mac/Linux: `...` = `.venv/bin/python fx_timing.py`. Windows: `.venv\Scripts\python fx_timing.py`.

---

## Troubleshooting

| Erro | Solução |
|---|---|
| `python3: command not found` | Reinstale o Python e marque "Add to PATH" no Windows |
| `No module named pip` | `python3 -m ensurepip` e repita o passo 4 |
| `ModuleNotFoundError: yfinance` | Você rodou sem ativar o venv. Use `.venv/bin/python` |
| Tela em branco / trava | Verifique a conexão com a internet — os dados são ao vivo |
| Erro com `\` vs `/` no Windows | Sempre use `\` no Windows: `.venv\Scripts\python` |

---

← [voltar ao README](../README.md)
