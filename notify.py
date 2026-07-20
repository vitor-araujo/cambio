"""
Notify — writes an HTML alert page to disk and opens it in the
default browser. Designed to interrupt the user only when the
model has high conviction that USD/BRL is about to fall.

The HTML is intentionally not "AI slop": serif headline + mono
data, dark green palette, CSS-only animations, no dependencies.
"""

import html
import os
import webbrowser
from datetime import datetime
from typing import Optional

ALERT_PATH = ".fx_alert.html"
HIGLOBE_URL = "https://higlobe.com/webapp/en/login"

_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cambio · agora, {name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Fraunces:opsz,wght@9..144,300;9..144,700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-0: #0a1612;
  --bg-1: #0f1f1a;
  --line: #1d3a32;
  --green: #4ade80;
  --amber: #fbbf24;
  --rose: #fb7185;
  --ink: #e7f5ee;
  --muted: #6b7f78;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
  height: 100%;
  background: var(--bg-0);
  color: var(--ink);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  overflow-x: hidden;
}}
body {{
  background:
    radial-gradient(1200px circle at 18% -10%, rgba(74,222,128,.14), transparent 55%),
    radial-gradient(900px circle at 90% 110%, rgba(251,191,36,.10), transparent 60%),
    linear-gradient(180deg, #0a1612 0%, #050b09 100%);
  background-attachment: fixed;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 4rem 1.5rem;
  min-height: 100vh;
}}
body::before {{
  content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .12;
  background-image:
    linear-gradient(var(--line) 1px, transparent 1px),
    linear-gradient(90deg, var(--line) 1px, transparent 1px);
  background-size: 56px 56px;
  -webkit-mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
}}
.card {{
  position: relative;
  max-width: 680px; width: 100%;
  background: rgba(15, 31, 26, .72);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: clamp(2rem, 5vw, 3.2rem);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  animation: fadeIn .9s cubic-bezier(.2,.8,.2,1);
}}
.card::before {{
  content: ""; position: absolute; inset: -1px; border-radius: 6px; pointer-events: none;
  background: linear-gradient(135deg, rgba(74,222,128,.4), transparent 30%, transparent 70%, rgba(251,191,36,.3));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude;
  padding: 1px;
}}
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(14px); }}
  to {{ opacity: 1; transform: none; }}
}}
.tag {{
  font-size: .72rem; letter-spacing: .22em; color: var(--green);
  text-transform: uppercase; margin-bottom: 1.4rem;
  display: inline-flex; align-items: center; gap: .5rem;
  animation: slide .6s .1s ease-out backwards;
}}
.tag .dot {{
  width: 6px; height: 6px; border-radius: 50%; background: var(--green);
  animation: pulse 1.6s ease-in-out infinite;
}}
@keyframes pulse {{
  0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(74,222,128,.6); }}
  50% {{ opacity: .7; box-shadow: 0 0 0 8px rgba(74,222,128,0); }}
}}
@keyframes slide {{
  from {{ opacity: 0; transform: translateX(-10px); }}
  to {{ opacity: 1; transform: none; }}
}}
h1 {{
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 300;
  font-size: clamp(2.4rem, 6.5vw, 4.2rem);
  line-height: 1.02;
  letter-spacing: -.025em;
  margin-bottom: 1.4rem;
  animation: slide .6s .2s ease-out backwards;
}}
h1 strong {{ font-weight: 700; font-style: italic; color: var(--amber); }}
.lede {{
  color: var(--muted);
  font-size: .95rem;
  line-height: 1.65;
  max-width: 54ch;
  margin-bottom: 2.4rem;
  animation: slide .6s .35s ease-out backwards;
}}
.rate {{
  display: flex; align-items: baseline; justify-content: space-between; gap: 1rem;
  padding: 1.3rem 0; margin-bottom: 2rem;
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  animation: slide .6s .45s ease-out backwards;
}}
.rate-num {{
  font-size: clamp(2rem, 5vw, 2.8rem); font-weight: 700; letter-spacing: -.02em;
  font-variant-numeric: tabular-nums;
}}
.rate-lbl {{
  font-size: .72rem; color: var(--muted); letter-spacing: .18em; text-transform: uppercase;
}}
.signals {{ display: grid; gap: .55rem; margin-bottom: 2.2rem; animation: slide .6s .55s ease-out backwards; }}
.signal {{
  display: flex; justify-content: space-between; align-items: center;
  font-size: .85rem; padding: .35rem 0;
  border-bottom: 1px dashed rgba(29,58,50,.55);
}}
.signal-name {{ color: var(--muted); letter-spacing: .04em; }}
.signal-score {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
.up {{ color: var(--green); }}
.dn {{ color: var(--rose); }}
.stat-line {{
  display: flex; flex-wrap: wrap; gap: 1.4rem;
  margin-bottom: 2.2rem; font-size: .78rem; color: var(--muted);
  animation: slide .6s .65s ease-out backwards;
}}
.stat-line strong {{ color: var(--ink); font-weight: 700; font-variant-numeric: tabular-nums; }}
.size-card {{
  margin-bottom: 2rem; padding: 1.4rem 1.6rem;
  border: 1px solid var(--line); border-radius: 4px;
  background: linear-gradient(135deg, rgba(74,222,128,.06), rgba(251,191,36,.04));
  animation: slide .6s .6s ease-out backwards;
}}
.size-label {{
  font-size: .68rem; letter-spacing: .2em; color: var(--green);
  text-transform: uppercase; margin-bottom: .5rem;
}}
.size-value {{
  font-family: 'Fraunces', serif; font-weight: 700; font-style: italic;
  font-size: clamp(1.8rem, 4vw, 2.4rem); color: var(--amber);
  letter-spacing: -.01em; line-height: 1;
}}
.size-of {{
  font-family: 'JetBrains Mono', monospace; font-style: normal;
  font-weight: 400; font-size: .85rem; color: var(--muted); letter-spacing: 0;
}}
.size-detail {{
  margin-top: .7rem; font-size: .78rem; color: var(--muted); letter-spacing: .02em;
}}
.cta {{
  display: inline-flex; align-items: center; gap: .9rem;
  background: var(--amber); color: #0a1612;
  padding: 1.1rem 1.8rem; text-decoration: none;
  font-weight: 800; letter-spacing: .06em; text-transform: uppercase;
  font-size: .88rem; border-radius: 3px;
  transition: transform .15s ease, box-shadow .25s ease;
  animation: pulseGlow 2.4s ease-in-out infinite, slide .6s .8s ease-out backwards;
}}
@keyframes pulseGlow {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(251,191,36,.45); }}
  50% {{ box-shadow: 0 0 0 18px rgba(251,191,36,0); }}
}}
.cta:hover {{ transform: translateY(-2px); }}
.cta-arrow {{ font-family: 'JetBrains Mono', monospace; font-weight: 800; transition: transform .2s; }}
.cta:hover .cta-arrow {{ transform: translateX(5px); }}
.foot {{
  margin-top: 2.2rem; font-size: .7rem; color: var(--muted); line-height: 1.6;
  animation: slide .6s 1s ease-out backwards;
}}
.foot a {{ color: var(--muted); }}
</style>
</head>
<body>
<main class="card">
  <div class="tag"><span class="dot"></span>sinal de câmbio · {ts}</div>
  <h1>Faz agora,<br><strong>{name}</strong>.</h1>
  <p class="lede">{subtext}</p>

  <div class="rate">
    <span class="rate-num">R$ {rate}</span>
    <span class="rate-lbl">USD / BRL · ao vivo</span>
  </div>

  <div class="signals">{signal_rows}</div>

  <div class="size-card">
    <div class="size-label">Sugestão de tamanho</div>
    <div class="size-value">{convert_pct} <span class="size-of">de {amount} USD</span></div>
    <div class="size-detail">{deadline_text}</div>
  </div>

  <div class="stat-line">
    <span>p(agora) <strong>{p_now}</strong></span>
    <span>composto <strong>{composite}</strong></span>
    <span>concordância <strong>{agreement}</strong></span>
    <span>regime <strong>{regime}</strong></span>
  </div>

  <a class="cta" href="{higlobe_url}" target="_blank" rel="noopener">
    Abrir Higlobe <span class="cta-arrow">→</span>
  </a>

  <p class="foot">
    cambio · análise probabilística baseada em dados públicos · não é recomendação financeira.<br>
    janela de validade: enquanto o sinal não virar. fecha esta aba quando concluir.
  </p>
</main>
</body>
</html>
"""


def _signal_row(name: str, score: float) -> str:
    cls = "up" if score > 0.05 else "dn" if score < -0.05 else ""
    arrow = "▲" if score > 0.05 else "▼" if score < -0.05 else "·"
    return (
        f'<div class="signal">'
        f'<span class="signal-name">{html.escape(name)}</span>'
        f'<span class="signal-score {cls}">{arrow} {score:+.2f}</span>'
        f"</div>"
    )


def render_alert(
    name: str,
    rate_live: float,
    p_now: float,
    composite: float,
    agreement: float,
    regime: float,
    top_signals: list[tuple[str, float]],
    subtext: str,
    convert_pct: float = 1.0,
    deadline_remaining: Optional[int] = None,
    amount_usd: int = 10_000,
    output_path: str = ALERT_PATH,
) -> str:
    """Write the alert HTML to disk and return its absolute path."""
    rows = "\n    ".join(_signal_row(n, s) for n, s in top_signals)
    if deadline_remaining is None:
        deadline_text = "registre a tranche depois do câmbio para iniciar o próximo ciclo"
    elif deadline_remaining <= 0:
        deadline_text = "janela aberta — execute a tranche prevista hoje"
    elif deadline_remaining <= 3:
        deadline_text = f"⚠ próxima tranche em até {deadline_remaining} dia(s)"
    else:
        deadline_text = f"próxima tranche em até {deadline_remaining} dias"

    html_out = _TEMPLATE.format(
        name=html.escape(name),
        ts=datetime.now().strftime("%d/%m/%Y · %H:%M"),
        rate=f"{rate_live:.4f}",
        signal_rows=rows,
        p_now=f"{p_now:.0%}",
        composite=f"{composite:+.2f}",
        agreement=f"{agreement:.0%}",
        regime=f"{regime:+.2f}",
        subtext=html.escape(subtext),
        higlobe_url=HIGLOBE_URL,
        convert_pct=f"{convert_pct:.0%}",
        amount=f"{amount_usd:,}",
        deadline_text=html.escape(deadline_text),
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    return os.path.abspath(output_path)


def open_in_browser(path: str) -> bool:
    try:
        return webbrowser.open(f"file://{path}")
    except Exception:
        return False


def alert(
    name: str,
    rate_live: float,
    p_now: float,
    composite: float,
    agreement: float,
    regime: float,
    top_signals: list[tuple[str, float]],
    subtext: Optional[str] = None,
    convert_pct: float = 1.0,
    deadline_remaining: Optional[int] = None,
    amount_usd: int = 10_000,
    output_path: str = ALERT_PATH,
) -> bool:
    """One-shot: render + open. Returns True on success."""
    sub = subtext or (
        "Vários indicadores apontam que o dólar pode estar em pico local. "
        "Janela curta — converte enquanto a taxa está alta."
    )
    path = render_alert(
        name=name,
        rate_live=rate_live,
        p_now=p_now,
        composite=composite,
        agreement=agreement,
        regime=regime,
        top_signals=top_signals,
        subtext=sub,
        convert_pct=convert_pct,
        deadline_remaining=deadline_remaining,
        amount_usd=amount_usd,
        output_path=output_path,
    )
    return open_in_browser(path)
