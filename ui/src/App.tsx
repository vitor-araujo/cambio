import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API = "/api";
const portfolioMode = import.meta.env.VITE_PORTFOLIO_MODE === "true";
const BRAND_MARK = `${import.meta.env.BASE_URL}cambio-mark.png`;
const REPOSITORY_URL = "https://github.com/vitor-araujo/cambio";

interface Thresholds {
  watch_interval_min: number;
  notify_threshold: number;
  notify_cooldown_hours: number;
  alert_threshold_pct: number;
  alert_cooldown_min: number;
  dca_floor: number;
  dca_ceiling: number;
  cadence_days: number;
  spread_bps: number;
  deadline_days: number;
}

interface JournalEntry {
  ts: string;
  rate_signal: number | null;
  rate_live: number | null;
  decision: "exchange_now" | "wait" | "split";
  size: number;
  p_now: number;
  p_split: number;
  p_wait: number;
  composite: number;
  agreement: number;
  regime: number;
  notified: boolean;
  executed: boolean;
  trigger: string;
  reason: string;
  cadence_days: number;
  days_to_due: number;
  opportunity_score: number;
  opportunity_threshold: number;
}

interface DashboardData {
  state: Record<string, unknown>;
  thresholds: Thresholds;
  last_signal: JournalEntry | null;
  recent_signals: JournalEntry[];
  total_signals: number;
  total_alerts: number;
  recent_alerts: JournalEntry[];
}

interface NotifierStatus {
  provider: string;
  is_configured: boolean;
  missing_keys: string[];
  telegram_bot_token: string;
  telegram_chat_id: string;
  has_token: boolean;
  has_chat_id: boolean;
}

interface MarketData {
  ok: boolean;
  price?: number;
  prev_close?: number;
  change?: number;
  change_pct?: number;
  sparkline?: { d: string; v: number }[];
}

interface Health {
  healthy: boolean;
  uptime_seconds: number;
  last_signal_age_min?: number | null;
}

type Tab = "desk" | "ledger" | "alerts" | "controls" | "telegram" | "cli";

type Toast = {
  message: string;
  tone?: "success" | "error";
  action?: { label: string; run: () => void };
};

const BROKERS = [
  { name: "Higlobe", url: "https://higlobe.com/webapp/en/login" },
  { name: "Husky", url: "https://app.husky.com.br/login" },
  { name: "TechFX", url: "https://techfx.com.br/login" },
];

const TRIGGER_COPY: Record<string, { title: string; detail: string }> = {
  initial_fill: {
    title: "Inicie a sequência",
    detail: "Nenhuma execução foi registrada. Faça a primeira tranche para ancorar o próximo ciclo.",
  },
  cadence_due: {
    title: "Limite de espera atingido",
    detail: "A janela de quatro dias terminou. Execute a tranche prevista, independentemente do sinal.",
  },
  exceptional_signal: {
    title: "Sinal excepcional",
    detail: "A qualidade do preço superou o limiar para antecipar a próxima tranche.",
  },
  opportunity_window: {
    title: "Preço dentro da janela",
    detail: "A janela está aberta e o sinal superou o limiar dinâmico de execução.",
  },
  window_open: {
    title: "Janela aberta",
    detail: "Aguarde apenas um sinal melhor ou o fim do prazo desta tranche.",
  },
  cadence_building: {
    title: "Preserve a opcionalidade",
    detail: "A última tranche ainda é recente. O sistema continua procurando um preço melhor.",
  },
};

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  if (portfolioMode) {
    const { portfolioRequest } = await import("./portfolioDemo");
    return portfolioRequest<T>(path, init);
  }
  const response = await fetch(`${API}${path}`, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Falha na API (${response.status})`);
  }
  return payload as T;
}

function fmtRate(value: number | null | undefined): string {
  return value == null || Number.isNaN(value) ? "—" : `R$ ${value.toFixed(4)}`;
}

function fmtPct(value: number | null | undefined, digits = 0): string {
  return value == null || Number.isNaN(value)
    ? "—"
    : `${(value * 100).toFixed(digits)}%`;
}

function fmtDate(value: string): string {
  const date = new Date(value);
  return date.toLocaleString("pt-BR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function dueCopy(days: number, actionNow = false): string {
  if (actionNow || days <= 0) return "agora";
  const hours = days * 24;
  if (hours < 24) return `em ${Math.max(1, Math.round(hours))}h`;
  return `em ${days.toFixed(1)}d`;
}

function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

function StatusDot({ on }: { on: boolean }) {
  return <span className={`status-dot ${on ? "on" : "off"}`} aria-hidden="true" />;
}

function DecisionBadge({ entry }: { entry: JournalEntry }) {
  const action = entry.decision === "exchange_now";
  return (
    <span className={`decision-badge ${action ? "execute" : "watch"}`}>
      <span>{action ? "Executar" : "Observar"}</span>
      <strong>{fmtPct(entry.size)}</strong>
    </span>
  );
}

function CountdownDial({ entry }: { entry: JournalEntry }) {
  const action = entry.decision === "exchange_now";
  const cadence = entry.cadence_days || 4;
  const progress = action ? 1 : Math.max(0.04, Math.min(1, 1 - entry.days_to_due / cadence));
  const radius = 46;
  const circumference = 2 * Math.PI * radius;

  return (
    <div className={`countdown-dial ${action ? "action" : ""}`}>
      <svg viewBox="0 0 112 112" role="img" aria-label={`Próxima tranche ${dueCopy(entry.days_to_due, action)}`}>
        <circle className="dial-track" cx="56" cy="56" r={radius} />
        <circle
          className="dial-progress"
          cx="56"
          cy="56"
          r={radius}
          strokeDasharray={`${circumference * progress} ${circumference}`}
        />
      </svg>
      <div className="dial-copy">
        <span>{action ? "janela" : "próxima"}</span>
        <strong>{dueCopy(entry.days_to_due, action)}</strong>
      </div>
    </div>
  );
}

function MarketMetric({ kind }: { kind: "usdbrl" | "ibov" }) {
  const [data, setData] = useState<MarketData | null>(null);

  useEffect(() => {
    let active = true;
    const load = () =>
      fetchJson<MarketData>(`/${kind}`)
        .then((value) => active && setData(value))
        .catch(() => undefined);
    load();
    const timer = portfolioMode ? null : window.setInterval(load, kind === "usdbrl" ? 120_000 : 60_000);
    return () => {
      active = false;
      if (timer) window.clearInterval(timer);
    };
  }, [kind]);

  const points = data?.sparkline ?? [];
  const values = points.map((point) => point.v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const polyline = points
    .map((point, index) => {
      const x = points.length > 1 ? (index / (points.length - 1)) * 120 : 60;
      const y = 28 - ((point.v - min) / range) * 24;
      return `${x},${y}`;
    })
    .join(" ");
  const change = data?.change_pct ?? 0;
  const favorable = kind === "usdbrl" ? change >= 0 : change >= 0;

  return (
    <article className="market-metric">
      <div className="metric-topline">
        <span>{kind === "usdbrl" ? `USD / BRL · ${portfolioMode ? "demo" : "PTAX"}` : `IBOV · ${portfolioMode ? "demo" : "spot"}`}</span>
        {data?.ok && (
          <span className={favorable ? "positive" : "negative"}>
            {change >= 0 ? "+" : ""}{change.toFixed(2)}%
          </span>
        )}
      </div>
      <strong className="market-value">
        {!data?.ok
          ? "—"
          : kind === "usdbrl"
            ? fmtRate(data.price)
            : Math.round(data.price ?? 0).toLocaleString("pt-BR")}
      </strong>
      {points.length > 2 && (
        <svg className="metric-spark" viewBox="0 0 120 32" preserveAspectRatio="none" aria-hidden="true">
          <polyline points={polyline} />
        </svg>
      )}
      <small>
        referência anterior {kind === "usdbrl" ? fmtRate(data?.prev_close) : Math.round(data?.prev_close ?? 0).toLocaleString("pt-BR")}
      </small>
    </article>
  );
}

function ExecutionTicket({
  entry,
  onMarkExecuted,
  marking,
}: {
  entry: JournalEntry;
  onMarkExecuted: () => void;
  marking: boolean;
}) {
  const [balance, setBalance] = useState(() => Number(localStorage.getItem("cambio-balance") || 10_000));
  const [broker, setBroker] = useState(() => localStorage.getItem("cambio-broker") || "Higlobe");
  const instruction = entry.executed
    ? {
        ...entry,
        decision: "wait" as const,
        trigger: "cadence_building",
        days_to_due: entry.cadence_days || 4,
      }
    : entry;
  const action = instruction.decision === "exchange_now";
  const rate = entry.rate_live || entry.rate_signal || 0;
  const tranche = Math.max(0, balance) * instruction.size;
  const estimatedBrl = tranche * rate;
  const narrative = TRIGGER_COPY[instruction.trigger] ?? {
    title: action ? "Janela de execução" : "Monitorando o preço",
    detail: entry.reason || "O próximo ciclo atualizará a orientação.",
  };
  const brokerChoice = BROKERS.find((item) => item.name === broker) ?? BROKERS[0];

  const updateBalance = (value: number) => {
    setBalance(value);
    localStorage.setItem("cambio-balance", String(value));
  };

  return (
    <section className={`execution-ticket ${action ? "is-action" : ""}`}>
      <div className="ticket-copy">
        <div className="eyebrow">Instrução de execução</div>
        <div className="ticket-heading">
          <div>
            <span className="ticket-verb">{action ? "Converter" : "Preparar"}</span>
            <strong className="ticket-size">{fmtPct(instruction.size)}</strong>
            <span className="ticket-unit">do saldo disponível</span>
          </div>
          <CountdownDial entry={instruction} />
        </div>

        <div className="execution-rationale">
          <span className="rationale-marker" aria-hidden="true">◆</span>
          <div>
            <strong>{narrative.title}</strong>
            <p>{narrative.detail}</p>
          </div>
        </div>

        <div className="ticket-actions">
          <a className={`primary-action ${!action ? "muted" : ""}`} href={portfolioMode ? REPOSITORY_URL : brokerChoice.url} target="_blank" rel="noreferrer">
            <span>{portfolioMode ? "Explorar implementação" : action ? `Abrir ${brokerChoice.name}` : "Ver cotação na corretora"}</span>
            <span aria-hidden="true">↗</span>
          </a>
          {action && !entry.executed && (
            <button className="secondary-action" onClick={onMarkExecuted} disabled={marking}>
              {marking ? <Spinner /> : <span aria-hidden="true">✓</span>}
              {portfolioMode ? "Simular execução" : "Marcar tranche executada"}
            </button>
          )}
          {entry.executed && <span className="filled-state">✓ Tranche registrada</span>}
        </div>
      </div>

      <aside className="order-ticket" aria-label="Prévia da tranche">
        <div className="order-title">
          <span>{portfolioMode ? "Ordem simulada" : "Ordem indicativa"}</span>
          <span className={`order-state ${action ? "open" : "queued"}`}>{portfolioMode ? "demo" : action ? "aberta" : "na fila"}</span>
        </div>
        <label htmlFor="usd-balance">Saldo em USD</label>
        <div className="money-input">
          <span>$</span>
          <input
            id="usd-balance"
            type="number"
            min="0"
            step="100"
            value={balance}
            onChange={(event) => updateBalance(Number(event.target.value))}
          />
          <span>USD</span>
        </div>
        <dl className="order-breakdown">
          <div><dt>Tranche</dt><dd>US$ {tranche.toLocaleString("pt-BR", { maximumFractionDigits: 0 })}</dd></div>
          <div><dt>Taxa de referência</dt><dd>{fmtRate(rate)}</dd></div>
          <div className="order-total"><dt>Recebimento estimado</dt><dd>R$ {estimatedBrl.toLocaleString("pt-BR", { maximumFractionDigits: 0 })}</dd></div>
          <div><dt>Qualidade da janela</dt><dd>{fmtPct(instruction.opportunity_score)}</dd></div>
          <div><dt>Limiar de execução</dt><dd>{fmtPct(instruction.opportunity_threshold)}</dd></div>
          <div><dt>Visão direcional</dt><dd>{fmtPct(instruction.p_now)}</dd></div>
        </dl>
        <label htmlFor="broker">Corretora</label>
        <select
          id="broker"
          value={broker}
          onChange={(event) => {
            setBroker(event.target.value);
            localStorage.setItem("cambio-broker", event.target.value);
          }}
        >
          {BROKERS.map((item) => <option key={item.name}>{item.name}</option>)}
        </select>
        <p className="order-disclaimer">{portfolioMode ? "Simulação local. Nenhuma ordem é enviada e nenhum valor é movimentado." : "Estimativa antes de spread, IOF e tarifas da corretora."}</p>
      </aside>
    </section>
  );
}

function MarketChart({ entries }: { entries: JournalEntry[] }) {
  const data = useMemo(
    () =>
      [...entries]
        .slice(0, 50)
        .reverse()
        .map((entry) => ({
          ts: entry.ts,
          label: new Date(entry.ts).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" }),
          rate: entry.rate_live || entry.rate_signal || 0,
          probability: +(entry.p_now * 100).toFixed(1),
          quality: +(entry.opportunity_score * 100).toFixed(1),
          threshold: +(entry.opportunity_threshold * 100).toFixed(1),
        })),
    [entries],
  );

  if (!data.length) return <EmptyState title="Aguardando o primeiro sinal" detail="O coletor preencherá a curva assim que concluir o primeiro ciclo." />;

  const rates = data.map((point) => point.rate).filter(Boolean);
  const minRate = Math.min(...rates);
  const maxRate = Math.max(...rates);
  const padding = Math.max(0.012, (maxRate - minRate) * 0.2);

  return (
    <section className="panel chart-panel">
      <header className="panel-heading">
        <div>
          <span className="eyebrow">Microestrutura do sinal</span>
          <h2>Preço versus convicção</h2>
        </div>
        <div className="chart-key">
          <span><i className="key-rate" /> USD/BRL</span>
          <span><i className="key-prob" /> qualidade</span>
          <span><i className="key-hurdle" /> limiar</span>
        </div>
      </header>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={284}>
          <AreaChart data={data} margin={{ top: 16, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="rateFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--chart-rate)" stopOpacity="0.24" />
                <stop offset="100%" stopColor="var(--chart-rate)" stopOpacity="0" />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} stroke="var(--line)" strokeDasharray="2 6" />
            <XAxis dataKey="label" axisLine={false} tickLine={false} minTickGap={32} tick={{ fill: "var(--muted)", fontSize: 11 }} />
            <YAxis yAxisId="rate" domain={[minRate - padding, maxRate + padding]} axisLine={false} tickLine={false} width={62} tickFormatter={(value) => value.toFixed(4)} tick={{ fill: "var(--muted)", fontSize: 11 }} />
            <YAxis yAxisId="prob" orientation="right" domain={[0, 100]} axisLine={false} tickLine={false} width={42} tickFormatter={(value) => `${value}%`} tick={{ fill: "var(--muted)", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "var(--surface-strong)", border: "1px solid var(--line-strong)", borderRadius: 10, boxShadow: "var(--shadow-float)" }}
              labelStyle={{ color: "var(--muted)", marginBottom: 8 }}
              formatter={(value: number, name: string) => [name === "USD/BRL" ? fmtRate(value) : `${value.toFixed(1)}%`, name]}
            />
            <ReferenceLine yAxisId="prob" y={50} stroke="var(--line-strong)" strokeDasharray="3 6" />
            <Area yAxisId="rate" type="monotone" dataKey="rate" name="USD/BRL" stroke="var(--chart-rate)" fill="url(#rateFill)" strokeWidth={2} dot={false} />
            <Line yAxisId="prob" type="monotone" dataKey="probability" name="p(agora)" stroke="var(--faint)" strokeDasharray="2 5" strokeWidth={1} dot={false} />
            <Line yAxisId="prob" type="monotone" dataKey="threshold" name="Limiar" stroke="var(--copper)" strokeDasharray="5 5" strokeWidth={1.5} dot={false} />
            <Line yAxisId="prob" type="monotone" dataKey="quality" name="Qualidade" stroke="var(--signal)" strokeWidth={2.5} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function Blotter({ entries, limit }: { entries: JournalEntry[]; limit?: number }) {
  const rows = limit ? entries.slice(0, limit) : entries;
  if (!rows.length) return <EmptyState title="Ledger vazio" detail="As decisões aparecerão aqui após o primeiro ciclo." />;

  return (
    <div className="table-shell">
      <table>
        <thead><tr><th>Horário</th><th>Estado</th><th>USD/BRL</th><th>Qualidade</th><th>p(agora)</th><th>Limiar</th><th>Tranche</th><th>Gatilho</th><th>Registro</th></tr></thead>
        <tbody>
          {rows.map((entry, index) => (
            <tr key={`${entry.ts}-${index}`} className={index === 0 ? "latest" : ""}>
              <td>{fmtDate(entry.ts)}</td>
              <td><DecisionBadge entry={entry} /></td>
              <td className="numeric strong">{fmtRate(entry.rate_live || entry.rate_signal)}</td>
              <td className="numeric">{fmtPct(entry.opportunity_score)}</td>
              <td className="numeric">{fmtPct(entry.p_now)}</td>
              <td className="numeric">{fmtPct(entry.opportunity_threshold)}</td>
              <td className="numeric">{fmtPct(entry.size)}</td>
              <td><span className="trigger-label">{TRIGGER_COPY[entry.trigger]?.title ?? "modelo legado"}</span></td>
              <td>{entry.executed ? <span className="executed-mark">✓ executada</span> : entry.notified ? "alertada" : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-state"><span aria-hidden="true">◇</span><strong>{title}</strong><p>{detail}</p></div>;
}

function Desk({ data, onMarkExecuted, marking }: { data: DashboardData; onMarkExecuted: () => void; marking: boolean }) {
  const entry = data.last_signal;
  const anchor = Number(data.state.anchor_rate || 0);

  return (
    <div className="view-stack">
      {entry ? <ExecutionTicket entry={entry} onMarkExecuted={onMarkExecuted} marking={marking} /> : <EmptyState title="Montando a mesa" detail="O primeiro ciclo está coletando dados de mercado e calibrando a execução." />}
      <section className="market-strip">
        <MarketMetric kind="usdbrl" />
        <MarketMetric kind="ibov" />
        <article className="market-metric anchor-metric">
          <div className="metric-topline"><span>Âncora intradiária</span><span>referência</span></div>
          <strong className="market-value">{anchor ? fmtRate(anchor) : "—"}</strong>
          <div className="anchor-rule"><span /></div>
          <small>{anchor ? "Base fixa para alertas de alta no dia" : "Será definida na primeira cotação do dia"}</small>
        </article>
      </section>
      <MarketChart entries={data.recent_signals} />
      <section className="panel">
        <header className="panel-heading"><div><span className="eyebrow">Execution blotter</span><h2>Decisões recentes</h2></div><span className="panel-meta">{data.total_signals.toLocaleString("pt-BR")} observações</span></header>
        <Blotter entries={data.recent_signals} limit={8} />
      </section>
    </div>
  );
}

const CONTROL_GROUPS: { title: string; detail: string; fields: { key: keyof Thresholds; label: string; suffix: string; step: number; min: number; max: number; help: string }[] }[] = [
  {
    title: "Mandato de execução",
    detail: "Define a cadência e o tamanho das tranches. O sinal pode antecipar, mas nunca adiar além do limite.",
    fields: [
      { key: "cadence_days", label: "Cadência máxima", suffix: "dias", step: 1, min: 2, max: 30, help: "A tranche é obrigatória quando este prazo termina." },
      { key: "dca_floor", label: "Tranche mínima", suffix: "fração", step: 0.05, min: 0.05, max: 1, help: "Tamanho usado quando o sinal tem baixa convicção." },
      { key: "dca_ceiling", label: "Tranche máxima", suffix: "fração", step: 0.05, min: 0.05, max: 1, help: "Limite mesmo em sinais excepcionais." },
      { key: "spread_bps", label: "Spread estimado", suffix: "bps", step: 5, min: 0, max: 1000, help: "Custo usado nas simulações e estimativas." },
    ],
  },
  {
    title: "Monitoramento",
    detail: "Controla a frequência do feed e o comportamento dos alertas.",
    fields: [
      { key: "watch_interval_min", label: "Atualização do feed", suffix: "min", step: 1, min: 1, max: 60, help: "Frequência de leitura; não é frequência de execução." },
      { key: "notify_threshold", label: "Limiar do navegador", suffix: "fração", step: 0.05, min: 0, max: 1, help: "Convicção mínima para alertas puramente oportunísticos." },
      { key: "notify_cooldown_hours", label: "Silêncio do navegador", suffix: "h", step: 1, min: 0, max: 72, help: "Intervalo mínimo entre alertas repetidos." },
      { key: "alert_threshold_pct", label: "Alta para Telegram", suffix: "%", step: 0.1, min: 0.1, max: 20, help: "Variação contra a âncora diária que dispara o celular." },
      { key: "alert_cooldown_min", label: "Silêncio do Telegram", suffix: "min", step: 1, min: 1, max: 1440, help: "Intervalo mínimo entre mensagens." },
    ],
  },
];

function Controls({ thresholds, onSave, saving }: { thresholds: Thresholds; onSave: (delta: Partial<Thresholds>) => void; saving: boolean }) {
  const [form, setForm] = useState(thresholds);
  useEffect(() => setForm(thresholds), [thresholds]);
  const dirty = JSON.stringify(form) !== JSON.stringify(thresholds);

  return (
    <div className="view-stack controls-view">
      <header className="page-intro"><span className="eyebrow">Policy engine</span><h1>Controles do mandato</h1><p>Ajustes explícitos, limites estreitos e sem parâmetros escondidos.</p></header>
      {CONTROL_GROUPS.map((group) => (
        <section className="panel control-panel" key={group.title}>
          <header className="panel-heading"><div><h2>{group.title}</h2><p>{group.detail}</p></div></header>
          <div className="control-grid">
            {group.fields.map((field) => (
              <label className="control-field" key={field.key}>
                <span>{field.label}</span>
                <div><input type="number" value={form[field.key]} step={field.step} min={field.min} max={field.max} onChange={(event) => setForm((current) => ({ ...current, [field.key]: Number(event.target.value) }))} /><i>{field.suffix}</i></div>
                <small>{field.help}</small>
              </label>
            ))}
          </div>
        </section>
      ))}
      <div className="sticky-save">
        <span>{dirty ? "Alterações ainda não aplicadas" : "Política sincronizada"}</span>
        <button className="secondary-action" disabled={!dirty} onClick={() => setForm(thresholds)}>Descartar</button>
        <button className="primary-action compact" disabled={!dirty || saving} onClick={() => onSave(form)}>{saving ? <Spinner /> : null} Aplicar política</button>
      </div>
    </div>
  );
}

function Telegram({ status, onSave, onTest, busy }: { status: NotifierStatus; onSave: (token: string, chatId: string) => void; onTest: () => void; busy: boolean }) {
  const [token, setToken] = useState("");
  const [chatId, setChatId] = useState("");
  if (portfolioMode) {
    return (
      <div className="view-stack compact-view">
        <header className="page-intro"><span className="eyebrow">Portfolio simulation</span><h1>Telegram</h1><p>Uma prévia segura do canal de alertas, sem credenciais e sem chamadas externas.</p></header>
        <section className="panel telegram-panel">
          <div className="connection-state"><StatusDot on /><div><strong>Canal demonstrativo</strong><span>A integração real usa token mascarado, descoberta de chat e teste explícito.</span></div></div>
          <div className="setup-note"><span>01</span><p>O backend guarda segredos fora do cliente e nunca os inclui no ledger.</p><span>02</span><p>Esta página estática simula apenas o feedback da operação no navegador.</p></div>
          <div className="form-actions"><button className="primary-action compact" disabled={busy} onClick={onTest}>{busy ? <Spinner /> : null} Simular alerta</button><a className="secondary-action" href={REPOSITORY_URL} target="_blank" rel="noreferrer">Ver integração ↗</a></div>
        </section>
      </div>
    );
  }
  return (
    <div className="view-stack compact-view">
      <header className="page-intro"><span className="eyebrow">Alert channel</span><h1>Telegram</h1><p>Receba apenas movimentos relevantes e janelas de execução.</p></header>
      <section className="panel telegram-panel">
        <div className="connection-state"><StatusDot on={status.is_configured} /><div><strong>{status.is_configured ? "Canal operacional" : "Canal desconectado"}</strong><span>{status.is_configured ? "Credenciais verificadas e prontas para teste." : "Crie um bot no @BotFather e conecte abaixo."}</span></div></div>
        <div className="setup-note"><span>01</span><p>Crie um bot com <a href="https://t.me/BotFather" target="_blank" rel="noreferrer">@BotFather</a> e envie uma mensagem para ele.</p><span>02</span><p>Cole o token. O chat ID pode ser descoberto automaticamente no primeiro teste.</p></div>
        <div className="form-grid">
          <label><span>Token do bot</span><input type="password" value={token} onChange={(event) => setToken(event.target.value)} placeholder={status.telegram_bot_token || "123456:AA..."} /></label>
          <label><span>Chat ID</span><input value={chatId} onChange={(event) => setChatId(event.target.value)} placeholder={status.telegram_chat_id || "descoberta automática"} /></label>
        </div>
        <div className="form-actions"><button className="primary-action compact" disabled={!token || busy} onClick={() => onSave(token, chatId)}>{busy ? <Spinner /> : null} Salvar canal</button><button className="secondary-action" disabled={!status.is_configured || busy} onClick={onTest}>Enviar teste</button></div>
      </section>
    </div>
  );
}

const CLI_GROUPS = [
  { title: "Operar", commands: [
    ["Mesa ao vivo", "python fx_timing.py --watch --notify"],
    ["Dashboard", "python server.py --dev"],
    ["Registrar tranche", "python fx_timing.py --mark-executed"],
  ] },
  { title: "Investigar", commands: [
    ["Análise pontual", "python fx_timing.py --lang pt"],
    ["Ledger recente", "python fx_timing.py --show-journal 30"],
    ["Auditoria de 90 dias", "python fx_timing.py --audit 90"],
    ["Backtest", "python fx_timing.py --backtest --days 5 20"],
  ] },
];

function Cli() {
  const [copied, setCopied] = useState<string | null>(null);
  const copy = (command: string) => {
    navigator.clipboard.writeText(command).then(() => {
      setCopied(command);
      window.setTimeout(() => setCopied(null), 1600);
    });
  };
  return (
    <div className="view-stack">
      <header className="page-intro"><span className="eyebrow">Developer surface</span><h1>CLI de execução</h1><p>Baixo ruído em sessões longas; contexto suficiente quando o estado muda.</p></header>
      <section className="terminal-preview" aria-label="Prévia do terminal">
        <div className="terminal-bar"><span /><span /><span /><i>cambio — live execution</i></div>
        <pre><span className="terminal-command">$ python fx_timing.py --watch --notify</span>{"\n\n"}<b>  CAMBIO</b>  /  <em>LIVE EXECUTION</em>{"\n"}<span className="terminal-rule">  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>{"\n"}  feed  1m   ·   tranche cadence  4d   ·   Ctrl+C to stop{"\n\n"}<span className="terminal-time">  11:42:08</span>  <mark> EXECUTE </mark>  <strong>R$ 5.1003</strong>  ·  quality 52% / hurdle 40%  ·  tranche <em>26%</em>  ·  window open now</pre>
      </section>
      <section className="command-layout">
        {CLI_GROUPS.map((group) => <div className="command-group" key={group.title}><span className="eyebrow">{group.title}</span>{group.commands.map(([label, command]) => <button className="command-row" key={command} onClick={() => copy(command)}><span>{label}</span><code>{command}</code><i>{copied === command ? "copiado ✓" : "copiar"}</i></button>)}</div>)}
      </section>
    </div>
  );
}

function LoadingView() {
  return <div className="loading-view"><img className="loading-mark" src={BRAND_MARK} alt="" /><div className="loading-brand">CAMBIO</div><div className="loading-line" /><div className="loading-grid"><span /><span /><span /></div><p>{portfolioMode ? "Preparando o case interativo…" : "Conectando ao feed de execução…"}</p></div>;
}

function PortfolioNotice() {
  if (!portfolioMode) return null;
  return (
    <aside className="portfolio-notice" aria-label="Contexto da demonstração">
      <span>Portfolio demo</span>
      <p><strong>Case interativo, dados sintéticos.</strong> Explore a mesa inteira; alterações ficam somente neste navegador.</p>
      <a href={REPOSITORY_URL} target="_blank" rel="noreferrer">Ver código <span aria-hidden="true">↗</span></a>
    </aside>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("desk");
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [journal, setJournal] = useState<JournalEntry[]>([]);
  const [thresholds, setThresholds] = useState<Thresholds | null>(null);
  const [notifier, setNotifier] = useState<NotifierStatus | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);

  const showToast = useCallback((value: Toast) => {
    setToast(value);
    window.setTimeout(() => setToast(null), 5000);
  }, []);

  const refresh = useCallback(async (quiet = false) => {
    try {
      const [nextDashboard, nextJournal, nextThresholds, nextNotifier, nextHealth] = await Promise.all([
        fetchJson<DashboardData>("/dashboard"),
        fetchJson<JournalEntry[]>("/journal?limit=1000"),
        fetchJson<Thresholds>("/thresholds"),
        fetchJson<NotifierStatus>("/notifier"),
        fetchJson<Health>("/health"),
      ]);
      setDashboard(nextDashboard);
      setJournal(nextJournal);
      setThresholds(nextThresholds);
      setNotifier(nextNotifier);
      setHealth(nextHealth);
      setError(null);
    } catch (cause) {
      if (!quiet) setError(cause instanceof Error ? cause.message : "Não foi possível conectar ao servidor.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    if (portfolioMode) return undefined;
    const timer = window.setInterval(() => refresh(true), 30_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const markExecuted = async () => {
    setBusy("execution");
    try {
      await fetchJson("/executions", { method: "POST" });
      await refresh(true);
      showToast({
        message: portfolioMode ? "Execução simulada. O relógio foi reiniciado apenas neste navegador." : "Tranche registrada. O relógio de quatro dias foi reiniciado.",
        tone: "success",
        action: {
          label: "Desfazer",
          run: async () => {
            await fetchJson("/executions/undo", { method: "POST" });
            await refresh(true);
            showToast({ message: portfolioMode ? "Simulação reiniciada." : "Registro de execução desfeito." });
          },
        },
      });
    } catch (cause) {
      showToast({ message: cause instanceof Error ? cause.message : "Falha ao registrar a tranche.", tone: "error" });
    } finally {
      setBusy(null);
    }
  };

  const saveThresholds = async (next: Partial<Thresholds>) => {
    setBusy("controls");
    try {
      const result = await fetchJson<{ ok: boolean; thresholds: Thresholds }>("/thresholds", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(next) });
      setThresholds(result.thresholds);
      await refresh(true);
      showToast({ message: portfolioMode ? "Política simulada aplicada nesta sessão." : "Política aplicada ao próximo ciclo.", tone: "success" });
    } catch (cause) {
      showToast({ message: cause instanceof Error ? cause.message : "Falha ao aplicar a política.", tone: "error" });
    } finally {
      setBusy(null);
    }
  };

  const saveTelegram = async (token: string, chatId: string) => {
    setBusy("telegram");
    try {
      await fetchJson("/notifier", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: "telegram", telegram_bot_token: token, telegram_chat_id: chatId || "auto" }) });
      await refresh(true);
      showToast({ message: "Canal do Telegram salvo.", tone: "success" });
    } catch (cause) {
      showToast({ message: cause instanceof Error ? cause.message : "Falha ao salvar o Telegram.", tone: "error" });
    } finally {
      setBusy(null);
    }
  };

  const testTelegram = async () => {
    setBusy("telegram");
    try {
      const result = await fetchJson<{ message?: string }>("/notifier/test", { method: "POST" });
      showToast({ message: result.message || (portfolioMode ? "Alerta simulado." : "Mensagem de teste enviada."), tone: "success" });
      await refresh(true);
    } catch (cause) {
      showToast({ message: cause instanceof Error ? cause.message : "Falha no teste.", tone: "error" });
    } finally {
      setBusy(null);
    }
  };

  if (loading) return <LoadingView />;

  const navigation: { id: Tab; label: string; glyph: string }[] = [
    { id: "desk", label: "Mesa", glyph: "⌁" },
    { id: "ledger", label: "Ledger", glyph: "≡" },
    { id: "alerts", label: "Alertas", glyph: "◉" },
    { id: "controls", label: "Controles", glyph: "⌘" },
    { id: "telegram", label: "Telegram", glyph: "↗" },
    { id: "cli", label: "CLI", glyph: ">_" },
  ];
  const latest = dashboard?.last_signal;
  const activeTitle = navigation.find((item) => item.id === tab)?.label ?? "Mesa";

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="brand"><img className="brand-mark" src={BRAND_MARK} alt="" /><div><strong>cambio</strong><span>{portfolioMode ? "portfolio case" : "execution desk"}</span></div></div>
        <nav aria-label="Navegação principal">
          {navigation.map((item) => <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)}><i>{item.glyph}</i><span>{item.label}</span>{item.id === "alerts" && dashboard?.total_alerts ? <b>{dashboard.total_alerts}</b> : null}</button>)}
        </nav>
        <div className="rail-status">
          <div><StatusDot on={Boolean(health?.healthy)} /><span>{portfolioMode ? "Demo local" : health?.healthy ? "Feed operacional" : "Feed indisponível"}</span></div>
          <small>{portfolioMode ? "sem backend · sem ordens" : health ? `uptime ${Math.floor(health.uptime_seconds / 3600)}h ${Math.floor((health.uptime_seconds % 3600) / 60)}m` : "sem telemetria"}</small>
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-bar">
          <div><span className="eyebrow">USD / BRL · {portfolioMode ? "engineering case study" : "mandato pessoal"}</span><h1>{activeTitle}</h1></div>
          <div className="workspace-live">
            {latest && <><span>{portfolioMode ? "snapshot" : "último tick"}</span><strong>{fmtRate(latest.rate_live || latest.rate_signal)}</strong><DecisionBadge entry={latest} /></>}
            <div className="live-feed"><StatusDot on={Boolean(health?.healthy)} /><span>{portfolioMode ? "simulação" : "live"}</span></div>
          </div>
        </header>

        {error && <div className="error-banner"><div><strong>O feed não respondeu</strong><span>{error}</span></div><button onClick={() => refresh()}>Tentar novamente</button></div>}

        <div className="workspace-content">
          <PortfolioNotice />
          {tab === "desk" && dashboard && <Desk data={dashboard} onMarkExecuted={markExecuted} marking={busy === "execution"} />}
          {tab === "ledger" && <div className="view-stack"><header className="page-intro"><span className="eyebrow">Audit trail</span><h1>Ledger de decisões</h1><p>Cada observação, gatilho e execução em uma trilha verificável.</p></header><Blotter entries={journal} /></div>}
          {tab === "alerts" && <div className="view-stack"><header className="page-intro"><span className="eyebrow">Exception queue</span><h1>Alertas</h1><p>Somente eventos que pediram atenção ou abriram uma janela.</p></header><Blotter entries={journal.filter((entry) => entry.notified || entry.decision === "exchange_now")} /></div>}
          {tab === "controls" && thresholds && <Controls thresholds={thresholds} onSave={saveThresholds} saving={busy === "controls"} />}
          {tab === "telegram" && notifier && <Telegram status={notifier} onSave={saveTelegram} onTest={testTelegram} busy={busy === "telegram"} />}
          {tab === "cli" && <Cli />}
        </div>
      </main>

      {toast && <div className={`toast ${toast.tone ?? ""}`} role="status"><span>{toast.tone === "success" ? "✓" : toast.tone === "error" ? "!" : "·"}</span><p>{toast.message}</p>{toast.action && <button onClick={() => { toast.action?.run(); setToast(null); }}>{toast.action.label}</button>}<button className="toast-close" onClick={() => setToast(null)} aria-label="Fechar">×</button></div>}
    </div>
  );
}
