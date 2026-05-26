import { useState, useEffect, useCallback } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  ReferenceArea,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const API = "/api";

// ── types ────────────────────────────────────────────────────────────────────

interface Thresholds {
  watch_interval_min: number;
  notify_threshold: number;
  notify_cooldown_hours: number;
  alert_threshold_pct: number;
  alert_cooldown_min: number;
  dca_floor: number;
  dca_ceiling: number;
  spread_bps: number;
  deadline_days: number;
}

interface JournalEntry {
  ts: string;
  rate_signal: number | null;
  rate_live: number | null;
  decision: string;
  size: number;
  p_now: number;
  p_split: number;
  p_wait: number;
  composite: number;
  agreement: number;
  regime: number;
  notified: boolean;
  executed: boolean;
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

interface DashboardData {
  state: Record<string, unknown>;
  thresholds: Thresholds;
  last_signal: JournalEntry | null;
  recent_signals: JournalEntry[];
  total_signals: number;
  total_alerts: number;
  recent_alerts: JournalEntry[];
}

type Tab = "dashboard" | "history" | "alerts" | "phone" | "thresholds" | "cli";

// ── helpers ───────────────────────────────────────────────────────────────────

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

function fmtRate(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `R$ ${v.toFixed(4)}`;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtTs(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleString("pt-BR", {
      day: "2-digit",
      month: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

function useCopy() {
  const [copied, setCopied] = useState<string | null>(null);
  const copy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(id);
      setTimeout(() => setCopied(null), 1500);
    });
  }, []);
  return { copied, copy };
}

// ── CLI command reference ────────────────────────────────────────────────────

const CLI_SECTIONS = [
  {
    title: "Iniciar",
    desc: "Comandos para iniciar o monitoramento.",
    cmds: [
      {
        label: "Monitoramento",
        code: "python fx_timing.py --watch --notify --phone-alerts",
      },
      { label: "Dashboard", code: "python server.py --dev" },
      { label: "Análise única", code: "python fx_timing.py" },
      { label: "PT-BR", code: "python fx_timing.py --lang pt --notify" },
    ],
  },
  {
    title: "Configuração",
    desc: "Ajuste thresholds e configure alertas.",
    cmds: [
      { label: "Setup Telegram", code: "python configure.py" },
      {
        label: "Intervalo custom",
        code: "python fx_timing.py --watch --watch-interval 10",
      },
      {
        label: "Limiar de alerta",
        code: "python fx_timing.py --watch --alert-threshold 2.0",
      },
      {
        label: "Cooldown alerta",
        code: "python fx_timing.py --watch --alert-cooldown 30",
      },
    ],
  },
  {
    title: "Diário & Auditoria",
    desc: "Consulte o histórico e auditoria de sinais.",
    cmds: [
      { label: "Ver sinais", code: "python fx_timing.py --show-journal" },
      { label: "Últimos 50", code: "python fx_timing.py --show-journal 50" },
      { label: "Auditoria 30d", code: "python fx_timing.py --audit" },
      { label: "Auditoria 90d", code: "python fx_timing.py --audit 90" },
    ],
  },
  {
    title: "Backtest & Marcações",
    desc: "Backtest para validar estratégia e marcar execuções.",
    cmds: [
      { label: "Backtest padrão", code: "python fx_timing.py --backtest" },
      {
        label: "Dias custom",
        code: "python fx_timing.py --backtest --days 5 20",
      },
      {
        label: "Prazo forçado",
        code: "python fx_timing.py --backtest --deadline-days 15",
      },
      {
        label: "Marcar executado",
        code: "python fx_timing.py --mark-executed",
      },
    ],
  },
];

const CLI_FLAGS = [
  { flag: "--watch", desc: "Loop contínuo de checagem" },
  { flag: "--notify", desc: "Abre alerta HTML no navegador" },
  { flag: "--phone-alerts", desc: "Envia alertas via Telegram" },
  { flag: "--watch-interval N", desc: "Minutos entre checks (padrão: 5)" },
  {
    flag: "--alert-threshold N",
    desc: "% de alta vs âncora para alerta (padrão: 1.0)",
  },
  { flag: "--alert-cooldown N", desc: "Minutos entre alertas Telegram" },
  {
    flag: "--dca-floor FRAC",
    desc: "Fração mínima por conversão (padrão: 0.25)",
  },
  {
    flag: "--dca-ceiling FRAC",
    desc: "Fração máxima por conversão (padrão: 0.75)",
  },
  { flag: "--deadline-days N", desc: "Prazo máximo em dias (padrão: 15)" },
  { flag: "--spread-bps N", desc: "Spread em basis points (padrão: 50)" },
  { flag: "--lang pt|en", desc: "Idioma de saída" },
  { flag: "--mark-executed", desc: "Marca último sinal como executado" },
  { flag: "--show-journal [N]", desc: "Mostra últimos N sinais" },
  { flag: "--audit [N]", desc: "Auditoria dos últimos N dias" },
];

// ── components ────────────────────────────────────────────────────────────────

function Spinner() {
  return <span className="spinner" />;
}

const BROKERS = [
  { name: "Higlobe", url: "https://app.higlobe.com/login" },
  { name: "Husky", url: "https://app.husky.com.br/login" },
  { name: "TechFX", url: "https://techfx.com.br/login" },
];
const PREF_KEY = "cambio-broker";

function BrokerButton({ pNow }: { pNow: number | null }) {
  const [open, setOpen] = useState(false);
  const [pref, setPref] = useState<string>(
    () => localStorage.getItem(PREF_KEY) || "Higlobe",
  );
  const hot = pNow != null && pNow >= 0.5;
  const warm = pNow != null && pNow >= 0.4;
  const choice = BROKERS.find((b) => b.name === pref) || BROKERS[0];

  const pick = (name: string) => {
    setPref(name);
    localStorage.setItem(PREF_KEY, name);
    setOpen(false);
  };

  if (pNow == null) return null;

  return (
    <div className="broker-wrap">
      <button
        className={`broker-trigger ${hot ? "broker-hot" : warm ? "broker-warm" : "broker-cold"}`}
        onClick={() => {
          if (hot) window.open(choice.url, "_blank");
          else setOpen((v) => !v);
        }}
        title={
          hot
            ? `p(agora) ≥ 50% — abrir ${choice.name}`
            : warm
              ? "p(agora) subindo…"
              : "p(agora) < 40%"
        }
      >
        <span className="broker-icon">↗</span>
        <span className="broker-label">
          {hot ? choice.name : warm ? "Quase…" : "Aguardar"}
        </span>
        {hot && <span className="broker-pulse" />}
      </button>
      {open && (
        <div className="broker-menu">
          <div className="broker-menu-title">Corretora padrão</div>
          {BROKERS.map((b) => (
            <button
              key={b.name}
              className={`broker-link ${b.name === pref ? "broker-active" : ""}`}
              onClick={() => pick(b.name)}
            >
              {b.name}
              {b.name === pref && <span className="broker-check">✓</span>}
            </button>
          ))}
          <a
            href={choice.url}
            target="_blank"
            rel="noopener"
            className="broker-open"
          >
            Abrir {choice.name}
          </a>
        </div>
      )}
    </div>
  );
}

// ── live pulse indicator ────────────────────────────────────────────────────

function LivePulse({
  intervalSec,
  lastSignalTs,
}: {
  intervalSec: number;
  lastSignalTs: string | null;
}) {
  // Anchor = timestamp of the last recorded signal.
  // Progress = time elapsed since that signal / watch interval.
  // This means refreshing the page keeps the same ring position —
  // it's tied to the actual data, not the frontend fetch.
  const anchorMs = lastSignalTs ? new Date(lastSignalTs).getTime() : 0;
  const intervalMs = intervalSec * 1000;
  const [progress, setProgress] = useState(() =>
    anchorMs ? Math.min((Date.now() - anchorMs) / intervalMs, 1) : 0,
  );

  useEffect(() => {
    if (!anchorMs) return;
    let rafId: number;
    const tick = () => {
      setProgress(Math.min((Date.now() - anchorMs) / intervalMs, 1));
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [anchorMs, intervalMs]);

  if (!anchorMs) return null;

  const remaining = Math.max(0, intervalSec * (1 - progress));
  const min = Math.floor(remaining / 60);
  const sec = Math.floor(remaining % 60);
  const timeStr =
    min > 0 ? `${min}:${String(sec).padStart(2, "0")}` : `${sec}s`;

  const r = 18;
  const circ = 2 * Math.PI * r;
  const dash = circ * progress;

  return (
    <div className="live-pulse">
      <div className="live-ring">
        <svg viewBox="0 0 44 44" width="44" height="44">
          <circle
            cx="22"
            cy="22"
            r={r}
            fill="none"
            stroke="var(--border)"
            strokeWidth="2.5"
          />
          <circle
            cx="22"
            cy="22"
            r={r}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="2.5"
            strokeDasharray={`${dash} ${circ - dash}`}
            strokeLinecap="round"
            transform="rotate(-90 22 22)"
          />
        </svg>
        <div className="live-dot" />
      </div>
      <div className="live-info">
        <span className="live-label">próxima coleta em</span>
        <span className="live-countdown">{timeStr}</span>
      </div>
    </div>
  );
}

function DecisionBadge({
  decision,
  size,
}: {
  decision: string;
  size?: number;
}) {
  const cls =
    decision === "exchange_now"
      ? "badge-now"
      : decision === "wait"
        ? "badge-wait"
        : "badge-split";
  const label =
    decision === "exchange_now"
      ? "NOW"
      : decision === "wait"
        ? "WAIT"
        : "SPLIT";
  const s = size !== undefined ? ` ${(size * 100).toFixed(0)}%` : "";
  return (
    <span className={`badge ${cls}`}>
      {label}
      {s}
    </span>
  );
}

function StatusDot({ on }: { on: boolean }) {
  return <span className={`status-dot ${on ? "on" : "off"}`} />;
}

function ConfirmDialog({
  title,
  message,
  onConfirm,
  onCancel,
  loading,
}: {
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
  loading?: boolean;
}) {
  return (
    <div className="overlay" onClick={onCancel}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        <p>{message}</p>
        <div className="flex-row gap-sm">
          <button
            className="btn btn-primary"
            onClick={onConfirm}
            disabled={loading}
          >
            {loading && <Spinner />}
            {loading ? "Salvando…" : "Confirmar"}
          </button>
          <button
            className="btn btn-ghost"
            onClick={onCancel}
            disabled={loading}
          >
            Cancelar
          </button>
        </div>
      </div>
    </div>
  );
}

// ── dashboard ────────────────────────────────────────────────────────────────

function Dashboard({ data }: { data: DashboardData }) {
  const { state, thresholds, recent_signals, total_signals, total_alerts } =
    data;
  const anchor = (state as Record<string, number | undefined>).anchor_rate;
  const lastAlertTs = (state as Record<string, string | undefined>)
    .last_alert_ts;
  const lastRate =
    recent_signals[0]?.rate_live || recent_signals[0]?.rate_signal;

  // Hero: latest rate big display
  const lastDecision = recent_signals[0];
  const watchIntervalSec = (thresholds.watch_interval_min || 5) * 60;

  // Chart data: last 30 entries.
  const chartRaw = [...recent_signals]
    .slice(0, 30)
    .reverse()
    .map((e) => {
      const d = new Date(e.ts);
      const hh = String(d.getHours()).padStart(2, "0");
      const mm = String(d.getMinutes()).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      const mon = String(d.getMonth() + 1).padStart(2, "0");
      return {
        ts: e.ts,
        label: `${hh}:${mm}`,
        fulllabel: `${day}/${mon} ${hh}:${mm}`,
        prevDay: "",
        p_now: +(e.p_now * 100).toFixed(1),
        rate: e.rate_live || e.rate_signal || 0,
      };
    });

  // Annotate day changes.
  for (let i = 0; i < chartRaw.length; i++) {
    const d = new Date(chartRaw[i].ts);
    const dayKey = `${d.getDate()}/${d.getMonth() + 1}`;
    if (i === 0 || dayKey !== chartRaw[i - 1].prevDay) {
      chartRaw[i].label = chartRaw[i].fulllabel;
    }
    chartRaw[i].prevDay = dayKey;
  }

  const chartData = chartRaw.map((d) => ({ ...d, tick: d.label }));

  // Center p(agora) Y-axis.
  const pVals = chartData.map((d) => d.p_now);
  const pMin = Math.min(...pVals);
  const pMax = Math.max(...pVals);
  const pSpread = pMax - pMin || 20;
  const pLower = Math.max(0, pMin - pSpread * 0.6);
  const pUpper = Math.min(100, pMax + pSpread * 0.6);

  return (
    <>
      {/* Broker button */}
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          marginBottom: "0.5rem",
        }}
      >
        <BrokerButton pNow={lastDecision?.p_now ?? null} />
      </div>

      {/* Hero rate */}
      {lastRate != null && (
        <div className="hero-rate">
          <div className="rate-value">R$ {lastRate.toFixed(4)}</div>
          <div className="rate-label">USD/BRL</div>
          {lastDecision && (
            <div className="hero-meta">
              <span>
                <DecisionBadge
                  decision={lastDecision.decision}
                  size={lastDecision.size}
                />
              </span>
              <span>
                p(agora){" "}
                <strong className="up">{fmtPct(lastDecision.p_now)}</strong>
              </span>
              <span>
                regime{" "}
                <strong className={lastDecision.regime >= 0 ? "up" : "down"}>
                  {lastDecision.regime >= 0 ? "+" : ""}
                  {lastDecision.regime.toFixed(2)}
                </strong>
              </span>
            </div>
          )}
        </div>
      )}

      {/* metrics */}
      <div className="metric-grid">
        <div className="metric">
          <div className="metric-label">Sinais</div>
          <div className="metric-value">{total_signals}</div>
        </div>
        <div className="metric">
          <div className="metric-label">Alertas</div>
          <div
            className="metric-value"
            style={{ color: total_alerts > 0 ? "var(--amber)" : undefined }}
          >
            {total_alerts}
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">Âncora</div>
          <div
            className="metric-value"
            style={{ color: anchor ? "var(--accent-bright)" : undefined }}
          >
            {anchor ? `R$ ${anchor.toFixed(2)}` : "—"}
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">Último alerta</div>
          <div className="metric-value">
            {lastAlertTs ? fmtTs(lastAlertTs) : "—"}
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">DCA floor</div>
          <div className="metric-value">
            {(thresholds.dca_floor * 100).toFixed(0)}%
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">DCA ceiling</div>
          <div className="metric-value">
            {(thresholds.dca_ceiling * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* chart */}
      {chartData.length > 0 && (
        <div className="section">
          <h2>Evolução</h2>
          <div className="card" style={{ paddingBottom: "0.25rem" }}>
            <div className="chart-legend">
              <div className="chart-legend-item">
                <span
                  className="chart-legend-swatch"
                  style={{ background: "#818cf8" }}
                />{" "}
                USD/BRL
              </div>
              <div className="chart-legend-item">
                <span style={{ display: "inline-flex", gap: 1 }}>
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: "#f87171",
                      display: "inline-block",
                    }}
                  />
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: "#fbbf24",
                      display: "inline-block",
                    }}
                  />
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: "#34d399",
                      display: "inline-block",
                    }}
                  />
                </span>
                p(agora) &lt;40 / 40–50 / ≥50%
              </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart
                data={chartData}
                margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="tick"
                  tick={{ fill: "var(--text-dim)", fontSize: 9 }}
                  tickLine={false}
                />
                <YAxis
                  yAxisId="rate"
                  orientation="left"
                  tick={{ fill: "#818cf8", fontSize: 9 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  domain={["auto", "auto"]}
                  tickFormatter={(v: number) => `R$${v.toFixed(2)}`}
                />
                <YAxis
                  yAxisId="pct"
                  orientation="right"
                  tick={{ fill: "#34d399", fontSize: 9 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  domain={[pLower, pUpper]}
                  tickFormatter={(v: number) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "0.75rem",
                    fontFamily: "var(--mono)",
                  }}
                  labelFormatter={(label: string) => label || "—"}
                />
                <ReferenceArea
                  yAxisId="pct"
                  y1={0}
                  y2={40}
                  fill="#f87171"
                  fillOpacity={0.08}
                />
                <ReferenceArea
                  yAxisId="pct"
                  y1={40}
                  y2={50}
                  fill="#fbbf24"
                  fillOpacity={0.08}
                />
                <ReferenceArea
                  yAxisId="pct"
                  y1={50}
                  y2={100}
                  fill="#34d399"
                  fillOpacity={0.08}
                />
                <Bar
                  yAxisId="rate"
                  dataKey="rate"
                  fill="#818cf8"
                  fillOpacity={0.85}
                  radius={[2, 2, 0, 0]}
                  barSize={9}
                  name="USD/BRL"
                />
                <Line
                  yAxisId="pct"
                  type="monotone"
                  dataKey="p_now"
                  stroke="#8b8a87"
                  strokeWidth={2.5}
                  dot={(props: any) => {
                    const { cx, cy, payload } = props;
                    const v = payload.p_now;
                    const fill =
                      v >= 50 ? "#34d399" : v >= 40 ? "#fbbf24" : "#f87171";
                    return (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={4}
                        fill={fill}
                        stroke="var(--bg-card)"
                        strokeWidth={2}
                      />
                    );
                  }}
                  name="p(agora)"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* live pulse */}
      <LivePulse
        intervalSec={watchIntervalSec}
        lastSignalTs={recent_signals[0]?.ts ?? null}
      />

      {/* last 10 signals */}
      <div className="section">
        <h2>Últimos sinais</h2>
        {recent_signals.length === 0 ? (
          <div className="empty">
            <div className="empty-icon">—</div>
            Nenhum sinal ainda. Rode <code>python fx_timing.py --watch</code>
          </div>
        ) : (
          <div className="signal-list">
            {recent_signals.slice(0, 10).map((e, i) => {
              const isPrimary = i === 0;
              const rate = e.rate_live || e.rate_signal || 0;
              return (
                <div
                  key={i}
                  className={`signal-row ${isPrimary ? "primary" : ""}`}
                >
                  <span className={`signal-ts ${isPrimary ? "signal-ts" : ""}`}>
                    {fmtTs(e.ts)}
                  </span>
                  <span className="signal-decision">
                    <DecisionBadge decision={e.decision} size={e.size} />
                  </span>
                  <span
                    className={`signal-rate ${isPrimary ? "signal-rate" : ""}`}
                  >
                    {fmtRate(rate)}
                  </span>
                  <span className="signal-details">
                    <span>
                      p(agora){" "}
                      <strong
                        className={
                          e.decision === "exchange_now"
                            ? "up"
                            : e.p_now >= 0.4
                              ? "up"
                              : ""
                        }
                      >
                        {fmtPct(e.p_now)}
                      </strong>
                    </span>
                    <span>
                      split{" "}
                      <strong className={e.decision === "split" ? "up" : ""}>
                        {fmtPct(e.p_split)}
                      </strong>
                    </span>
                    <span>
                      wait{" "}
                      <strong className={e.decision === "wait" ? "down" : ""}>
                        {fmtPct(e.p_wait)}
                      </strong>
                    </span>
                    <span>
                      comp{" "}
                      <strong>
                        {e.composite >= 0 ? "+" : ""}
                        {e.composite.toFixed(2)}
                      </strong>
                    </span>
                    <span>
                      regime{" "}
                      <strong className={e.regime >= 0 ? "up" : "down"}>
                        {e.regime >= 0 ? "+" : ""}
                        {e.regime.toFixed(2)}
                      </strong>
                    </span>
                  </span>
                  <span className="signal-tags">
                    {e.notified && (
                      <span className="badge badge-icon" title="alertado">
                        📱
                      </span>
                    )}
                    {e.executed && (
                      <span className="badge badge-icon" title="executado">
                        ✓
                      </span>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* thresholds summary */}
      <div className="section">
        <h2>Thresholds</h2>
        <div className="card">
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
              gap: "0.65rem",
              fontSize: "0.78rem",
            }}
          >
            <div>
              watch <strong>{thresholds.watch_interval_min}m</strong>
            </div>
            <div>
              notify ≥ <strong>{fmtPct(thresholds.notify_threshold)}</strong>
            </div>
            <div>
              cooldown <strong>{thresholds.notify_cooldown_hours}h</strong>
            </div>
            <div>
              alerta +
              <strong>{thresholds.alert_threshold_pct.toFixed(2)}%</strong>
            </div>
            <div>
              spread <strong>{thresholds.spread_bps}bps</strong>
            </div>
            <div>
              deadline <strong>{thresholds.deadline_days}d</strong>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ── history ──────────────────────────────────────────────────────────────────

function HistoryTable({ entries }: { entries: JournalEntry[] }) {
  if (!entries.length) {
    return (
      <div className="empty">
        <div className="empty-icon">📋</div>
        Nenhum sinal ainda.
        <br />
        Rode <code>python fx_timing.py --watch</code>
      </div>
    );
  }
  const reversed = [...entries].reverse();
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>data</th>
            <th>decisão</th>
            <th>taxa</th>
            <th>p(agora)</th>
            <th>p(split)</th>
            <th>p(wait)</th>
            <th>comp</th>
            <th>regime</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {reversed.map((e, i) => {
            const rate = e.rate_live || e.rate_signal;
            return (
              <tr key={i}>
                <td>{fmtTs(e.ts)}</td>
                <td>
                  <DecisionBadge decision={e.decision} />
                </td>
                <td>{fmtRate(rate)}</td>
                <td
                  className={
                    e.decision === "exchange_now"
                      ? "up"
                      : e.p_now >= 0.4
                        ? "up"
                        : ""
                  }
                >
                  {fmtPct(e.p_now)}
                </td>
                <td className={e.decision === "split" ? "up" : ""}>
                  {fmtPct(e.p_split)}
                </td>
                <td className={e.decision === "wait" ? "down" : ""}>
                  {fmtPct(e.p_wait)}
                </td>
                <td className={e.composite >= 0 ? "up" : "down"}>
                  {e.composite >= 0 ? "+" : ""}
                  {e.composite.toFixed(2)}
                </td>
                <td className={e.regime >= 0 ? "up" : "down"}>
                  {e.regime >= 0 ? "+" : ""}
                  {e.regime.toFixed(2)}
                </td>
                <td>
                  {e.notified && <span title="alertado">📱</span>}
                  {e.executed && (
                    <span title="executado" style={{ marginLeft: 4 }}>
                      ✓
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AlertsTable({ entries }: { entries: JournalEntry[] }) {
  const notified = entries.filter((e) => e.notified);
  if (!notified.length) {
    return (
      <div className="empty">
        <div className="empty-icon">🔔</div>Nenhum alerta ainda.
      </div>
    );
  }
  return <HistoryTable entries={notified} />;
}

// ── thresholds panel ─────────────────────────────────────────────────────────

type ThresholdGroup = {
  title: string;
  desc: string;
  fields: {
    key: keyof Thresholds;
    label: string;
    hint: string;
    step?: string;
    suffix?: string;
  }[];
};

function ThresholdsPanel({
  thresholds,
  onSave,
}: {
  thresholds: Thresholds;
  onSave: (t: Partial<Thresholds>) => void;
}) {
  const [form, setForm] = useState<Thresholds>(thresholds);
  const [dirty, setDirty] = useState(false);
  const [confirm, setConfirm] = useState(false);

  useEffect(() => {
    setForm(thresholds);
    setDirty(false);
  }, [thresholds]);

  const set =
    (key: keyof Thresholds) => (e: React.ChangeEvent<HTMLInputElement>) => {
      setForm((p) => ({ ...p, [key]: parseFloat(e.target.value) || 0 }));
      setDirty(true);
    };

  const groups: ThresholdGroup[] = [
    {
      title: "Monitoramento",
      desc: "Controlam a frequência de checagem e quando abrir o alerta HTML no navegador.",
      fields: [
        {
          key: "watch_interval_min",
          label: "Intervalo de watch",
          hint: "A cada quantos minutos o sistema consulta a cotação USD/BRL em modo --watch.",
          suffix: "min",
        },
        {
          key: "notify_threshold",
          label: "p(agora) mínimo",
          hint: "Probabilidade mínima para abrir o alerta HTML. 0.40 = só alerta quando o modelo tem ≥ 40% de convicção em converter agora.",
          step: "0.05",
        },
        {
          key: "notify_cooldown_hours",
          label: "Cooldown de alerta",
          hint: "Horas de silêncio entre um alerta HTML e o próximo. Evita spam em alta volatilidade.",
          suffix: "h",
        },
      ],
    },
    {
      title: "Alertas no celular",
      desc: "Disparam via Telegram quando o USD/BRL sobe além da âncora.",
      fields: [
        {
          key: "alert_threshold_pct",
          label: "Limiar de alta",
          hint: "Percentual de alta vs. a âncora que dispara o alerta. Ex: 1.0 = alerta quando o dólar sobe 1% desde a última notificação.",
          step: "0.1",
          suffix: "%",
        },
        {
          key: "alert_cooldown_min",
          label: "Cooldown de celular",
          hint: "Minutos de silêncio entre alertas no Telegram. Alinhe com o watch_interval para não perder oportunidades.",
          suffix: "min",
        },
      ],
    },
    {
      title: "Disciplina de conversão",
      desc: "Inspirado no Vanguard DCA — controlam quanto converter por vez e o custo do spread.",
      fields: [
        {
          key: "dca_floor",
          label: "Fração mínima (floor)",
          hint: "Mesmo com baixa convicção, sempre converte ao menos esta fração do saldo. DCA puro usa 0.25.",
          step: "0.05",
        },
        {
          key: "dca_ceiling",
          label: "Fração máxima (ceiling)",
          hint: "Quando a convicção é alta, converte no máximo esta fração. Evita all-in em sinais falsos.",
          step: "0.05",
        },
        {
          key: "spread_bps",
          label: "Spread cambial",
          hint: "Custo efetivo do câmbio em basis points. Usado no backtest para calcular se o modelo bate o custo. 50 bps ≈ 0.50% (fintech).",
          suffix: "bps",
        },
        {
          key: "deadline_days",
          label: "Prazo máximo",
          hint: "Dias até execução forçada. Após marcar um câmbio executado, o relógio reseta. Força disciplina: converte mesmo sem sinal se o prazo estourar.",
          suffix: "d",
        },
      ],
    },
  ];

  return (
    <>
      <div className="section">
        <h2>Thresholds</h2>
        <p
          style={{
            color: "var(--text-dim)",
            fontSize: "0.82rem",
            marginBottom: "1.5rem",
            lineHeight: 1.65,
            maxWidth: "64ch",
          }}
        >
          Cada threshold controla um aspecto do comportamento do monitor. Passe
          o mouse sobre o{" "}
          <span
            style={{
              color: "var(--text-dim)",
              borderBottom: "1px dashed var(--text-faint)",
            }}
          >
            label
          </span>{" "}
          para ver a explicação.
        </p>

        {groups.map((g) => (
          <div key={g.title} className="mb" style={{ marginBottom: "1.8rem" }}>
            <h2 style={{ fontSize: "1rem", marginBottom: "0.15rem" }}>
              {g.title}
            </h2>
            <p
              style={{
                color: "var(--text-dim)",
                fontSize: "0.75rem",
                marginBottom: "0.75rem",
                lineHeight: 1.5,
              }}
            >
              {g.desc}
            </p>
            <div className="card">
              <div className="field-grid">
                {g.fields.map(({ key, label, hint, step, suffix }) => (
                  <div
                    className="field-group"
                    key={key}
                    style={{ gap: "0.15rem" }}
                  >
                    <label
                      title={hint}
                      style={{
                        cursor: "help",
                        borderBottom: "1px dashed transparent",
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.borderBottomColor =
                          "var(--text-faint)")
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.borderBottomColor =
                          "transparent")
                      }
                    >
                      {label}
                    </label>
                    <div
                      style={{
                        display: "flex",
                        gap: "0.35rem",
                        alignItems: "center",
                      }}
                    >
                      <input
                        type="number"
                        step={step ?? "any"}
                        value={form[key]}
                        onChange={set(key)}
                        style={{ flex: 1 }}
                      />
                      {suffix && (
                        <span
                          style={{
                            color: "var(--text-dim)",
                            fontSize: "0.72rem",
                            flexShrink: 0,
                          }}
                        >
                          {suffix}
                        </span>
                      )}
                    </div>
                    <span
                      style={{
                        fontSize: "0.65rem",
                        color: "var(--text-faint)",
                        lineHeight: 1.4,
                        marginTop: "0.15rem",
                      }}
                    >
                      {hint}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}

        <div className="flex-row gap-sm" style={{ marginTop: "0.5rem" }}>
          <button
            className="btn btn-primary"
            disabled={!dirty}
            onClick={() => setConfirm(true)}
          >
            Salvar thresholds
          </button>
          <button
            className="btn btn-ghost"
            disabled={!dirty}
            onClick={() => {
              setForm(thresholds);
              setDirty(false);
            }}
          >
            Reset
          </button>
        </div>
      </div>
      {confirm && (
        <ConfirmDialog
          title="Salvar thresholds?"
          message="Alterar thresholds afeta o comportamento do --watch, alertas e decisões de tamanho."
          onConfirm={() => {
            const delta: Partial<Thresholds> = {};
            for (const key of Object.keys(thresholds) as (keyof Thresholds)[]) {
              if (form[key] !== thresholds[key]) delta[key] = form[key];
            }
            if (Object.keys(delta).length) onSave(delta);
            setConfirm(false);
          }}
          onCancel={() => setConfirm(false)}
        />
      )}
    </>
  );
}

// ── phone alerts config ──────────────────────────────────────────────────────

function PhoneConfig({
  notifier,
  onSave,
  onTest,
  saving,
  testing,
}: {
  notifier: NotifierStatus;
  onSave: (cfg: {
    provider: string;
    telegram_bot_token: string;
    telegram_chat_id: string;
  }) => void;
  onTest: () => void;
  saving: boolean;
  testing: boolean;
}) {
  const [token, setToken] = useState("");
  const [chatId, setChatId] = useState("");
  const [confirmSave, setConfirmSave] = useState(false);

  const configured = notifier.is_configured;

  return (
    <>
      <div className="section">
        <h2>Telegram</h2>
        <div className="card">
          <div className="config-status">
            <StatusDot on={configured} />
            <strong>{configured ? "Configurado ✓" : "Não configurado"}</strong>
            {!configured && notifier.missing_keys.length > 0 && (
              <span style={{ color: "var(--text-dim)", fontSize: "0.78rem" }}>
                faltam: {notifier.missing_keys.join(", ")}
              </span>
            )}
          </div>

          <div className="config-steps mb">
            <strong>1.</strong> Crie um bot com{" "}
            <a href="https://t.me/BotFather" target="_blank" rel="noopener">
              @BotFather
            </a>{" "}
            no Telegram, cole o token abaixo.
            <br />
            <strong>2.</strong> Mande qualquer mensagem pro bot. O chat_id será
            descoberto automaticamente no teste, ou você pode colar manualmente.
          </div>

          <div className="field-grid mb">
            <div className="field-group">
              <label>Bot Token</label>
              <input
                type="text"
                placeholder={notifier.telegram_bot_token || "1234567890:AAH..."}
                value={token}
                onChange={(e) => setToken(e.target.value)}
              />
            </div>
            <div className="field-group">
              <label>
                Chat ID{" "}
                {notifier.has_chat_id && (
                  <span style={{ color: "var(--green)", fontSize: "0.65rem" }}>
                    (configurado)
                  </span>
                )}
              </label>
              <input
                type="text"
                placeholder={
                  notifier.telegram_chat_id ||
                  "Deixe vazio — descoberto no teste"
                }
                value={chatId}
                onChange={(e) => setChatId(e.target.value)}
              />
            </div>
          </div>

          <div className="flex-row gap-sm">
            <button
              className="btn btn-primary"
              disabled={!token}
              onClick={() => setConfirmSave(true)}
            >
              {saving && <Spinner />}
              Salvar config
            </button>
            <button
              className="btn btn-success"
              disabled={!configured && !token}
              onClick={onTest}
            >
              {testing && <Spinner />}
              Testar envio
            </button>
          </div>
        </div>
      </div>

      {confirmSave && (
        <ConfirmDialog
          title="Salvar Telegram?"
          message={
            configured
              ? "Isso vai sobrescrever a configuração atual do Telegram."
              : `Configurar alertas via ${token ? "bot" : "Telegram"}?`
          }
          onConfirm={() => {
            onSave({
              provider: "telegram",
              telegram_bot_token: token,
              telegram_chat_id: chatId || "auto",
            });
            setConfirmSave(false);
          }}
          onCancel={() => setConfirmSave(false)}
        />
      )}
    </>
  );
}

// ── CLI reference tab ─────────────────────────────────────────────────────────

function CliPanel() {
  const { copied, copy } = useCopy();

  return (
    <div>
      <div className="section">
        <h2>Referência CLI</h2>
        <p
          style={{
            color: "var(--text-dim)",
            fontSize: "0.82rem",
            lineHeight: 1.65,
            maxWidth: "64ch",
            marginBottom: "1.5rem",
          }}
        >
          Todos os comandos para rodar o monitor. Clique em{" "}
          <strong>copiar</strong> para copiar o comando.
        </p>

        {CLI_SECTIONS.map((section) => (
          <div key={section.title} className="cli-section">
            <h2>{section.title}</h2>
            <p className="cli-desc">{section.desc}</p>
            <div className="cmd-grid">
              {section.cmds.map((cmd) => {
                const id = `${section.title}-${cmd.label}`;
                return (
                  <div className="cmd-row" key={id}>
                    <span className="cmd-label">{cmd.label}</span>
                    <span className="cmd-code">{cmd.code}</span>
                    <button
                      className={`cmd-copy ${copied === id ? "copied" : ""}`}
                      onClick={() => copy(cmd.code, id)}
                    >
                      {copied === id ? "✓ copiado" : "copiar"}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        <div className="cli-section">
          <h2>Flags</h2>
          <p className="cli-desc">
            Parâmetros disponíveis para todos os comandos.
          </p>
          <dl className="cli-flags">
            {CLI_FLAGS.map((f) => (
              <>
                <dt>{f.flag}</dt>
                <dd>{f.desc}</dd>
              </>
            ))}
          </dl>
        </div>

        <div className="cli-section">
          <h2>API</h2>
          <p className="cli-desc">
            O servidor em <code>server.py</code> expõe endpoints REST para
            integração externa.
          </p>
          <div className="cmd-grid">
            {[
              {
                label: "Dashboard",
                code: "curl http://127.0.0.1:8765/api/dashboard",
              },
              {
                label: "Journal",
                code: "curl http://127.0.0.1:8765/api/journal",
              },
              { label: "State", code: "curl http://127.0.0.1:8765/api/state" },
              {
                label: "Thresholds",
                code: "curl http://127.0.0.1:8765/api/thresholds",
              },
              {
                label: "Health",
                code: "curl http://127.0.0.1:8765/api/health",
              },
            ].map((cmd) => {
              const id = `api-${cmd.label}`;
              return (
                <div className="cmd-row" key={id}>
                  <span className="cmd-label">{cmd.label}</span>
                  <span className="cmd-code">{cmd.code}</span>
                  <button
                    className={`cmd-copy ${copied === id ? "copied" : ""}`}
                    onClick={() => copy(cmd.code, id)}
                  >
                    {copied === id ? "✓ copiado" : "copiar"}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── app ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [journal, setJournal] = useState<JournalEntry[]>([]);
  const [thresholds, setThresholds] = useState<Thresholds | null>(null);
  const [notifier, setNotifier] = useState<NotifierStatus | null>(null);
  const [toast, setToast] = useState<{ msg: string; error?: boolean } | null>(
    null,
  );
  const [health, setHealth] = useState<{
    healthy: boolean;
    uptime_seconds: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingNotifier, setSavingNotifier] = useState(false);
  const [testingNotifier, setTestingNotifier] = useState(false);

  const showToast = useCallback((msg: string, error?: boolean) => {
    setToast({ msg, error });
    setTimeout(() => setToast(null), 3000);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [dash, j, t, n, h] = await Promise.all([
        fetchJson<DashboardData>("/dashboard"),
        fetchJson<JournalEntry[]>("/journal"),
        fetchJson<Thresholds>("/thresholds"),
        fetchJson<NotifierStatus>("/notifier"),
        fetchJson<any>("/health"),
      ]);
      setDashboard(dash);
      setJournal(j);
      setThresholds(t);
      setNotifier(n);
      setHealth(h);
    } catch {
      // silent refresh failure
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  const saveThresholds = async (delta: Partial<Thresholds>) => {
    try {
      const res = await fetch(`${API}/thresholds`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(delta),
      });
      const data = await res.json();
      if (data.ok) {
        setThresholds(data.thresholds);
        showToast("Thresholds salvos ✓");
      }
    } catch {
      showToast("Erro ao salvar", true);
    }
  };

  const saveNotifier = async (cfg: {
    provider: string;
    telegram_bot_token: string;
    telegram_chat_id: string;
  }) => {
    setSavingNotifier(true);
    try {
      const res = await fetch(`${API}/notifier`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      const data = await res.json();
      if (data.ok) {
        setNotifier(data.status);
        showToast("Configuração salva ✓");
      } else {
        showToast(data.error || "Erro", true);
      }
    } catch {
      showToast("Erro ao salvar", true);
    } finally {
      setSavingNotifier(false);
    }
  };

  const testNotifier = async () => {
    setTestingNotifier(true);
    try {
      const res = await fetch(`${API}/notifier/test`, { method: "POST" });
      const data = await res.json();
      if (data.ok) {
        showToast(data.message || "Teste enviado ✓");
        await refresh();
      } else {
        showToast(data.error || "Falha no teste", true);
      }
    } catch {
      showToast("Erro no teste", true);
    } finally {
      setTestingNotifier(false);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <Spinner />
        carregando…
      </div>
    );
  }

  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  };

  return (
    <div>
      <header className="header">
        <div>
          <h1>
            cambio <em>monitor</em>
          </h1>
          <div className="header-tag">USD/BRL</div>
        </div>
        <div className="header-meta">
          {health && (
            <div className="status" style={{ marginBottom: 4 }}>
              <StatusDot on={health.healthy} />
              {health.healthy ? "api ok" : "api down"}
              {" · "}
              {formatUptime(health.uptime_seconds)}
            </div>
          )}
          {journal.length} sinais · {journal.filter((e) => e.notified).length}{" "}
          alertas
          {notifier && (
            <div style={{ marginTop: 4 }}>
              <StatusDot on={notifier.is_configured} />{" "}
              {notifier.is_configured ? "telegram on" : "telegram off"}
            </div>
          )}
        </div>
      </header>

      <nav className="tabs">
        {(
          [
            ["dashboard", "Dashboard"],
            ["history", "Histórico"],
            ["alerts", "Alertas"],
            ["phone", "Telegram"],
            ["thresholds", "Thresholds"],
            ["cli", "CLI"],
          ] as [Tab, string][]
        ).map(([t, label]) => (
          <button
            key={t}
            className={`tab ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === "dashboard" && dashboard && <Dashboard data={dashboard} />}
      {tab === "history" && <HistoryTable entries={journal} />}
      {tab === "alerts" && <AlertsTable entries={journal} />}
      {tab === "phone" && notifier && (
        <PhoneConfig
          notifier={notifier}
          onSave={saveNotifier}
          onTest={testNotifier}
          saving={savingNotifier}
          testing={testingNotifier}
        />
      )}
      {tab === "thresholds" && thresholds && (
        <ThresholdsPanel thresholds={thresholds} onSave={saveThresholds} />
      )}
      {tab === "cli" && <CliPanel />}

      {toast && (
        <div className={`toast ${toast.error ? "error" : ""}`}>{toast.msg}</div>
      )}
    </div>
  );
}
