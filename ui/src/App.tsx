import React, { useState, useEffect, useCallback, useRef } from "react";
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

// UI-visible thresholds (subset of Thresholds)
type VisibleThresholds = Pick<
  Thresholds,
  | "watch_interval_min"
  | "notify_threshold"
  | "notify_cooldown_hours"
  | "alert_threshold_pct"
  | "alert_cooldown_min"
>;

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

interface IbovPoint {
  d: string;
  v: number;
}

interface IbovData {
  ok: boolean;
  price?: number;
  prev_close?: number;
  change?: number;
  change_pct?: number;
  sparkline?: IbovPoint[];
  error?: string;
}

type Tab = "dashboard" | "history" | "alerts" | "phone" | "thresholds" | "cli";

// ── helpers ───────────────────────────────────────────────────────────────────

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

function fmtRate(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "\u2014";
  return `R$ ${v.toFixed(4)}`;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "\u2014";
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
      { label: "Analise unica", code: "python fx_timing.py" },
      { label: "PT-BR", code: "python fx_timing.py --lang pt --notify" },
    ],
  },
  {
    title: "Configuracao",
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
    title: "Diario & Auditoria",
    desc: "Consulte o historico e auditoria de sinais.",
    cmds: [
      { label: "Ver sinais", code: "python fx_timing.py --show-journal" },
      { label: "Ultimos 50", code: "python fx_timing.py --show-journal 50" },
      { label: "Auditoria 30d", code: "python fx_timing.py --audit" },
      { label: "Auditoria 90d", code: "python fx_timing.py --audit 90" },
    ],
  },
  {
    title: "Backtest & Marcacoes",
    desc: "Backtest para validar estrategia e marcar execucoes.",
    cmds: [
      { label: "Backtest padrao", code: "python fx_timing.py --backtest" },
      {
        label: "Dias custom",
        code: "python fx_timing.py --backtest --days 5 20",
      },
      {
        label: "Prazo forcado",
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
  { flag: "--watch", desc: "Loop continuo de checagem" },
  { flag: "--notify", desc: "Abre alerta HTML no navegador" },
  { flag: "--phone-alerts", desc: "Envia alertas via Telegram" },
  { flag: "--watch-interval N", desc: "Minutos entre checks (padrao: 5)" },
  {
    flag: "--alert-threshold N",
    desc: "% de alta vs ancora para alerta (padrao: 1.0)",
  },
  { flag: "--alert-cooldown N", desc: "Minutos entre alertas Telegram" },
  {
    flag: "--dca-floor FRAC",
    desc: "Fracao minima por conversao (padrao: 0.25)",
  },
  {
    flag: "--dca-ceiling FRAC",
    desc: "Fracao maxima por conversao (padrao: 0.75)",
  },
  { flag: "--deadline-days N", desc: "Prazo maximo em dias (padrao: 15)" },
  { flag: "--spread-bps N", desc: "Spread em basis points (padrao: 50)" },
  { flag: "--lang pt|en", desc: "Idioma de saida" },
  { flag: "--mark-executed", desc: "Marca ultimo sinal como executado" },
  { flag: "--show-journal [N]", desc: "Mostra ultimos N sinais" },
  { flag: "--audit [N]", desc: "Auditoria dos ultimos N dias" },
];

// ── components ────────────────────────────────────────────────────────────────

function Spinner() {
  return <span className="spinner" />;
}

const BROKERS = [
  { name: "Higlobe", url: "https://higlobe.com/webapp/en/login" },
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
            ? `p(agora) >= 50% - abrir ${choice.name}`
            : warm
              ? "p(agora) subindo"
              : "p(agora) < 40%"
        }
      >
        <span className="broker-icon">&#8599;</span>
        <span className="broker-label">
          {hot ? choice.name : warm ? "Quase" : "Aguardar"}
        </span>
        {hot && <span className="broker-pulse" />}
      </button>
      {open && (
        <div className="broker-menu">
          <div className="broker-menu-title">Corretora padrao</div>
          {BROKERS.map((b) => (
            <button
              key={b.name}
              className={`broker-link ${b.name === pref ? "broker-active" : ""}`}
              onClick={() => pick(b.name)}
            >
              {b.name}
              {b.name === pref && (
                <span className="broker-check">&#10003;</span>
              )}
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
  const anchorMs = lastSignalTs ? new Date(lastSignalTs).getTime() : 0;
  const intervalMs = intervalSec * 1000;
  const [progress, setProgress] = useState(() =>
    anchorMs ? ((Date.now() - anchorMs) / intervalMs) % 1 : 0,
  );

  useEffect(() => {
    if (!anchorMs) return;
    let rafId: number;
    const tick = () => {
      setProgress(((Date.now() - anchorMs) / intervalMs) % 1);
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
            stroke="var(--green)"
            strokeWidth="2.5"
            strokeDasharray={`${dash} ${circ - dash}`}
            strokeLinecap="round"
            transform="rotate(-90 22 22)"
          />
        </svg>
        <div className="live-dot" />
      </div>
      <div className="live-info">
        <span className="live-label">proxima coleta em</span>
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
            {loading ? "Salvando..." : "Confirmar"}
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

// ── ibov ──────────────────────────────────────────────────────────────────────

function IbovMetric() {
  const [ibov, setIbov] = useState<IbovData | null>(null);

  const load = useCallback(async () => {
    try {
      setIbov(await fetchJson<IbovData>("/ibov"));
    } catch {
      /* silent */
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000);
    return () => clearInterval(id);
  }, [load]);

  if (!ibov || !ibov.ok) {
    return (
      <div className="metric">
        <div className="metric-label">IBOV</div>
        <div className="metric-value">&mdash;</div>
      </div>
    );
  }

  const isUp = (ibov.change ?? 0) >= 0;
  const arrow = isUp ? "\u2191" : "\u2193";
  const pct = ibov.change_pct ?? 0;
  const spark = ibov.sparkline ?? [];

  const fmtNum = (n: number) =>
    n.toLocaleString("pt-BR", { maximumFractionDigits: 0 });

  // mini sparkline SVG
  const W = 120,
    H = 32;
  const vals = spark.map((p) => p.v);
  const mn = Math.min(...vals);
  const mx = Math.max(...vals);
  const range = mx - mn || 1;
  const pts = spark.map((p, i) => {
    const x = spark.length > 1 ? (i / (spark.length - 1)) * W : W / 2;
    const y = 2 + (1 - (p.v - mn) / range) * (H - 4);
    return `${x},${y}`;
  });
  const line = pts.join(" ");
  const fill =
    spark.length > 1
      ? `M${pts[0]} L${pts.slice(1).join(" L")} L${W},${H} L0,${H} Z`
      : "";

  const color = isUp ? "var(--green)" : "var(--red)";

  return (
    <div className="metric metric-ibov">
      <div className="metric-label">
        IBOV{" "}
        <span className={`ibov-badge ${isUp ? "ibov-up" : "ibov-down"}`}>
          {arrow} {pct >= 0 ? "+" : ""}
          {pct.toFixed(2)}%
        </span>
      </div>
      <div className="metric-value">{fmtNum(ibov.price ?? 0)}</div>
      {spark.length > 2 && (
        <svg
          className="ibov-spark"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          width="100%"
          height={H}
        >
          <defs>
            <linearGradient id="ig" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.3" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={fill} fill="url(#ig)" />
          <polyline
            points={line}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </svg>
      )}
      <div className="ibov-prev">prev {fmtNum(ibov.prev_close ?? 0)}</div>
    </div>
  );
}

// ── dashboard ────────────────────────────────────────────────────────────────

function UsdbrlMetric() {
  const [usdbrl, setUsdbrl] = useState<IbovData | null>(null);

  const load = useCallback(async () => {
    try {
      setUsdbrl(await fetchJson<IbovData>("/usdbrl"));
    } catch {
      /* silent */
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 120_000);
    return () => clearInterval(id);
  }, [load]);

  if (!usdbrl || !usdbrl.ok) {
    return (
      <div className="metric">
        <div className="metric-label">USD/BRL</div>
        <div className="metric-value">&mdash;</div>
      </div>
    );
  }

  const isUp = (usdbrl.change ?? 0) < 0;
  const arrow = isUp ? "↑" : "↓";
  const pct = usdbrl.change_pct ?? 0;
  const spark = usdbrl.sparkline ?? [];

  const W = 120,
    H = 32;
  const vals = spark.map((p) => p.v);
  const mn = Math.min(...vals);
  const mx = Math.max(...vals);
  const range = mx - mn || 0.01;
  const pts = spark.map((p, i) => {
    const x = spark.length > 1 ? (i / (spark.length - 1)) * W : W / 2;
    const y = 2 + (1 - (p.v - mn) / range) * (H - 4);
    return `${x},${y}`;
  });
  const line = pts.join(" ");
  const fill =
    spark.length > 1
      ? `M${pts[0]} L${pts.slice(1).join(" L")} L${W},${H} L0,${H} Z`
      : "";

  // USD/BRL colour logic is inverted: falling rate = BRL stronger = GOOD
  const color = isUp ? "var(--green)" : "var(--red)";
  const fmt = (n: number) => n.toFixed(4).replace(/\.?0+$/, "");

  return (
    <div className="metric metric-ibov">
      <div className="metric-label">
        Dólar{" "}
        <span className={`ibov-badge ${isUp ? "ibov-up" : "ibov-down"}`}>
          {arrow} {pct >= 0 ? "+" : ""}
          {pct.toFixed(2)}%
        </span>
      </div>
      <div className="metric-value">R$ {fmt(usdbrl.price ?? 0)}</div>
      {spark.length > 2 && (
        <svg
          className="ibov-spark"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          width="100%"
          height={H}
        >
          <defs>
            <linearGradient id="ug" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.3" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={fill} fill="url(#ug)" />
          <polyline
            points={line}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </svg>
      )}
      <div className="ibov-prev">
        PTAX anterior R$ {fmt(usdbrl.prev_close ?? 0)}
      </div>
    </div>
  );
}

function Dashboard({ data }: { data: DashboardData }) {
  const { state, thresholds, recent_signals, total_alerts } = data;
  const anchor = (state as Record<string, number | undefined>).anchor_rate;
  const lastRate =
    recent_signals[0]?.rate_live || recent_signals[0]?.rate_signal;

  const lastDecision = recent_signals[0];
  const watchIntervalSec = (thresholds.watch_interval_min || 5) * 60;
  const [chartRange, setChartRange] = useState<number>(30);
  const chartRef = useRef<HTMLDivElement>(null);

  // Chart data: last N entries.
  const chartRaw = [...recent_signals]
    .slice(0, chartRange)
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

  // ── rate Y-axis: auto domain with ≥ 3-centavo spread ──────────────────
  const rateVals = chartData.map((d) => d.rate);
  const rateMin = Math.min(...rateVals);
  const rateMax = Math.max(...rateVals);
  const rateSpread = rateMax - rateMin;
  const MIN_SPREAD = 0.03;
  const pad = Math.max(0, (MIN_SPREAD - rateSpread) / 2);
  const defaultRateDomain: [number, number] = [
    +(rateMin - pad - 0.003).toFixed(4),
    +(rateMax + pad + 0.003).toFixed(4),
  ];

  // ── zoom state (null = auto) ──────────────────────────────────────────
  const [rateDomain, setRateDomain] = useState<[number, number] | null>(null);
  const [pctDomain, setPctDomain] = useState<[number, number] | null>(null);

  // Reset zoom when the user changes the visible range
  useEffect(() => {
    setRateDomain(null);
    setPctDomain(null);
  }, [chartRange]);

  const effectiveRateDomain = rateDomain ?? defaultRateDomain;
  const effectivePctDomain = pctDomain ?? [0, 100];

  // Refs for the wheel handler — avoid stale closures
  const rateDomainRef = useRef(effectiveRateDomain);
  rateDomainRef.current = effectiveRateDomain;
  const pctDomainRef = useRef(effectivePctDomain);
  pctDomainRef.current = effectivePctDomain;
  const rateRangeRef = useRef({ min: rateMin, max: rateMax });
  rateRangeRef.current = { min: rateMin, max: rateMax };

  // ── wheel → zoom the y-axis under the cursor ──────────────────────────
  useEffect(() => {
    const el = chartRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.12 : 1 / 1.12;
      const rect = el.getBoundingClientRect();
      const isRight = e.clientX - rect.left > rect.width * 0.55;

      if (isRight) {
        const [lo, hi] = pctDomainRef.current;
        const center = (lo + hi) / 2;
        const half = ((hi - lo) / 2) * factor;
        setPctDomain([
          Math.max(-5, center - half),
          Math.min(105, center + half),
        ]);
      } else {
        const [lo, hi] = rateDomainRef.current;
        const center = (lo + hi) / 2;
        const half = ((hi - lo) / 2) * factor;
        const { min, max } = rateRangeRef.current;
        setRateDomain([
          Math.max(min - 0.1, center - half),
          Math.min(max + 0.1, center + half),
        ]);
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const resetZoom = useCallback(() => {
    setRateDomain(null);
    setPctDomain(null);
  }, []);
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
            </div>
          )}
        </div>
      )}

      {/* metrics - only what matters */}
      <div className="metric-grid">
        <IbovMetric />
        <UsdbrlMetric />
        <div className="metric">
          <div className="metric-label">ancora</div>
          <div
            className="metric-value"
            style={{ color: anchor ? "var(--green)" : undefined }}
          >
            {anchor ? `R$ ${anchor.toFixed(2)}` : "\u2014"}
          </div>
        </div>
      </div>

      {/* chart */}
      {chartData.length > 0 && (
        <div className="section">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "1rem",
            }}
          >
            <h2 style={{ marginBottom: 0 }}>Evolucao</h2>
            <div className="chart-range">
              {[10, 20, 30, 50].map((n) => (
                <button
                  key={n}
                  className={`chart-range-btn ${chartRange === n ? "active" : ""}`}
                  onClick={() => setChartRange(n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
          <div
            ref={chartRef}
            className="card"
            style={{ paddingBottom: "0.25rem" }}
            onDoubleClick={resetZoom}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div className="chart-legend">
                <div className="chart-legend-item">
                  <span
                    className="chart-legend-swatch"
                    style={{ background: "var(--text-dim)" }}
                  />
                  USD/BRL
                </div>
                <div className="chart-legend-item">
                  <span style={{ display: "inline-flex", gap: 3 }}>
                    <span
                      className="chart-legend-dot"
                      style={{ background: "#f87171" }}
                    />
                    <span
                      className="chart-legend-dot"
                      style={{ background: "#fbbf24" }}
                    />
                    <span
                      className="chart-legend-dot"
                      style={{ background: "var(--blue)" }}
                    />
                  </span>
                  p(agora)
                </div>
                <span
                  style={{
                    fontSize: "0.65rem",
                    color: "var(--text-faint)",
                    opacity: rateDomain || pctDomain ? 0.6 : 0,
                    transition: "opacity 0.2s",
                    cursor: "pointer",
                  }}
                  onClick={resetZoom}
                  title="Double-click chart to reset zoom"
                >
                  scroll=zoom · dbl-click=reset
                </span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart
                data={chartData}
                margin={{ top: 4, right: 8, left: -10, bottom: 0 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="var(--border)"
                  vertical={false}
                />
                <XAxis
                  dataKey="tick"
                  tick={{ fill: "var(--text-faint)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  yAxisId="rate"
                  orientation="left"
                  tick={{ fill: "var(--text-dim)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  domain={effectiveRateDomain}
                  tickFormatter={(v: number) => v.toFixed(4)}
                  interval="preserveStartEnd"
                />
                <YAxis
                  yAxisId="pct"
                  orientation="right"
                  tick={{ fill: "var(--text-dim)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  domain={effectivePctDomain as [number, number]}
                  allowDataOverflow
                  tickFormatter={(v: number) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "0.75rem",
                    fontFamily: "var(--font-geist-mono), monospace",
                  }}
                  labelFormatter={(label: string) => label || "\u2014"}
                  formatter={(value: number, name: string) => [
                    name === "rate"
                      ? `R$ ${value.toFixed(4)}`
                      : `${value.toFixed(1)}%`,
                    name === "rate" ? "USD/BRL" : "p(agora)",
                  ]}
                />
                <ReferenceArea
                  yAxisId="pct"
                  y1={0}
                  y2={40}
                  fill="#f87171"
                  fillOpacity={0.04}
                />
                <ReferenceArea
                  yAxisId="pct"
                  y1={40}
                  y2={50}
                  fill="#fbbf24"
                  fillOpacity={0.04}
                />
                <ReferenceArea
                  yAxisId="pct"
                  y1={50}
                  y2={100}
                  fill="var(--blue)"
                  fillOpacity={0.04}
                />
                <Bar
                  yAxisId="rate"
                  dataKey="rate"
                  fill="var(--text-faint)"
                  fillOpacity={0.4}
                  radius={[2, 2, 0, 0]}
                  barSize={8}
                  name="rate"
                />
                <Line
                  yAxisId="pct"
                  type="natural"
                  dataKey="p_now"
                  stroke="var(--text-dim)"
                  strokeWidth={2}
                  dot={(props: any) => {
                    const { cx, cy, payload } = props;
                    const v = payload.p_now;
                    const fill =
                      v >= 50 ? "var(--blue)" : v >= 40 ? "#fbbf24" : "#f87171";
                    return (
                      <circle
                        key={`dot-${payload.ts}`}
                        cx={cx}
                        cy={cy}
                        r={4}
                        fill={fill}
                        stroke="var(--bg-card)"
                        strokeWidth={2}
                      />
                    );
                  }}
                  activeDot={{
                    r: 6,
                    fill: "var(--blue)",
                    stroke: "var(--bg-card)",
                    strokeWidth: 2,
                  }}
                  name="p_now"
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
        <h2>Ultimos sinais</h2>
        {recent_signals.length === 0 ? (
          <div className="empty">
            <div className="empty-icon">&#8212;</div>
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
                  <span className="signal-ts">{fmtTs(e.ts)}</span>
                  <span className="signal-decision">
                    <DecisionBadge decision={e.decision} size={e.size} />
                  </span>
                  <span className="signal-rate">{fmtRate(rate)}</span>
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
                  </span>
                  <span className="signal-tags">
                    {e.notified && (
                      <span className="badge badge-icon" title="alertado">
                        &#x1F4F1;
                      </span>
                    )}
                    {e.executed && (
                      <span className="badge badge-icon" title="executado">
                        &#10003;
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
              notify <strong>{fmtPct(thresholds.notify_threshold)}</strong>
            </div>
            <div>
              cooldown <strong>{thresholds.notify_cooldown_hours}h</strong>
            </div>
            <div>
              alerta{" "}
              <strong>{thresholds.alert_threshold_pct.toFixed(2)}%</strong>
            </div>
            <div>
              alert cooldown <strong>{thresholds.alert_cooldown_min}min</strong>
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
        <div className="empty-icon">&#x1F4CB;</div>
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
            <th>decisao</th>
            <th>taxa</th>
            <th>p(agora)</th>
            <th>tamanho</th>
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
                <td>{(e.size * 100).toFixed(0)}%</td>
                <td>
                  {e.notified && <span title="alertado">&#x1F4F1;</span>}
                  {e.executed && (
                    <span title="executado" style={{ marginLeft: 4 }}>
                      &#10003;
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
        <div className="empty-icon">&#x1F514;</div>Nenhum alerta ainda.
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
    key: keyof VisibleThresholds;
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
  const [form, setForm] = useState<VisibleThresholds>({
    watch_interval_min: thresholds.watch_interval_min,
    notify_threshold: thresholds.notify_threshold,
    notify_cooldown_hours: thresholds.notify_cooldown_hours,
    alert_threshold_pct: thresholds.alert_threshold_pct,
    alert_cooldown_min: thresholds.alert_cooldown_min,
  });
  const [dirty, setDirty] = useState(false);
  const [confirm, setConfirm] = useState(false);

  useEffect(() => {
    setForm({
      watch_interval_min: thresholds.watch_interval_min,
      notify_threshold: thresholds.notify_threshold,
      notify_cooldown_hours: thresholds.notify_cooldown_hours,
      alert_threshold_pct: thresholds.alert_threshold_pct,
      alert_cooldown_min: thresholds.alert_cooldown_min,
    });
    setDirty(false);
  }, [thresholds]);

  const set =
    (key: keyof VisibleThresholds) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setForm((p) => ({ ...p, [key]: parseFloat(e.target.value) || 0 }));
      setDirty(true);
    };

  const groups: ThresholdGroup[] = [
    {
      title: "Monitoramento",
      desc: "Controlam a frequencia de checagem e quando abrir o alerta HTML no navegador.",
      fields: [
        {
          key: "watch_interval_min",
          label: "Intervalo de watch",
          hint: "A cada quantos minutos o sistema consulta a cotacao USD/BRL em modo --watch.",
          suffix: "min",
        },
        {
          key: "notify_threshold",
          label: "p(agora) minimo",
          hint: "Probabilidade minima para abrir o alerta HTML. 0.40 = so alerta quando o modelo tem >= 40% de conviccao em converter agora.",
          step: "0.05",
        },
        {
          key: "notify_cooldown_hours",
          label: "Cooldown de alerta",
          hint: "Horas de silencio entre um alerta HTML e o proximo. Evita spam em alta volatilidade.",
          suffix: "h",
        },
      ],
    },
    {
      title: "Alertas no celular",
      desc: "Disparam via Telegram quando o USD/BRL sobe alem da ancora.",
      fields: [
        {
          key: "alert_threshold_pct",
          label: "Limiar de alta",
          hint: "Percentual de alta vs. a ancora que dispara o alerta. Ex: 1.0 = alerta quando o dolar sobe 1% desde a ultima notificacao.",
          step: "0.1",
          suffix: "%",
        },
        {
          key: "alert_cooldown_min",
          label: "Cooldown de celular",
          hint: "Minutos de silencio entre alertas no Telegram. Alinhe com o watch_interval para nao perder oportunidades.",
          suffix: "min",
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
          para ver a explicacao.
        </p>

        {groups.map((g) => (
          <div key={g.title} className="mb" style={{ marginBottom: "1.8rem" }}>
            <h2
              style={{
                fontSize: "1rem",
                marginBottom: "0.15rem",
                textTransform: "none",
                letterSpacing: "0",
              }}
            >
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
              setForm({
                watch_interval_min: thresholds.watch_interval_min,
                notify_threshold: thresholds.notify_threshold,
                notify_cooldown_hours: thresholds.notify_cooldown_hours,
                alert_threshold_pct: thresholds.alert_threshold_pct,
                alert_cooldown_min: thresholds.alert_cooldown_min,
              });
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
          message="Alterar thresholds afeta o comportamento do --watch, alertas e decisoes de tamanho."
          onConfirm={() => {
            const delta: Partial<Thresholds> = {};
            const visibleKeys: (keyof VisibleThresholds)[] = [
              "watch_interval_min",
              "notify_threshold",
              "notify_cooldown_hours",
              "alert_threshold_pct",
              "alert_cooldown_min",
            ];
            for (const key of visibleKeys) {
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
            <strong>{configured ? "Configurado" : "Nao configurado"}</strong>
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
            <strong>2.</strong> Mande qualquer mensagem pro bot. O chat_id sera
            descoberto automaticamente no teste, ou voce pode colar manualmente.
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
                  "Deixe vazio - descoberto no teste"
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
              ? "Isso vai sobrescrever a configuracao atual do Telegram."
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
        <h2>Referencia CLI</h2>
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
                      {copied === id ? "copiado" : "copiar"}
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
            Parametros disponiveis para todos os comandos.
          </p>
          <dl className="cli-flags">
            {CLI_FLAGS.map((f) => (
              <React.Fragment key={f.flag}>
                <dt>{f.flag}</dt>
                <dd>{f.desc}</dd>
              </React.Fragment>
            ))}
          </dl>
        </div>

        <div className="cli-section">
          <h2>API</h2>
          <p className="cli-desc">
            O servidor em <code>server.py</code> expoe endpoints REST para
            integracao externa.
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
                    {copied === id ? "copiado" : "copiar"}
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
        showToast("Thresholds salvos");
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
        showToast("Configuracao salva");
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
        showToast(data.message || "Teste enviado");
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
        carregando...
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
            <div className="status">
              <StatusDot on={health.healthy} />
              <span>{health.healthy ? "api ok" : "api down"}</span>
              <span className="status-sep">
                {formatUptime(health.uptime_seconds)}
              </span>
            </div>
          )}
          {notifier && (
            <div className="status">
              <StatusDot on={notifier.is_configured} />
              <span>
                {notifier.is_configured ? "telegram" : "telegram off"}
              </span>
            </div>
          )}
        </div>
      </header>

      <nav className="tabs">
        {(
          [
            ["dashboard", "Dashboard"],
            ["history", "Historico"],
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
