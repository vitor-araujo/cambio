type DemoThresholds = {
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
};

type DemoEntry = {
  ts: string;
  rate_signal: number;
  rate_live: number;
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
};

type DemoState = {
  executed: boolean;
  thresholds: DemoThresholds;
  notifierConfigured: boolean;
};

const STORAGE_KEY = "cambio-portfolio-demo-v1";

const defaultThresholds: DemoThresholds = {
  watch_interval_min: 5,
  notify_threshold: 0.4,
  notify_cooldown_hours: 6,
  alert_threshold_pct: 1,
  alert_cooldown_min: 5,
  dca_floor: 0.25,
  dca_ceiling: 0.5,
  cadence_days: 4,
  spread_bps: 50,
  deadline_days: 15,
};

function loadState(): DemoState {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return {
      executed: Boolean(saved.executed),
      thresholds: { ...defaultThresholds, ...(saved.thresholds || {}) },
      notifierConfigured: saved.notifierConfigured !== false,
    };
  } catch {
    return {
      executed: false,
      thresholds: { ...defaultThresholds },
      notifierConfigured: true,
    };
  }
}

let demoState = loadState();

function persistState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(demoState));
  } catch {
    // The demo remains functional when storage is blocked or unavailable.
  }
}

function round(value: number, digits = 4): number {
  return Number(value.toFixed(digits));
}

function createJournal(): DemoEntry[] {
  const now = new Date();
  now.setSeconds(0, 0);
  const ascending: DemoEntry[] = [];

  for (let index = 0; index < 64; index += 1) {
    const progress = index / 63;
    const ts = new Date(now.getTime() - (63 - index) * 15 * 60_000);
    const rate = 5.066 + progress * 0.027 + Math.sin(index / 4.1) * 0.007 + Math.cos(index / 8.2) * 0.003;
    const quality = Math.max(0.08, Math.min(0.92, 0.3 + progress * 0.24 + Math.sin(index / 5.2) * 0.19));
    const hurdle = Math.max(0.35, 0.72 - progress * 0.29);
    const pNow = Math.max(0.18, Math.min(0.68, 0.34 + Math.sin(index / 7.3) * 0.09));
    const action = index === 63 || (index > 8 && index % 19 === 0 && quality >= hurdle);
    const split = !action && pNow >= 0.4;

    ascending.push({
      ts: ts.toISOString(),
      rate_signal: round(rate - 0.0007),
      rate_live: index === 63 ? 5.1029 : round(rate),
      decision: action ? "exchange_now" : split ? "split" : "wait",
      size: round(0.25 + Math.max(0, pNow - 0.3) * 0.42, 3),
      p_now: round(pNow, 3),
      p_split: round(0.2 + (1 - pNow) * 0.17, 3),
      p_wait: round(0.8 - pNow * 1.17, 3),
      composite: round((pNow - 0.5) * 2, 3),
      agreement: round(0.58 + Math.cos(index / 6) * 0.12, 3),
      regime: round(Math.sin(index / 11) * 0.25, 3),
      notified: action,
      executed: false,
      trigger: index === 63 ? "opportunity_window" : action ? "opportunity_window" : quality >= hurdle * 0.92 ? "window_open" : "cadence_building",
      reason: index === 63
        ? "A janela está aberta e a qualidade da taxa superou o limiar dinâmico."
        : "A simulação acompanha o preço dentro da janela limitada.",
      cadence_days: 4,
      days_to_due: index === 63 ? 0.7 : round(Math.max(0.2, 4 - progress * 3.3), 2),
      opportunity_score: index === 63 ? 0.58 : round(quality, 3),
      opportunity_threshold: index === 63 ? 0.43 : round(hurdle, 3),
    });
  }

  return ascending.reverse();
}

const journal = createJournal();

function currentJournal(): DemoEntry[] {
  return journal.map((entry, index) => (
    index === 0 ? { ...entry, executed: demoState.executed } : { ...entry }
  ));
}

function dashboard() {
  const entries = currentJournal();
  return {
    state: {
      anchor_rate: 5.0781,
      anchor_date: new Date().toISOString().slice(0, 10),
      portfolio_demo: true,
    },
    thresholds: { ...demoState.thresholds },
    last_signal: entries[0],
    recent_signals: entries.slice(0, 50),
    total_signals: 79_166,
    total_alerts: 143,
    recent_alerts: entries.filter((entry) => entry.notified).slice(0, 20),
  };
}

function marketSeries(base: number, amplitude: number, count: number) {
  return Array.from({ length: count }, (_, index) => ({
    d: String(index + 1).padStart(2, "0"),
    v: round(base + index * amplitude * 0.08 + Math.sin(index / 2.1) * amplitude, base > 1_000 ? 0 : 4),
  }));
}

function parseBody(init?: RequestInit): Record<string, unknown> {
  if (typeof init?.body !== "string") return {};
  try {
    return JSON.parse(init.body);
  } catch {
    return {};
  }
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

export async function portfolioRequest<T>(path: string, init?: RequestInit): Promise<T> {
  await new Promise((resolve) => window.setTimeout(resolve, init?.method === "POST" ? 220 : 45));
  const pathname = new URL(path, "https://portfolio.local").pathname;
  const method = init?.method || "GET";
  let payload: unknown;

  if (pathname === "/dashboard") {
    payload = dashboard();
  } else if (pathname === "/journal") {
    payload = currentJournal();
  } else if (pathname === "/thresholds" && method === "POST") {
    demoState = {
      ...demoState,
      thresholds: { ...demoState.thresholds, ...parseBody(init) },
    };
    persistState();
    payload = { ok: true, thresholds: { ...demoState.thresholds } };
  } else if (pathname === "/thresholds") {
    payload = { ...demoState.thresholds };
  } else if (pathname === "/executions" && method === "POST") {
    demoState = { ...demoState, executed: true };
    persistState();
    payload = { ok: true, simulated: true };
  } else if (pathname === "/executions/undo" && method === "POST") {
    demoState = { ...demoState, executed: false };
    persistState();
    payload = { ok: true, simulated: true };
  } else if (pathname === "/notifier" && method === "POST") {
    demoState = { ...demoState, notifierConfigured: true };
    persistState();
    payload = { ok: true, simulated: true };
  } else if (pathname === "/notifier") {
    payload = {
      provider: "telegram",
      is_configured: demoState.notifierConfigured,
      missing_keys: [],
      telegram_bot_token: "demo••••••portfolio",
      telegram_chat_id: "local simulation",
      has_token: demoState.notifierConfigured,
      has_chat_id: demoState.notifierConfigured,
    };
  } else if (pathname === "/notifier/test" && method === "POST") {
    payload = { ok: true, message: "Alerta simulado no navegador — nenhuma mensagem foi enviada." };
  } else if (pathname === "/health") {
    payload = { healthy: true, uptime_seconds: 12_640, last_signal_age_min: 0.4, portfolio_demo: true };
  } else if (pathname === "/usdbrl") {
    payload = {
      ok: true,
      price: 5.1029,
      prev_close: 5.0814,
      change: 0.0215,
      change_pct: 0.42,
      sparkline: marketSeries(5.071, 0.009, 24),
    };
  } else if (pathname === "/ibov") {
    payload = {
      ok: true,
      price: 134_822,
      prev_close: 133_794,
      change: 1_028,
      change_pct: 0.77,
      sparkline: marketSeries(132_600, 890, 22),
    };
  } else {
    throw new Error(`Demo route not implemented: ${method} ${pathname}`);
  }

  return clone(payload) as T;
}
