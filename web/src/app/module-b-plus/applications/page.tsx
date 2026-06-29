"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, Note, PageTitle, Stat } from "@/components/ui-kit";
import { apiGet, apiPost } from "@/lib/api";

// ---------------------------------------------------------------------------
// types (page-local, per project convention)
// ---------------------------------------------------------------------------
interface RulRow {
  condition: string;
  bearing: string;
  minute: number;
  rul_true: number | null;
  rul_pred: number | null;
  health: number;
  is_degrading: boolean;
}
interface Advice {
  risk: "green" | "yellow" | "red";
  risk_label_zh: string;
  suggested_window_hours: number | null;
  rationale: string[];
  cost_note: string | null;
}
interface ReplayFrame {
  k: number;
  minute: number;
  hours: number;
  health: number;
  rul_hours: number | null;
  past_fpt: boolean;
  risk: "green" | "yellow" | "red";
  risk_label_zh: string;
  suggested_window_hours: number | null;
  x: number[];
  y: number[];
}
interface Replay {
  available: boolean;
  reason?: string;
  condition?: string;
  bearing?: string;
  hi_base?: number;
  hi_fail?: number;
  fpt_index?: number;
  fpt_minute?: number;
  fpt_hi?: number;
  last_minute?: number;
  frames?: ReplayFrame[];
}
interface Curve {
  condition: string;
  bearing: string;
  life_pct: number[];
  hi: number[];
}
interface Overlay {
  available: boolean;
  curves: Curve[];
}
interface LoboLoco {
  lobo: { pooled?: Record<string, number> };
  loco: { pooled?: Record<string, number> };
}

type RiskKey = "green" | "yellow" | "red";
const RISK_META: Record<RiskKey, { chip: string; dot: string; stroke: string }> = {
  green: {
    chip: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 ring-1 ring-inset ring-emerald-500/30",
    dot: "bg-emerald-400",
    stroke: "#34d399",
  },
  yellow: {
    chip: "bg-amber-500/15 text-amber-700 dark:text-amber-300 ring-1 ring-inset ring-amber-500/30",
    dot: "bg-amber-400",
    stroke: "#fbbf24",
  },
  red: {
    chip: "bg-red-500/15 text-red-700 dark:text-red-300 ring-1 ring-inset ring-red-500/30",
    dot: "bg-red-400",
    stroke: "#f87171",
  },
};

const fmtHours = (h: number | null | undefined) =>
  h == null ? "—" : `${h.toFixed(1)} h`;

// ===========================================================================
// page
// ===========================================================================
export default function ModuleBPlusApplicationsPage() {
  const [rows, setRows] = useState<RulRow[] | null>(null);
  const [ll, setLl] = useState<LoboLoco | null>(null);
  const [ov, setOv] = useState<Overlay | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    Promise.all([
      apiGet<RulRow[]>("/xjtu/rul_predictions"),
      apiGet<LoboLoco>("/xjtu/lobo_loco"),
      apiGet<Overlay>("/xjtu/health_overlay"),
    ])
      .then(([r, l, o]) => {
        setRows(r);
        setLl(l);
        setOv(o);
      })
      .catch(() => setErr(true));
  }, []);

  // group rows by trajectory, sorted by minute
  const groups = useMemo(() => {
    const m = new Map<string, RulRow[]>();
    for (const r of rows ?? []) {
      const k = `${r.condition}|${r.bearing}`;
      let arr = m.get(k);
      if (!arr) {
        arr = [];
        m.set(k, arr);
      }
      arr.push(r);
    }
    for (const arr of m.values()) arr.sort((a, b) => a.minute - b.minute);
    return m;
  }, [rows]);
  const bearingKeys = useMemo(() => Array.from(groups.keys()), [groups]);

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 B+ · 延伸應用（XJTU-SY）"
        desc="把多軌跡泛化的結果接到可操作的決策層：維護建議（E2）、即時串流回放（E3）、以及監督式 RUL 的 LOBO/LOCO 泛化落差。"
      />
      <Note tone="warn" className="mb-6">
        本頁為<b>決策支援（DECISION SUPPORT）非控制</b>：建議時間窗 = 剩餘壽命 ×（1 − 安全裕度 30%），
        成本為示意值，未對真實維護結果驗證。RUL 為趨勢外推估計，跨工況絕對小時數並未被「解決」（見下方 LOBO/LOCO）。
      </Note>
      {err && (
        <Note tone="danger" className="mb-6">
          無法載入 B+ 延伸應用資料，請確認後端已啟動。
        </Note>
      )}

      {rows != null && bearingKeys.length > 0 && (
        <div className="space-y-8">
          <MaintenanceAdvice groups={groups} bearingKeys={bearingKeys} />
          <StreamingReplay bearingKeys={bearingKeys} />
          {ll && <LoboLocoTable ll={ll} />}
        </div>
      )}

      {rows != null && bearingKeys.length === 0 && (
        <Note tone="info">
          尚未產生 XJTU RUL 預測（outputs/metrics/xjtu_rul_predictions.csv）。請於本機執行
          <code className="mx-1">python -m src.models.train_rul_lobo</code>等流程後再試。
        </Note>
      )}

      {ov && <HealthOverlay ov={ov} />}
    </div>
  );
}

// ===========================================================================
// E2 — maintenance advice fleet at a chosen inspection checkpoint
// ===========================================================================
function MaintenanceAdvice({
  groups,
  bearingKeys,
}: {
  groups: Map<string, RulRow[]>;
  bearingKeys: string[];
}) {
  const [checkpoint, setCheckpoint] = useState(70);
  const [showCost, setShowCost] = useState(false);
  const [advice, setAdvice] = useState<Map<string, Advice>>(new Map());
  const [busy, setBusy] = useState(false);

  // Debounce so dragging the slider doesn't fire a request storm; advice is
  // always computed server-side (/maintenance/advice is the single source of
  // truth, shared with Streamlit) rather than duplicating the heuristic here.
  useEffect(() => {
    let cancelled = false;
    const t = setTimeout(() => {
      setBusy(true);
      const reqs = bearingKeys.map(async (k) => {
        const arr = groups.get(k)!;
        const idx = Math.min(
          Math.round((checkpoint / 100) * (arr.length - 1)),
          arr.length - 1,
        );
        const row = arr[idx];
        const adv = await apiPost<Advice>("/maintenance/advice", {
          health: row.health,
          rul_hours: row.rul_pred,
          past_fpt: row.is_degrading,
          ...(showCost ? { cost_unplanned: 10000, cost_planned: 2000 } : {}),
        });
        return [k, adv] as const;
      });
      Promise.all(reqs)
        .then((pairs) => {
          if (!cancelled) setAdvice(new Map(pairs));
        })
        .catch(() => {})
        .finally(() => {
          if (!cancelled) setBusy(false);
        });
    }, 200);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [checkpoint, showCost, groups, bearingKeys]);

  const counts = useMemo(() => {
    const c = { green: 0, yellow: 0, red: 0 };
    for (const a of advice.values()) c[a.risk] += 1;
    return c;
  }, [advice]);

  return (
    <section>
      <h2 className="mb-1 text-lg font-semibold">維護建議（決策支援 · E2）</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        run-to-failure 資料沒有「真實的現在」；此處模擬在某巡檢時點評估，顯示系統當下會給的建議。
        拉到 100% 即各軸承失效當下。
      </p>

      <Card className="mb-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <label className="flex-1">
            <span className="text-xs text-muted-foreground">
              巡檢檢查點（佔壽命比例）：<b className="text-foreground">{checkpoint}%</b>
            </span>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={checkpoint}
              onChange={(e) => setCheckpoint(Number(e.target.value))}
              className="mt-2 w-full accent-primary"
            />
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={showCost}
              onChange={(e) => setShowCost(e.target.checked)}
              className="accent-primary"
            />
            顯示成本對照（示意）
          </label>
        </div>
      </Card>

      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="巡檢時點" value={`${checkpoint}%`} sub="佔各軸承壽命比例" />
        <Stat label="🟢 健康" value={String(counts.green)} sub="尚未退化" valueClass="text-emerald-400" />
        <Stat label="🟡 退化中" value={String(counts.yellow)} sub="已過 FPT、可規劃" valueClass="text-amber-400" />
        <Stat label="🔴 迫近失效" value={String(counts.red)} sub="健康度跌破告警" valueClass="text-red-400" />
      </div>

      <div className={busy ? "opacity-60 transition-opacity" : "transition-opacity"}>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
          {bearingKeys.map((k) => {
            const adv = advice.get(k);
            const bearing = k.split("|")[1];
            const meta = adv ? RISK_META[adv.risk] : RISK_META.green;
            return (
              <div
                key={k}
                className="rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm"
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm font-semibold">{bearing}</span>
                  {adv && (
                    <span className={`rounded-full px-2 py-0.5 text-[11px] ${meta.chip}`}>
                      <span className={`mr-1 inline-block h-1.5 w-1.5 rounded-full ${meta.dot}`} />
                      {adv.risk_label_zh}
                    </span>
                  )}
                </div>
                {adv ? (
                  <>
                    <p className="text-xs text-muted-foreground">
                      建議維護時間窗：
                      <b className="ml-1 text-foreground">{fmtHours(adv.suggested_window_hours)}</b>
                    </p>
                    <ul className="mt-2 space-y-1 text-xs leading-relaxed text-muted-foreground">
                      {adv.rationale.map((r, i) => (
                        <li key={i}>· {r}</li>
                      ))}
                    </ul>
                    {adv.cost_note && (
                      <p className="mt-2 rounded-md bg-muted/40 px-2 py-1.5 text-[11px] text-muted-foreground">
                        {adv.cost_note}
                      </p>
                    )}
                  </>
                ) : (
                  <p className="text-xs text-muted-foreground">計算中…</p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

// ===========================================================================
// E3 — client-side streaming replay of one bearing
// ===========================================================================
function StreamingReplay({ bearingKeys }: { bearingKeys: string[] }) {
  const [selected, setSelected] = useState(bearingKeys[0]);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [err, setErr] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  // load frames when the bearing changes (resets live in the change handler so
  // the effect body has no synchronous setState — cascading-render lint rule)
  useEffect(() => {
    const [condition, bearing] = selected.split("|");
    let cancelled = false;
    apiGet<Replay>(`/xjtu/replay/${encodeURIComponent(condition)}/${encodeURIComponent(bearing)}`)
      .then((r) => {
        if (!cancelled) setReplay(r);
      })
      .catch(() => {
        if (!cancelled) setErr(true);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const selectBearing = (k: string) => {
    setReplay(null);
    setErr(false);
    setPlaying(false);
    setFrameIdx(0);
    setSelected(k);
  };

  const frames = replay?.frames ?? [];

  // playback timer
  useEffect(() => {
    if (!playing || frames.length === 0) return;
    const id = setInterval(() => {
      setFrameIdx((i) => {
        if (i + 1 >= frames.length) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 120);
    return () => clearInterval(id);
  }, [playing, frames.length]);

  const frame = frames[Math.min(frameIdx, Math.max(0, frames.length - 1))];
  const chartData = useMemo(
    () => (frame ? frame.x.map((x, i) => ({ x, y: frame.y[i] })) : []),
    [frame],
  );
  const meta = frame ? RISK_META[frame.risk] : RISK_META.green;

  return (
    <section>
      <h2 className="mb-1 text-lg font-semibold">即時串流回放（會動的監測台 · E3）</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        選一顆軸承播放：健康指標 HI 一格格長、跨過退化起點 FPT 後目前點轉紅，狀態框即時顯示健康度 / RUL / 風險（重用 E2 邏輯）。
        RUL 以當前可見前綴外推（回溯式，與逐前綴重算等價）。
      </p>

      <Card>
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <select
            value={selected}
            onChange={(e) => selectBearing(e.target.value)}
            className="rounded-md border border-border bg-card px-2 py-1.5 text-sm"
          >
            {bearingKeys.map((k) => (
              <option key={k} value={k}>
                {k.replace("|", " · ")}
              </option>
            ))}
          </select>
          <button
            onClick={() => setPlaying((p) => !p)}
            disabled={frames.length === 0}
            className="rounded-md border border-border bg-primary/10 px-3 py-1.5 text-sm font-medium text-primary hover:bg-primary/20 disabled:opacity-50"
          >
            {playing ? "⏸ 暫停" : "▶ 播放"}
          </button>
          <button
            onClick={() => {
              setPlaying(false);
              setFrameIdx(0);
            }}
            disabled={frames.length === 0}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted/40 disabled:opacity-50"
          >
            ↺ 重置
          </button>
          {frame && (
            <span
              className={`ml-auto rounded-full px-2.5 py-1 text-xs ${meta.chip}`}
            >
              <span className={`mr-1 inline-block h-1.5 w-1.5 rounded-full ${meta.dot}`} />
              {frame.risk_label_zh}　健康 {frame.health.toFixed(0)}　RUL {fmtHours(frame.rul_hours)}
            </span>
          )}
        </div>

        {err && <Note tone="danger">無法載入回放資料。</Note>}
        {!err && frames.length === 0 && (
          <p className="py-10 text-center text-sm text-muted-foreground">載入軌跡中…</p>
        )}

        {frame && replay?.available && (
          <>
            <div className="h-[320px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 4, left: -12 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" vertical={false} />
                  <XAxis
                    type="number"
                    dataKey="x"
                    domain={[0, replay.last_minute ?? "dataMax"]}
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    className="text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v: number) => `${Math.round(v)}m`}
                  />
                  <YAxis
                    domain={[0, "dataMax"]}
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    className="text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                    width={40}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--popover)",
                      border: "1px solid var(--border)",
                      borderRadius: 8,
                      fontSize: 12,
                      color: "var(--popover-foreground)",
                    }}
                    labelFormatter={(v) => `第 ${Math.round(Number(v))} 分鐘`}
                    formatter={(v) => [Number(v).toFixed(3), "HI"]}
                  />
                  {replay.hi_base != null && (
                    <ReferenceLine y={replay.hi_base} stroke="var(--muted-foreground)" strokeDasharray="2 4" />
                  )}
                  {replay.hi_fail != null && (
                    <ReferenceLine y={replay.hi_fail} stroke="#f87171" strokeDasharray="4 4" />
                  )}
                  {replay.fpt_minute != null && (
                    <ReferenceLine x={replay.fpt_minute} stroke="#22d3ee" strokeDasharray="3 3" />
                  )}
                  <Line type="monotone" dataKey="y" stroke={meta.stroke} strokeWidth={2.5} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(0, frames.length - 1)}
              step={1}
              value={frameIdx}
              onChange={(e) => {
                setPlaying(false);
                setFrameIdx(Number(e.target.value));
              }}
              className="mt-2 w-full accent-primary"
            />
            <div className="flex justify-between text-[11px] text-muted-foreground">
              <span>快照 {frame.k}</span>
              <span>HI 基線 / 失效門檻 · ✦ FPT 第 {replay.fpt_index} 快照</span>
            </div>
          </>
        )}

        {replay && !replay.available && (
          <Note tone="info">{replay.reason ?? "回放資料尚未就緒。"}</Note>
        )}
      </Card>
    </section>
  );
}

// ===========================================================================
// LOBO vs LOCO — supervised-RUL generalization gap
// ===========================================================================
function LoboLocoTable({ ll }: { ll: LoboLoco }) {
  const lobo = ll.lobo?.pooled;
  const loco = ll.loco?.pooled;
  if (!lobo && !loco) return null;
  const rowsDef: { key: string; label: string; fmt: (v: number) => string }[] = [
    { key: "r2", label: "合併 R²", fmt: (v) => v.toFixed(2) },
    { key: "mae_hours", label: "合併 MAE (h)", fmt: (v) => v.toFixed(1) },
    { key: "rmse_hours", label: "合併 RMSE (h)", fmt: (v) => v.toFixed(1) },
  ];
  return (
    <section>
      <h2 className="mb-1 text-lg font-semibold">監督式 RUL 泛化：LOBO vs LOCO</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        同一組模型 / 特徵，唯一差別是測試工況有沒有在訓練裡看過：LOBO（同工況、留一軸承）vs LOCO（留一工況）。
        兩者 R² 皆為負，且 LOCO 更差 —— 這就是跨工況 domain shift 的代價（E1 域適應只把它抬高、未解決）。
      </p>
      <Card>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="py-2">指標</th>
              <th className="py-2 text-right tabular-nums">LOBO（留一軸承）</th>
              <th className="py-2 text-right tabular-nums">LOCO（留一工況）</th>
            </tr>
          </thead>
          <tbody>
            {rowsDef.map((d) => (
              <tr key={d.key} className="border-b border-border/40 hover:bg-muted/30">
                <td className="py-2">{d.label}</td>
                <td className="py-2 text-right tabular-nums">
                  {lobo?.[d.key] != null ? d.fmt(lobo[d.key]) : "—"}
                </td>
                <td className="py-2 text-right tabular-nums">
                  {loco?.[d.key] != null ? d.fmt(loco[d.key]) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </section>
  );
}

// ===========================================================================
// HI overlay (kept; degrades on cloud where the 21 GB raw data isn't packaged)
// ===========================================================================
function HealthOverlay({ ov }: { ov: Overlay }) {
  const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];
  return (
    <section className="mt-8">
      <h2 className="mb-1 text-lg font-semibold">多軌跡健康指標重疊（HI）</h2>
      {ov.available ? (
        <div className="rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-semibold">HI 重疊圖（{ov.curves.length} 顆軸承）</span>
            <span className="text-[11px] text-muted-foreground">x：壽命 % · y：健康指標</span>
          </div>
          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ top: 8, right: 8, bottom: 4, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" vertical={false} />
                <XAxis
                  type="number"
                  dataKey="x"
                  domain={[0, 100]}
                  tickFormatter={(v: number) => `${Math.round(v)}%`}
                  tick={{ fontSize: 11, fill: "currentColor" }}
                  className="text-muted-foreground"
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "currentColor" }}
                  className="text-muted-foreground"
                  tickLine={false}
                  axisLine={false}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--popover)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "var(--popover-foreground)",
                  }}
                />
                {ov.curves.map((c, i) => (
                  <Line
                    key={`${c.condition}-${c.bearing}`}
                    type="monotone"
                    dataKey="y"
                    data={c.life_pct.map((x, j) => ({ x, y: c.hi[j] }))}
                    name={c.bearing}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={1.5}
                    strokeOpacity={0.55}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <Note tone="info">
          HI 重疊圖需<b>原始振動資料</b>即時重算；雲端 demo 未打包該資料（約 21 GB）。
          E2 / E3 / LOBO-LOCO 皆由已提交的彙整結果驅動，於雲端可正常運作。
        </Note>
      )}
    </section>
  );
}
