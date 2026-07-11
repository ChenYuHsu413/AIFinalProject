"use client";

import { useEffect, useRef, useState } from "react";
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
import {
  API_BASE,
  apiGet,
  type ServoMonitorEvent,
  type ServoMonitorEventsResponse,
  type ServoMonitorFrame,
  type ServoMonitorWorkOrder,
} from "@/lib/api";
import { HEALTH_COLOR, HEALTH_ZH, RISK_COLOR, RISK_ZH } from "@/lib/servo";

// The two replay scripts the backend `segment` query param selects.
const SEGMENTS = [
  { key: "normal", label: "一般段落（LN→LO→HI）" },
  { key: "drift", label: "注入漂移段（HI 段感測器增益）" },
] as const;
type SegmentKey = (typeof SEGMENTS)[number]["key"];

const BUFFER = 160; // rolling windows kept for the trend charts

// Feed styling per event type (main alert / consistency / DRIFT colour-coded).
const EVENT_STYLE: Record<string, { border: string; text: string; label: string }> = {
  alert_triggered: { border: "#ef4444", text: "text-red-500", label: "🔴 主告警觸發" },
  alert_cleared: { border: "#10b981", text: "text-emerald-500", label: "🟢 告警解除" },
  consistency_warning: { border: "#f59e0b", text: "text-amber-500", label: "⚠ 矛盾提示" },
  drift_detected: { border: "#8b5cf6", text: "text-violet-500", label: "🟣 DRIFT 漂移偵測" },
  drift_cleared: { border: "#0ea5e9", text: "text-sky-500", label: "🔵 漂移解除" },
};

interface SeriesPoint {
  ts: number;
  dv: number;
  conf: number;
  driftErr: number | null;
}

export default function ServoMonitorPage() {
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [offline, setOffline] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [segment, setSegment] = useState<SegmentKey>("normal");

  const [frame, setFrame] = useState<ServoMonitorFrame | null>(null);
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const [feed, setFeed] = useState<ServoMonitorEvent[]>([]);
  const [workOrders, setWorkOrders] = useState<Record<string, ServoMonitorWorkOrder>>({});
  const seenRef = useRef<Set<string>>(new Set());

  // --- SSE: consume the backend's enriched per-window stream ------------------
  useEffect(() => {
    if (!running) return;
    const es = new EventSource(`${API_BASE}/servo/monitor/stream?segment=${segment}`);
    es.onopen = () => {
      setConnected(true);
      setOffline(false);
    };
    es.onmessage = (e) => {
      let f: ServoMonitorFrame;
      try {
        f = JSON.parse(e.data) as ServoMonitorFrame;
      } catch {
        return;
      }
      setConnected(true);
      setOffline(false);
      setFrame(f);
      setSeries((s) =>
        [
          ...s,
          {
            ts: f.window_ts,
            dv: f.degradation_score,
            conf: f.model_confidence,
            driftErr: f.drift_status.available
              ? (f.drift_status.instant_recon_error ?? null)
              : null,
          },
        ].slice(-BUFFER),
      );
      if (f.events.length) {
        const fresh = f.events.filter(
          (ev) => EVENT_STYLE[ev.type] && !seenRef.current.has(ev.id),
        );
        for (const ev of fresh) seenRef.current.add(ev.id);
        if (fresh.length) setFeed((prev) => [...fresh.reverse(), ...prev].slice(0, 40));
      }
    };
    es.addEventListener("done", () => {
      setCompleted(true);
      setRunning(false);
      es.close();
    });
    es.onerror = () => {
      // EventSource can't distinguish a dropped connection from a backend that
      // was never up; reflect it and let it auto-retry (no white screen).
      setConnected(false);
      setOffline(true);
    };
    return () => es.close();
  }, [running, segment]);

  // --- Poll the events endpoint for async LLM work-order drafts ---------------
  // The draft is generated on a backend thread AFTER the alert window streams by,
  // so it never rides the SSE frame — pick it up here and attach by alert_id.
  useEffect(() => {
    if (!running && !completed) return;
    let alive = true;
    const pull = async () => {
      try {
        const r = await apiGet<ServoMonitorEventsResponse>("/servo/monitor/events?limit=50");
        if (!alive) return;
        setWorkOrders((prev) => {
          const next = { ...prev };
          for (const wo of r.work_orders) next[wo.alert_id] = wo;
          return next;
        });
      } catch {
        /* transient; the SSE offline banner already covers backend-down */
      }
    };
    pull();
    const id = setInterval(pull, 2500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [running, completed]);

  const toggle = () => {
    if (running) {
      setRunning(false);
      return;
    }
    // fresh run
    seenRef.current = new Set();
    setFrame(null);
    setSeries([]);
    setFeed([]);
    setWorkOrders({});
    setCompleted(false);
    setOffline(false);
    setRunning(true);
  };

  const smoothed = frame?.smoothed_state ?? "LN";
  const smoothColor = HEALTH_COLOR[smoothed]?.hex ?? "#94a3b8";
  const raw = frame?.predicted_health_state ?? smoothed;
  const rawMatchesSmoothed = raw === smoothed;
  const drift = frame?.drift_status;
  const threshold = drift?.available ? (drift.threshold_p95 ?? null) : null;
  const alertActive = frame?.alert_state.active ?? false;
  const driftTriggered = drift?.triggered ?? false;

  return (
    <div className="space-y-6">
      <PageTitle
        title="伺服馬達即時監控（FMCRD replay）"
        desc="真實 FMCRD 測試資料回放 → 後端視窗聚合 + 參考模型推論 + 告警遲滯 + 漂移偵測；前端只渲染。後端算、前端畫。"
      />

      {/* status / controls bar */}
      <Card>
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={toggle}
            className={`rounded-md px-4 py-1.5 text-sm font-semibold ${
              running
                ? "bg-muted text-foreground hover:bg-muted/70"
                : "bg-primary text-primary-foreground"
            }`}
          >
            {running ? "⏸ 停止監控" : "▶ 開始監控"}
          </button>
          <span className="flex items-center gap-2 text-sm font-semibold">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${
                running && connected
                  ? "animate-pulse bg-emerald-500"
                  : running
                    ? "bg-amber-500"
                    : completed
                      ? "bg-sky-500"
                      : "bg-muted-foreground"
              }`}
            />
            {!running
              ? completed
                ? "本輪回放結束"
                : "已停止"
              : connected
                ? "● LIVE 串流中"
                : "連線中…"}
          </span>

          {/* segment (replay script) selector — locked while streaming */}
          <div className="ml-auto flex items-center gap-1 text-xs">
            <span className="text-muted-foreground">劇本</span>
            {SEGMENTS.map((s) => (
              <button
                key={s.key}
                disabled={running}
                onClick={() => setSegment(s.key)}
                className={`rounded px-2.5 py-1 font-medium disabled:opacity-50 ${
                  segment === s.key
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:bg-muted/70"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {offline && running && (
        <Note tone="danger">
          後端串流不可用或連線中斷。請確認 FastAPI 已啟動
          （<code>uvicorn app.backend.main:app --port 8000</code>），瀏覽器會自動重試。
        </Note>
      )}

      {/* data-source annotation (honest: high-fidelity simulation, not a live line) */}
      <Note tone="warn">
        數據源：<strong>真實 FMCRD 測試資料</strong>回放
        {frame ? `（${frame.replay_segment.label}）` : `（${SEGMENTS.find((s) => s.key === segment)?.label}）`}
        ，為高擬真模擬資料集、非真實產線 log。
        {frame?.replay_segment.injected ? "本劇本已注入模擬感測器漂移，用於觸發漂移偵測。" : ""}
      </Note>

      {/* top: status light + raw-vs-smoothed honesty + model_version badge */}
      <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <Card>
          <div className="flex items-center gap-4">
            <div
              className="h-14 w-14 flex-none rounded-full"
              style={{ background: smoothColor, boxShadow: `0 0 20px ${smoothColor}99` }}
            />
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold" style={{ color: smoothColor }}>
                  {HEALTH_ZH[smoothed] ?? smoothed}
                </span>
                <span className="text-sm text-muted-foreground">（{smoothed}）</span>
              </div>
              <p className="mt-0.5 text-xs text-muted-foreground">
                狀態燈＝近 {frame?.recent_states.length ?? 3} 窗多數決平滑
                {frame ? (
                  rawMatchesSmoothed ? (
                    <span>（與本窗原始預測一致）</span>
                  ) : (
                    <span>
                      （本窗原始預測：
                      <strong className="text-foreground">
                        {HEALTH_ZH[raw] ?? raw}（{raw}）
                      </strong>
                      ）
                    </span>
                  )
                ) : (
                  <span>（尚未開始）</span>
                )}
              </p>
              {frame && (
                <p className="mt-1 font-mono text-xs text-muted-foreground">
                  逐窗原始：{frame.recent_states.join(" · ")}
                </p>
              )}
            </div>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-md bg-violet-500/15 px-2 py-1 font-medium text-violet-600 ring-1 ring-inset ring-violet-500/25 dark:text-violet-300">
              模型版本 {frame?.model_version ?? "—"}
            </span>
            {frame && (
              <span
                className={`rounded-md px-2 py-1 font-medium ${RISK_COLOR[frame.risk_level] ?? ""}`}
              >
                風險 {RISK_ZH[frame.risk_level] ?? frame.risk_level}
              </span>
            )}
            {frame && (
              <span className="text-muted-foreground">
                視窗 #{frame.window_index} · t={frame.window_ts.toFixed(1)}s · true=
                {frame.true_label ?? "?"}
              </span>
            )}
          </div>
        </Card>

        <div className="grid grid-cols-2 gap-4">
          <Stat
            label="退化分數 DV"
            value={frame ? frame.degradation_score.toFixed(3) : "—"}
            sub="0 健康 → 1 重度退化"
            valueClass={alertActive ? "text-red-500" : undefined}
          />
          <Stat
            label="模型信心"
            value={frame ? `${(frame.model_confidence * 100).toFixed(0)}%` : "—"}
            sub="分類器最大機率"
          />
          <Stat
            label="告警狀態"
            value={alertActive ? "🔴 已觸發" : "🟢 正常"}
            sub={
              frame
                ? `High 連續 ${frame.alert_state.high_streak} · Low 連續 ${frame.alert_state.low_streak}`
                : "遲滯 N/M"
            }
            valueClass={alertActive ? "text-red-500" : "text-emerald-500"}
          />
          <Stat
            label="漂移狀態"
            value={!drift?.available ? "無基線" : driftTriggered ? "🟣 觸發" : "正常"}
            sub={
              drift?.available && drift.rolling_recon_error != null
                ? `重建誤差 ${drift.rolling_recon_error.toFixed(3)} / P95 ${threshold?.toFixed(3) ?? "—"}`
                : "此版本無漂移基線"
            }
            valueClass={driftTriggered ? "text-violet-500" : undefined}
          />
        </div>
      </div>

      {alertActive && (
        <Note tone="danger">
          🔴 主告警作用中（連續高風險視窗達遲滯門檻）· 告警編號{" "}
          <strong>{frame?.alert_state.active_alert_id ?? "—"}</strong>
        </Note>
      )}
      {driftTriggered && (
        <Note tone="warn">
          🟣 偵測到資料分布漂移（滾動重建誤差 &gt; 訓練 P95）——退化不會誤觸此訊號，此為真正的分布偏移。
        </Note>
      )}

      {/* middle: rolling trends + drift error mini-chart */}
      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <Card title="退化分數 / 模型信心（滾動時序）">
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series} margin={{ top: 6, right: 10, bottom: 0, left: -18 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="currentColor"
                  className="text-border"
                  vertical={false}
                />
                <XAxis
                  dataKey="ts"
                  tick={{ fontSize: 10, fill: "currentColor" }}
                  className="text-muted-foreground"
                  tickLine={false}
                  axisLine={false}
                  minTickGap={28}
                  tickFormatter={(v: number) => `${v.toFixed(0)}s`}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fill: "currentColor" }}
                  className="text-muted-foreground"
                  tickLine={false}
                  axisLine={false}
                  width={42}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--popover)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "var(--popover-foreground)",
                  }}
                  labelFormatter={(v) => `t=${Number(v).toFixed(1)}s`}
                />
                <Line
                  type="monotone"
                  dataKey="dv"
                  name="退化分數"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="conf"
                  name="模型信心"
                  stroke="#6366f1"
                  strokeWidth={2}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            <span className="text-red-500">■</span> 退化分數（DV）
            <span className="mx-2 text-indigo-500">■</span> 模型信心
          </p>
        </Card>

        <Card title="漂移：重建誤差 vs 閾值">
          {drift?.available ? (
            <div className="h-[260px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series} margin={{ top: 6, right: 10, bottom: 0, left: -12 }}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="currentColor"
                    className="text-border"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="ts"
                    tick={{ fontSize: 10, fill: "currentColor" }}
                    className="text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                    minTickGap={28}
                    tickFormatter={(v: number) => `${v.toFixed(0)}s`}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "currentColor" }}
                    className="text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                    width={48}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--popover)",
                      border: "1px solid var(--border)",
                      borderRadius: 8,
                      fontSize: 12,
                      color: "var(--popover-foreground)",
                    }}
                    labelFormatter={(v) => `t=${Number(v).toFixed(1)}s`}
                  />
                  {threshold != null && (
                    <ReferenceLine
                      y={threshold}
                      stroke="#8b5cf6"
                      strokeDasharray="5 4"
                      label={{
                        value: `P95 ${threshold.toFixed(2)}`,
                        fill: "#8b5cf6",
                        fontSize: 10,
                        position: "insideTopRight",
                      }}
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="driftErr"
                    name="重建誤差"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex h-[260px] items-center justify-center text-center text-sm text-muted-foreground">
              此模型版本尚無漂移基線，
              <br />
              漂移偵測略過。
            </div>
          )}
          <p className="mt-1 text-xs text-muted-foreground">
            誤差高於 P95 且持續 N 窗才觸發；正常退化維持在分布內、不誤觸。
          </p>
        </Card>
      </div>

      {/* bottom: colour-coded event feed with expandable work-order draft */}
      <Card title="事件流（最新在上 · 主告警 / 矛盾 / DRIFT 分色）">
        {feed.length === 0 ? (
          <p className="text-sm text-muted-foreground">尚無事件。</p>
        ) : (
          <ul className="space-y-2">
            {feed.map((ev) => {
              const style = EVENT_STYLE[ev.type];
              const wo = ev.type === "alert_triggered" ? workOrders[ev.id] : undefined;
              return (
                <li
                  key={ev.id}
                  className="rounded-md border-l-4 bg-muted/30 px-3 py-2"
                  style={{ borderLeftColor: style.border }}
                >
                  <div className="flex items-center gap-2 text-sm">
                    <span className={`font-semibold ${style.text}`}>{style.label}</span>
                    {ev.stream_t != null && (
                      <span className="tabular-nums text-xs text-muted-foreground">
                        t={ev.stream_t}s
                      </span>
                    )}
                    <span className="text-xs text-muted-foreground">{ev.id}</span>
                  </div>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {ev.trigger_rule ?? ev.clear_rule ?? ev.message ?? ev.reason ?? ""}
                  </p>
                  {ev.type === "alert_triggered" && (
                    <details className="mt-1">
                      <summary className="cursor-pointer text-xs font-medium text-foreground">
                        📝 工單草稿{wo ? `（${wo.llm_source ?? "offline"}）` : "（生成中…）"}
                      </summary>
                      <pre className="mt-1 whitespace-pre-wrap rounded bg-background/60 p-2 text-xs text-muted-foreground">
                        {wo ? wo.text : "工單草稿由 LLM 於背景非同步生成，不阻塞告警。稍候刷新…"}
                      </pre>
                    </details>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </Card>
    </div>
  );
}
