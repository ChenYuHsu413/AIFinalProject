"use client";

import { useEffect, useRef, useState } from "react";

import { Bar, Card, Note, Stat } from "@/components/ui-kit";
import { API_BASE } from "@/lib/api";
import { CanvasTrend, type TrendSeries } from "@/components/monitor/CanvasTrend";
import {
  HealthRing,
  severityColorClass,
  STAGE,
  SUB_ZH,
} from "@/components/monitor/shared";

const SUBS = Object.keys(SUB_ZH);
const LIVE_SPEEDS = [1, 2, 4];
const STREAM_HZ = 10; // calmer than 20 Hz for a readable live feel
const BUFFER = 200; // ~20 s of history at 10 Hz
const EMA = 0.3; // smoothing for the instantaneous readouts

// distinct line colours per subsystem (kept away from the red early-warning line)
const SUB_COLORS: Record<string, string> = {
  temperature: "#f97316",
  current: "#eab308",
  vibration: "#a855f7",
  encoder: "#3b82f6",
  motion: "#14b8a6",
  communication: "#94a3b8",
};

interface LiveFrame {
  sid: number;
  seq: number;
  t: number;
  global_t: number;
  new_scenario: boolean;
  scenario_name: string;
  fault_category: string;
  root_cause: string;
  stage_int: number;
  health: number;
  rul: number;
  warning: number;
  alarm: number;
  trip: number;
  pred_prob: number;
  pred_cat: string;
  [subsystem: string]: number | string | boolean;
}

interface FeedEvent {
  t: number;
  kind: "event" | "predict" | "alarm";
  text: string;
}

export function LiveView() {
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [connected, setConnected] = useState(false);
  // Display state is flushed on a steady ~8 fps tick (below), decoupled from the
  // 10 Hz SSE receipt, so the heavy chart re-renders evenly instead of jankily.
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [smooth, setSmooth] = useState<Record<string, number>>({});
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const prev = useRef<{ alarm: number; alerted: boolean; sid: number }>({
    alarm: 0,
    alerted: false,
    sid: -1,
  });
  const smoothRef = useRef<Record<string, number>>({});
  const latestRef = useRef<LiveFrame | null>(null);
  const bufferRef = useRef<LiveFrame[]>([]);

  useEffect(() => {
    if (!running) return; // paused: no stream (indicator keyed off `running`)
    const es = new EventSource(
      `${API_BASE}/monitor/stream?speed=${speed}&hz=${STREAM_HZ}`,
    );
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (e) => {
      const f: LiveFrame = JSON.parse(e.data);
      latestRef.current = f;
      // EMA-smooth the instantaneous readouts (health / pred / subsystems)
      const s = smoothRef.current;
      const keys = ["health", "pred_prob", ...SUBS];
      for (const k of keys) {
        const v = f[k] as number;
        s[k] = s[k] == null ? v : EMA * v + (1 - EMA) * s[k];
      }
      const b = bufferRef.current;
      b.push(f);
      if (b.length > BUFFER) b.shift();
      // rising-edge event detection (rare -> commit immediately)
      const p = prev.current;
      const newEvents: FeedEvent[] = [];
      if (f.sid !== p.sid) {
        newEvents.push({ t: f.global_t, kind: "event", text: `新事件注入：${f.scenario_name}` });
        p.alerted = false;
      }
      if (!p.alerted && f.pred_prob >= 0.5) {
        newEvents.push({ t: f.global_t, kind: "predict", text: `⚠ 模型預警：疑似 ${f.pred_cat}` });
        p.alerted = true;
      }
      if (p.alarm === 0 && f.alarm === 1) {
        newEvents.push({ t: f.global_t, kind: "alarm", text: `🔴 觸發告警：${f.scenario_name}` });
      }
      p.alarm = f.alarm;
      p.sid = f.sid;
      if (newEvents.length) setEvents((ev) => [...newEvents.reverse(), ...ev].slice(0, 8));
    };
    return () => es.close();
  }, [running, speed]);

  // Steady display flush (~8 fps) — smooths rendering vs the 10 Hz SSE receipt.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      const f = latestRef.current;
      if (!f) return;
      setFrame(f);
      setSmooth({ ...smoothRef.current });
    }, 125);
    return () => clearInterval(id);
  }, [running]);

  const toggleRun = () => {
    if (running) {
      setRunning(false); // pause: freeze the current view
    } else {
      // start a fresh simulation run
      bufferRef.current = [];
      latestRef.current = null;
      smoothRef.current = {};
      prev.current = { alarm: 0, alerted: false, sid: -1 };
      setEvents([]);
      setFrame(null);
      setSmooth({});
      setRunning(true);
    }
  };

  const stageInt = frame ? frame.stage_int : 0;
  const stage = STAGE[stageInt] ?? STAGE[0];
  const sHealth = smooth.health ?? (frame ? frame.health : 100);
  const sPred = smooth.pred_prob ?? 0;
  const alerted = frame ? frame.pred_prob >= 0.5 : false;
  // full-page alarm state (only while actively streaming)
  const inAlarm = running && !!frame && frame.alarm === 1;
  const inTrip = running && !!frame && frame.trip === 1;

  const canvasSeries: TrendSeries[] = [
    ...SUBS.map((s) => ({ key: s, label: SUB_ZH[s], color: SUB_COLORS[s] })),
    { key: "pred_prob", label: "預警", color: "#ef4444", thick: true },
  ];

  return (
    <div className="space-y-6">
      {inAlarm && (
        <>
          <div className={`monitor-alarm-overlay ${inTrip ? "is-trip" : ""}`} />
          <div className="monitor-alarm-banner command-pulse rounded-lg bg-red-600 px-6 py-2 text-lg font-black text-white shadow-lg ring-2 ring-red-300/60">
            {inTrip ? "🛑 TRIP · 伺服跳機停機" : "⚠ ALARM · 故障告警"}
            {frame ? ` — ${frame.scenario_name}` : ""}
          </div>
        </>
      )}
      {/* live status bar */}
      <Card>
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={toggleRun}
            className={`rounded-md px-4 py-1.5 text-sm font-semibold ${
              running
                ? "bg-muted text-foreground hover:bg-muted/70"
                : "bg-primary text-primary-foreground"
            }`}
          >
            {running ? "⏸ 暫停" : "▶ 開始模擬串流"}
          </button>
          <span className="flex items-center gap-2 text-sm font-semibold">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${
                running && connected
                  ? "animate-pulse bg-red-500"
                  : running
                    ? "bg-amber-500"
                    : "bg-muted-foreground"
              }`}
            />
            {!running ? "已暫停" : connected ? "● LIVE 感測器串流中" : "連線中…"}
          </span>
          <span className="text-sm text-muted-foreground">
            {frame ? `目前設備事件：${frame.scenario_name}` : "尚未開始"}
          </span>
          <div className="ml-auto flex items-center gap-1 text-sm">
            <span className="text-muted-foreground">速度</span>
            {LIVE_SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`rounded px-2 py-1 text-xs ${
                  speed === s
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:bg-muted/70"
                }`}
              >
                {s}×
              </button>
            ))}
            <span className="ml-2 tabular-nums text-muted-foreground">
              t+{frame ? frame.global_t.toFixed(1) : "0.0"}s
            </span>
          </div>
        </div>
      </Card>

      {alerted && (
        <Note tone={frame && frame.alarm ? "danger" : "warn"}>
          {frame && frame.alarm ? "🔴 已進入告警" : "⚠ 早期預警"} · 疑似原因{" "}
          <strong>{frame?.pred_cat}</strong> · 模型預警機率{" "}
          {frame ? (frame.pred_prob * 100).toFixed(0) : 0}%
        </Note>
      )}

      {/* hero: scrolling multi-channel trend + status stack */}
      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <Card title="即時趨勢（往左捲動 · 子系統嚴重度 + 模型預警，0–1）">
          <CanvasTrend
            frames={() => bufferRef.current}
            series={canvasSeries}
            running={running}
            height={360}
          />
        </Card>

        {/* status stack */}
        <div className="space-y-4">
          <Card className="text-center">
            <p className="text-xs text-muted-foreground">目前狀態</p>
            <p className={`mt-1 text-4xl font-black tracking-tight ${stage.tone}`}>
              {stage.zh}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {frame?.fault_category ?? "—"}
            </p>
          </Card>
          <div className="grid grid-cols-2 gap-4">
            <Stat label="健康分數" value={frame ? sHealth.toFixed(0) : "—"} sub="資料內建" />
            <Stat label="RUL 剩餘" value={frame ? `${frame.rul.toFixed(1)}s` : "—"} sub="資料內建" />
          </div>
          <Card title="AI 早期預警（即時推論）">
            <Bar
              label="模型預警機率"
              right={frame ? `${(sPred * 100).toFixed(0)}%` : "—"}
              value={sPred}
              colorClass={alerted ? "bg-red-500" : "bg-emerald-500"}
              sub={alerted ? "⚠ 模型判定即將／已進入告警" : "運轉正常"}
            />
            <div className="mt-3 flex items-center justify-between text-sm">
              <span className="text-muted-foreground">疑似原因</span>
              <span className="font-medium">{alerted ? frame?.pred_cat : "—"}</span>
            </div>
          </Card>
          <Card className="flex items-center justify-center">
            <HealthRing value={frame ? sHealth : 100} />
          </Card>
        </div>
      </div>

      {/* current-state bar meters (calmer than radar) */}
      <Card title="子系統當前嚴重度（0=正常，1=嚴重；已平滑）">
        <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
          {SUBS.map((s) => {
            const v = smooth[s] ?? 0;
            return (
              <Bar
                key={s}
                label={
                  <span className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-2 w-2 rounded-sm"
                      style={{ backgroundColor: SUB_COLORS[s] }}
                    />
                    {SUB_ZH[s]}
                  </span>
                }
                right={v.toFixed(2)}
                value={v}
                colorClass={severityColorClass(v)}
              />
            );
          })}
        </div>
      </Card>

      {/* event feed */}
      <Card title="事件串流 · Event Feed">
        {events.length === 0 ? (
          <p className="text-sm text-muted-foreground">尚無事件。</p>
        ) : (
          <ul className="space-y-1.5 text-sm">
            {events.map((ev, i) => (
              <li key={`${ev.t}-${i}`} className="flex items-center gap-2">
                <span className="tabular-nums text-xs text-muted-foreground">
                  t+{ev.t.toFixed(1)}s
                </span>
                <span
                  className={
                    ev.kind === "alarm"
                      ? "text-red-500"
                      : ev.kind === "predict"
                        ? "text-amber-600 dark:text-amber-400"
                        : "text-foreground"
                  }
                >
                  {ev.text}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
