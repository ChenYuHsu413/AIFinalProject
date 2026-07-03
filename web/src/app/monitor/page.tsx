"use client";

import { useEffect, useState } from "react";
import {
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Bar, Card, Note, PageTitle, Stat } from "@/components/ui-kit";
import {
  apiGet,
  type MonitorFrame,
  type MonitorPack,
  type MonitorScenarios,
} from "@/lib/api";
import { LiveView } from "@/components/monitor/LiveView";
import {
  HealthRing,
  severityColorClass,
  SPEEDS,
  STAGE,
  SUB_ZH,
} from "@/components/monitor/shared";

export default function MonitorPage() {
  // Two modes: live SSE sensor feed (default) and finite scenario replay.
  const [mode, setMode] = useState<"live" | "replay">("live");

  return (
    <div className="space-y-6">
      <PageTitle
        title="即時監控雷達（合成 demo）"
        desc="以 AI 生成的伺服訊號模擬即時感測器串流，六角雷達一眼看出哪個子系統在惡化；早期預警模型在真正告警前先亮燈。"
      />

      <Note tone="warn">
        <strong>合成資料 demo。</strong>
        此軌以 vendored 產生器**即時生成**模擬伺服訊號（非真實產線遙測、非真實感測器，與真實 PHM 主線分開、不取代之）。
        雷達各軸為透明的「相對健康基線偏離度」；告警燈與提前量來自僅用物理遙測訓練、於留出情境評估的早期預警模型。
      </Note>

      {/* mode toggle */}
      <div className="inline-flex rounded-lg border border-border/70 bg-card/70 p-1 text-sm">
        {(["live", "replay"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-md px-4 py-1.5 font-medium ${
              mode === m
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {m === "live" ? "● 即時串流" : "情境回放"}
          </button>
        ))}
      </div>

      {mode === "live" ? <LiveView /> : <ReplayView />}
    </div>
  );
}

function ReplayView() {
  const [index, setIndex] = useState<MonitorScenarios | null>(null);
  const [scenarioId, setScenarioId] = useState<number | null>(null);
  const [pack, setPack] = useState<MonitorPack | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [error, setError] = useState<string | null>(null);

  // load scenario index once
  useEffect(() => {
    apiGet<MonitorScenarios>("/monitor/scenarios")
      .then((d) => {
        setIndex(d);
        if (d.available && d.scenarios.length) {
          // default to the first fault scenario (skip healthy baseline)
          const first = d.scenarios.find((s) => s.fault_category !== "NORMAL");
          setScenarioId((first ?? d.scenarios[0]).scenario_id);
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  // load a scenario pack when the selection changes (playback/frame reset lives
  // in the change handler so the effect body has no synchronous setState)
  useEffect(() => {
    if (scenarioId == null) return;
    apiGet<MonitorPack>(`/monitor/scenario/${scenarioId}`)
      .then(setPack)
      .catch((e) => setError(String(e)));
  }, [scenarioId]);

  const selectScenario = (id: number) => {
    setPlaying(false);
    setFrameIdx(0);
    setPack(null);
    setScenarioId(id);
  };

  // real-time replay loop: advance by out_hz * speed frames per wall second.
  // setInterval (not rAF) so replay keeps running even when the tab is hidden —
  // a monitoring view should not freeze in the background.
  useEffect(() => {
    if (!playing || !pack) return;
    let last = performance.now();
    const id = setInterval(() => {
      const now = performance.now();
      const dt = (now - last) / 1000;
      last = now;
      setFrameIdx((i) => {
        const next = i + dt * pack.meta.out_hz * speed;
        if (next >= pack.frames.length - 1) {
          setPlaying(false);
          return pack.frames.length - 1;
        }
        return next;
      });
    }, 66); // ~15 fps when visible
    return () => clearInterval(id);
  }, [playing, pack, speed]);

  if (error) return <Note tone="danger">載入失敗：{error}</Note>;
  if (!index) return <p className="text-sm text-muted-foreground">載入中…</p>;
  if (!index.available)
    return <Note tone="warn">{index.reason ?? "回放包尚未建立。"}</Note>;

  const fi = Math.min(Math.round(frameIdx), (pack?.frames.length ?? 1) - 1);
  const frame: MonitorFrame | null = pack ? pack.frames[fi] : null;
  const stageInt = frame ? (frame.stage_int as number) : 0;
  const stage = STAGE[stageInt] ?? STAGE[0];
  const subs = index.subsystems;

  const radarData = frame
    ? subs.map((s) => ({ label: SUB_ZH[s] ?? s, value: (frame[s] as number) ?? 0 }))
    : [];

  const onPlayPause = () => {
    if (pack && fi >= pack.frames.length - 1) setFrameIdx(0);
    setPlaying((p) => !p);
  };

  return (
    <div className="space-y-6">
      {/* controls */}
      <Card>
        <div className="flex flex-wrap items-center gap-3">
          <select
            className="rounded-md border border-border/70 bg-background px-3 py-1.5 text-sm"
            value={scenarioId ?? ""}
            onChange={(e) => selectScenario(Number(e.target.value))}
          >
            {index.scenarios.map((s) => (
              <option key={s.scenario_id} value={s.scenario_id}>
                {String(s.scenario_id).padStart(2, "0")} · {s.scenario_name}
                {s.held_out ? " ⟨留出⟩" : ""}
              </option>
            ))}
          </select>

          <button
            onClick={onPlayPause}
            disabled={!pack}
            className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {playing ? "⏸ 暫停" : "▶ 播放"}
          </button>

          <div className="flex items-center gap-1 text-sm">
            <span className="text-muted-foreground">速度</span>
            {SPEEDS.map((s) => (
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
          </div>

          <div className="ml-auto flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">t</span>
            <input
              type="range"
              min={0}
              max={(pack?.frames.length ?? 1) - 1}
              value={fi}
              onChange={(e) => {
                setPlaying(false);
                setFrameIdx(Number(e.target.value));
              }}
              className="w-48 accent-primary"
              disabled={!pack}
            />
            <span className="tabular-nums text-muted-foreground">
              {frame ? (frame.t as number).toFixed(2) : "0.00"}s
            </span>
          </div>
        </div>
      </Card>

      {!pack ? (
        <p className="text-sm text-muted-foreground">載入情境…</p>
      ) : (
        <>
          {/* hero: radar + status stack */}
          <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
            <Card title="子系統健康雷達（0=正常，1=嚴重偏離）">
              <div className="h-[340px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData} outerRadius="72%">
                    <PolarGrid stroke="currentColor" className="text-border" />
                    <PolarAngleAxis
                      dataKey="label"
                      tick={{ fontSize: 13, fill: "currentColor" }}
                      className="text-muted-foreground"
                    />
                    {/* Fixed 0-1 radius so the outer ring is always severity 1;
                        otherwise recharts auto-scales to the current max and the
                        shape jumps around / looks full even when healthy. */}
                    <PolarRadiusAxis
                      domain={[0, 1]}
                      tick={false}
                      axisLine={false}
                      tickCount={5}
                    />
                    <Radar
                      dataKey="value"
                      stroke={stage.color}
                      fill={stage.color}
                      fillOpacity={0.35}
                      isAnimationActive={false}
                    />
                    <Tooltip
                      formatter={(v) => [Number(v ?? 0).toFixed(2), "嚴重度"]}
                      contentStyle={{ fontSize: 12 }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
                {subs.map((s) => {
                  const v = (frame?.[s] as number) ?? 0;
                  return (
                    <Bar
                      key={s}
                      label={SUB_ZH[s] ?? s}
                      right={v.toFixed(2)}
                      value={v}
                      colorClass={severityColorClass(v)}
                    />
                  );
                })}
              </div>
            </Card>

            {/* status stack */}
            <div className="space-y-4">
              <Card className="text-center">
                <p className="text-xs text-muted-foreground">目前狀態</p>
                <p className={`mt-1 text-4xl font-black tracking-tight ${stage.tone}`}>
                  {stage.zh}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {pack.meta.scenario_name} · {pack.meta.fault_category}
                </p>
              </Card>

              <div className="grid grid-cols-2 gap-4">
                <Stat
                  label="健康分數"
                  value={frame ? `${(frame.health as number).toFixed(0)}` : "—"}
                  valueClass=""
                  sub="資料內建"
                />
                <Stat
                  label="RUL 剩餘"
                  value={frame ? `${(frame.rul as number).toFixed(1)}s` : "—"}
                  sub="資料內建"
                />
              </div>

              <Card title="AI 早期預警">
                <Bar
                  label="模型預警機率"
                  right={frame ? `${((frame.pred_prob as number) * 100).toFixed(0)}%` : "—"}
                  value={frame ? (frame.pred_prob as number) : 0}
                  colorClass={
                    (frame?.pred_prob as number) >= 0.5 ? "bg-red-500" : "bg-emerald-500"
                  }
                  sub={
                    (frame?.pred_prob as number) >= 0.5
                      ? "⚠ 模型判定即將進入警告"
                      : "運轉正常，未觸發預警"
                  }
                />
                <div className="mt-3 flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">疑似原因</span>
                  <span className="font-medium">{pack.meta.predicted_category}</span>
                </div>
                {pack.meta.lead_to_alarm_s != null && (
                  <div className="mt-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-700 dark:text-emerald-300">
                    模型比真實告警提前{" "}
                    <strong className="tabular-nums">{pack.meta.lead_to_alarm_s}s</strong> 預警
                  </div>
                )}
              </Card>

              {/* big health ring */}
              <Card className="flex items-center justify-center">
                <HealthRing value={frame ? (frame.health as number) : 0} />
              </Card>
            </div>
          </div>

          {/* timeline strip */}
          <Card title="演進時間軸（底色=真實階段；標記=模型示警點 vs 真實告警）">
            <StageTimeline pack={pack} fi={fi} />
          </Card>

          {/* probability overlay */}
          <Card title="故障機率：模型預測 vs 資料內建">
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={pack.frames.map((f) => ({
                    t: f.t,
                    pred: f.pred_prob,
                    gt: f.gt_prob,
                  }))}
                  margin={{ top: 8, right: 16, bottom: 4, left: -8 }}
                >
                  <XAxis
                    dataKey="t"
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    className="text-muted-foreground"
                    tickFormatter={(v) => `${v.toFixed(0)}s`}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    className="text-muted-foreground"
                  />
                  <Tooltip contentStyle={{ fontSize: 12 }} />
                  <ReferenceLine x={frame?.t as number} stroke="#a855f7" strokeWidth={2} />
                  <Line
                    dataKey="pred"
                    name="模型預警"
                    stroke="#ef4444"
                    dot={false}
                    isAnimationActive={false}
                    strokeWidth={2}
                  />
                  <Line
                    dataKey="gt"
                    name="資料內建"
                    stroke="#64748b"
                    dot={false}
                    isAnimationActive={false}
                    strokeDasharray="4 4"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Note tone="info">
            早期預警（留出情境）F1 = {index.eval.early_warning_f1_heldout ?? "—"}；
            中位提前量 = {index.eval.median_lead_to_alarm_s ?? "—"}s（預警至真實告警）。
            合成退化平滑、易分離，故指標偏高——本軌重點為即時監控視覺與提前量概念的呈現，非困難基準。
            「疑似原因」的類別標註為 in-distribution 參考（部分類別僅單一情境），非泛化宣稱。
          </Note>
        </>
      )}
    </div>
  );
}

// --- stage timeline ----------------------------------------------------------
function StageTimeline({ pack, fi }: { pack: MonitorPack; fi: number }) {
  const frames = pack.frames;
  const n = frames.length;
  // contiguous stage segments
  const segs: { stage: number; start: number; end: number }[] = [];
  for (let i = 0; i < n; i++) {
    const s = frames[i].stage_int as number;
    const last = segs[segs.length - 1];
    if (last && last.stage === s) last.end = i;
    else segs.push({ stage: s, start: i, end: i });
  }
  const pctOfT = (t: number | null) =>
    t == null ? null : (t / (frames[n - 1].t as number)) * 100;
  const alertPct = pctOfT(pack.meta.model_alert_t);
  const alarmPct = pctOfT(pack.meta.gt_alarm_t);
  const playPct = (fi / (n - 1)) * 100;

  return (
    <div>
      <div className="relative h-8 w-full overflow-hidden rounded-md">
        <div className="flex h-full w-full">
          {segs.map((seg, i) => (
            <div
              key={i}
              style={{
                width: `${((seg.end - seg.start + 1) / n) * 100}%`,
                backgroundColor: STAGE[seg.stage]?.color ?? "#10b981",
              }}
              className="h-full opacity-80"
            />
          ))}
        </div>
        {/* markers */}
        {alertPct != null && (
          <Marker pct={alertPct} color="#ef4444" label="模型示警" />
        )}
        {alarmPct != null && (
          <Marker pct={alarmPct} color="#7f1d1d" label="真實告警" top />
        )}
        {/* playhead */}
        <div
          className="absolute top-0 h-full w-0.5 bg-violet-500"
          style={{ left: `${playPct}%` }}
        />
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {STAGE.map((s) => (
          <span key={s.key} className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
            {s.zh}
          </span>
        ))}
      </div>
    </div>
  );
}

function Marker({
  pct,
  color,
  label,
  top,
}: {
  pct: number;
  color: string;
  label: string;
  top?: boolean;
}) {
  return (
    <div className="absolute top-0 h-full" style={{ left: `${pct}%` }}>
      <div className="h-full w-0.5" style={{ backgroundColor: color }} />
      <span
        className={`absolute whitespace-nowrap rounded px-1 text-[10px] font-medium text-white ${
          top ? "top-0" : "bottom-0"
        }`}
        style={{ backgroundColor: color, transform: "translateX(-50%)" }}
      >
        {label}
      </span>
    </div>
  );
}
