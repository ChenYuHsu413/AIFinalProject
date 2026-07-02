"use client";

import { useMemo, useState } from "react";
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

const WARN_THRESHOLD = 50;
const CRIT_THRESHOLD = 30;

const RANGES = [
  { key: "1H", points: 12, step: 5 }, // 5-min
  { key: "6H", points: 12, step: 30 },
  { key: "24H", points: 12, step: 120 },
  { key: "7D", points: 14, step: 0 },
] as const;

type RangeKey = (typeof RANGES)[number]["key"];

interface TrendPoint {
  t: string;
  avg: number;
  worst: number;
  alerts: number;
}

/** Deterministic series per range (no Date.now / Math.random). */
function seriesFor(range: RangeKey): TrendPoint[] {
  const cfg = RANGES.find((r) => r.key === range)!;
  const seed = { "1H": 1, "6H": 2, "24H": 3, "7D": 4 }[range];
  return Array.from({ length: cfg.points }, (_, i) => {
    const wob = Math.sin((i + seed) / 1.7) * 4 + Math.cos((i + seed) / 2.9) * 3;
    const trend = range === "7D" ? i * 0.9 : i * 0.4;
    const avg = Math.round(70 - trend + wob);
    const worst = Math.max(12, Math.round(avg - 30 - (i % 3) * 2));
    const alerts = Math.max(0, Math.round(2 + Math.sin((i + seed) / 1.3) * 1.6));
    // Elapsed H:MM keeps every label unique — `% 60` alone repeats (24H would
    // render twelve "00'" ticks and duplicate ReferenceDot/React keys).
    const mins = i * cfg.step;
    const t =
      range === "7D"
        ? `D${String(i + 1).padStart(2, "0")}`
        : `${Math.floor(mins / 60)}:${String(mins % 60).padStart(2, "0")}`;
    return { t, avg, worst, alerts };
  });
}

/**
 * Health & Alert Trend — average + worst health on a 0-100 axis with warning
 * (<50) and critical (<30) threshold lines, alert volume as bars on a secondary
 * axis, anomaly markers where the worst unit drops into the critical band, and a
 * 1H / 6H / 24H / 7D range toggle. Data is deterministic mock for the demo.
 */
export function HealthTrendPanel() {
  const [range, setRange] = useState<RangeKey>("7D");
  const data = useMemo(() => seriesFor(range), [range]);
  const anomalies = data.filter((d) => d.worst < CRIT_THRESHOLD);

  return (
    <Card>
      <CardHeader>
        <CardDescription>健康 / 告警趨勢</CardDescription>
        <CardTitle className="text-lg">平均 vs 最低設備健康 · 告警量</CardTitle>
        <CardAction>
          <div className="inline-flex rounded-lg border border-border bg-card/60 p-0.5">
            {RANGES.map((r) => (
              <button
                key={r.key}
                type="button"
                onClick={() => setRange(r.key)}
                className={cn(
                  "rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
                  range === r.key
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {r.key}
              </button>
            ))}
          </div>
        </CardAction>
      </CardHeader>
      <CardContent>
        <div className="h-[280px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <defs>
                <linearGradient id="fillAvgT" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" vertical={false} />
              <XAxis
                dataKey="t"
                tick={{ fontSize: 11, fill: "currentColor" }}
                className="text-muted-foreground"
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                yAxisId="health"
                domain={[0, 100]}
                tick={{ fontSize: 11, fill: "currentColor" }}
                className="text-muted-foreground"
                tickLine={false}
                axisLine={false}
                width={36}
              />
              <YAxis
                yAxisId="alerts"
                orientation="right"
                domain={[0, 6]}
                tick={{ fontSize: 11, fill: "currentColor" }}
                className="text-muted-foreground"
                tickLine={false}
                axisLine={false}
                width={24}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  fontSize: 12,
                  color: "var(--popover-foreground)",
                }}
                labelStyle={{ color: "var(--muted-foreground)" }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />

              {/* threshold lines */}
              <ReferenceLine
                yAxisId="health"
                y={WARN_THRESHOLD}
                stroke="var(--chart-4)"
                strokeDasharray="4 4"
                strokeOpacity={0.7}
                label={{ value: "警戒 50", position: "insideTopLeft", fontSize: 10, fill: "var(--chart-4)" }}
              />
              <ReferenceLine
                yAxisId="health"
                y={CRIT_THRESHOLD}
                stroke="var(--chart-5)"
                strokeDasharray="4 4"
                strokeOpacity={0.8}
                label={{ value: "危險 30", position: "insideBottomLeft", fontSize: 10, fill: "var(--chart-5)" }}
              />

              <Bar
                yAxisId="alerts"
                name="告警數"
                dataKey="alerts"
                fill="var(--chart-2)"
                fillOpacity={0.35}
                barSize={10}
                radius={[2, 2, 0, 0]}
              />
              <Area
                yAxisId="health"
                name="平均健康"
                type="monotone"
                dataKey="avg"
                stroke="var(--chart-1)"
                strokeWidth={2}
                fill="url(#fillAvgT)"
                dot={false}
                activeDot={{ r: 3 }}
              />
              <Line
                yAxisId="health"
                name="最低設備"
                type="monotone"
                dataKey="worst"
                stroke="var(--chart-5)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3 }}
              />

              {/* anomaly markers where worst unit is in the critical band */}
              {anomalies.map((d) => (
                <ReferenceDot
                  key={d.t}
                  yAxisId="health"
                  x={d.t}
                  y={d.worst}
                  r={4}
                  fill="var(--chart-5)"
                  stroke="var(--popover)"
                  strokeWidth={1.5}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-[11px] text-muted-foreground">
          紅點為最低設備跌入危險區（&lt; {CRIT_THRESHOLD}）的時點 · 趨勢為示意資料
        </p>
      </CardContent>
    </Card>
  );
}
