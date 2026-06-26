"use client";

import type { TelemetryPoint } from "@/lib/mock";

import { TrendChart } from "./TrendChart";

const CHANNELS: {
  key: keyof TelemetryPoint & string;
  label: string;
  unit: string;
  color: string;
}[] = [
  { key: "torque", label: "扭矩 Torque", unit: "Nm", color: "var(--chart-1)" },
  { key: "rotor_speed", label: "轉速 Rotor Speed", unit: "rpm", color: "var(--chart-2)" },
  { key: "position_error", label: "位置誤差 Position Error", unit: "mm", color: "var(--chart-4)" },
  { key: "current", label: "電流 Current", unit: "A", color: "var(--chart-3)" },
];

/** 2×2 grid of telemetry trend charts for one unit's recent window. */
export function TelemetryTrends({ data }: { data: TelemetryPoint[] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {CHANNELS.map((ch) => (
        <div
          key={ch.key}
          className="rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm"
        >
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-semibold">{ch.label}</span>
            <span className="text-[11px] text-muted-foreground">{ch.unit}</span>
          </div>
          <TrendChart
            data={data}
            dataKey={ch.key}
            xKey="t"
            color={ch.color}
            unit={ch.unit}
          />
        </div>
      ))}
    </div>
  );
}
