"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { cn } from "@/lib/utils";

export interface TrendChartProps {
  data: unknown[];
  dataKey: string;
  xKey: string;
  /** CSS colour (e.g. var(--chart-1) or a hex). */
  color?: string;
  height?: number;
  unit?: string;
  className?: string;
}

/** Compact area trend chart with a gradient fill and dark tooltip. */
export function TrendChart({
  data,
  dataKey,
  xKey,
  color = "var(--chart-1)",
  height = 140,
  unit,
  className,
}: TrendChartProps) {
  const gradId = `grad-${dataKey}`;
  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -18 }}>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.35} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="currentColor"
            className="text-border"
            vertical={false}
          />
          <XAxis
            dataKey={xKey}
            tick={{ fontSize: 10, fill: "currentColor" }}
            className="text-muted-foreground"
            tickLine={false}
            axisLine={false}
            minTickGap={24}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "currentColor" }}
            className="text-muted-foreground"
            tickLine={false}
            axisLine={false}
            width={42}
          />
          <Tooltip
            cursor={{ stroke: color, strokeOpacity: 0.4 }}
            contentStyle={{
              background: "var(--popover)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
              color: "var(--popover-foreground)",
            }}
            labelStyle={{ color: "var(--muted-foreground)" }}
            formatter={(value: unknown) =>
              [`${value as string | number}${unit ? ` ${unit}` : ""}`, dataKey] as [
                string,
                string,
              ]
            }
          />
          <Area
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={2}
            fill={`url(#${gradId})`}
            dot={false}
            activeDot={{ r: 3, fill: color }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
