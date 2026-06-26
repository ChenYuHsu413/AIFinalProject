"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";
import { FLEET_HEALTH_HISTORY } from "@/lib/mock";

/** Hero area chart — fleet average vs worst-unit health over recent shifts. */
export function FleetHealthChart() {
  return (
    <Card className="@container/card">
      <CardHeader>
        <CardDescription>機群健康趨勢</CardDescription>
        <CardTitle className="text-lg">平均 vs 最低設備健康分數</CardTitle>
        <CardAction>
          <Badge variant="outline">近 14 班次</Badge>
        </CardAction>
      </CardHeader>
      <CardContent>
        <div className="h-[260px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={FLEET_HEALTH_HISTORY}
              margin={{ top: 8, right: 8, bottom: 0, left: -16 }}
            >
              <defs>
                <linearGradient id="fillAvg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="fillWorst" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--chart-5)" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="var(--chart-5)" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="currentColor"
                className="text-border"
                vertical={false}
              />
              <XAxis
                dataKey="t"
                tick={{ fontSize: 11, fill: "currentColor" }}
                className="text-muted-foreground"
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 11, fill: "currentColor" }}
                className="text-muted-foreground"
                tickLine={false}
                axisLine={false}
                width={36}
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
              <Legend
                wrapperStyle={{ fontSize: 12 }}
                iconType="plainline"
              />
              <Area
                name="平均健康"
                type="monotone"
                dataKey="avg"
                stroke="var(--chart-1)"
                strokeWidth={2}
                fill="url(#fillAvg)"
                dot={false}
                activeDot={{ r: 3 }}
              />
              <Area
                name="最低設備"
                type="monotone"
                dataKey="worst"
                stroke="var(--chart-5)"
                strokeWidth={2}
                fill="url(#fillWorst)"
                dot={false}
                activeDot={{ r: 3 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
