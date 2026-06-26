"use client";

import { useEffect, useState } from "react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { TrendChart } from "@/components/dashboard/TrendChart";
import { Note, PageTitle } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

interface CurvePoint {
  timestamp: string;
  rul_true: number;
  rul_pred: number | null;
  health: number;
  is_degrading: boolean;
}
interface ImsMetrics {
  metrics: { mae_hours: number; rmse_hours: number; r2: number };
  near_failure_metrics: { mae_hours: number; r2: number };
  lead_time_days: number;
}

export default function ModuleBRulPage() {
  const [curve, setCurve] = useState<CurvePoint[]>([]);
  const [m, setM] = useState<ImsMetrics | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    Promise.all([
      apiGet<CurvePoint[]>("/ims/health_curve"),
      apiGet<ImsMetrics>("/ims/metrics"),
    ])
      .then(([c, mm]) => {
        setCurve(c);
        setM(mm);
      })
      .catch(() => setErr(true));
  }, []);

  // downsample to ~120 points for the chart
  const step = Math.max(1, Math.floor(curve.length / 120));
  const data = curve
    .filter((_, i) => i % step === 0)
    .map((p) => ({ t: p.timestamp.slice(5, 16), rul: +p.rul_true.toFixed(1) }));

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 B · RUL 預測（IMS）"
        desc="以趨勢外推估剩餘壽命（RUL）。真實 RUL 隨運轉線性下降至失效。"
      />
      <Note tone="warn" className="mb-6">
        本模組採<b>趨勢外推</b>而非逐點深度回歸（單軌跡資料不做深度 RUL 回歸）；
        近失效段誤差另列，因外推牆效應。
      </Note>
      {err && <Note tone="danger" className="mb-6">無法載入 RUL 曲線，請確認後端已啟動。</Note>}

      {m && (
        <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="整體 MAE（小時）" value={m.metrics.mae_hours.toFixed(1)} valueClassName="text-cyan-400" />
          <MetricCard label="整體 R²" value={m.metrics.r2.toFixed(3)} valueClassName="text-emerald-400" />
          <MetricCard label="近失效 MAE（小時）" value={m.near_failure_metrics.mae_hours.toFixed(1)} valueClassName="text-amber-400" footerMuted="外推牆效應" />
          <MetricCard label="提前預警" value={m.lead_time_days.toFixed(1)} unit="天" />
        </section>
      )}

      {data.length > 0 && (
        <div className="rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
          <span className="text-sm font-semibold">真實剩餘壽命 RUL（小時）隨時間</span>
          <TrendChart data={data} dataKey="rul" xKey="t" unit="h" height={280} color="var(--chart-2)" />
        </div>
      )}
    </div>
  );
}
