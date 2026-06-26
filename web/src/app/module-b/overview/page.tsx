"use client";

import { useEffect, useState } from "react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { TrendChart } from "@/components/dashboard/TrendChart";
import { Note, PageTitle } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

interface ImsMetrics {
  indicator: string;
  metrics: { mae_hours: number; rmse_hours: number; r2: number; n_eval: number };
  fpt_time: string;
  lead_time_days: number;
}
interface ImsHi {
  available: boolean;
  indicator: string;
  fpt_index: number;
  alarm_health: number;
  points: { timestamp: string; health: number }[];
}

export default function ModuleBOverviewPage() {
  const [m, setM] = useState<ImsMetrics | null>(null);
  const [hi, setHi] = useState<ImsHi | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    Promise.all([
      apiGet<ImsMetrics>("/ims/metrics"),
      apiGet<ImsHi>("/ims/health_indicator"),
    ])
      .then(([mm, hh]) => {
        setM(mm);
        setHi(hh);
      })
      .catch(() => setErr(true));
  }, []);

  const curve =
    hi?.points.map((p, i) => ({ t: p.timestamp.slice(5, 16), health: p.health, i })) ?? [];

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 B · 動態健康度（IMS）"
        desc="以振動趨勢指標建健康曲線、偵測退化起點（FPT）並趨勢外推 RUL。"
      />
      <Note tone="warn" className="mb-6">
        <b>誠實性：</b>IMS Set 2 為<b>單軌跡</b>資料，結果<b>不可泛化</b>到其他軸承／馬達；
        不在單軌跡上做深度 RUL 回歸（會撞外推牆）。
      </Note>
      {err && <Note tone="danger" className="mb-6">無法載入 IMS 結果，請確認後端已啟動。</Note>}

      {m && (
        <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="MAE（小時）" value={m.metrics.mae_hours.toFixed(1)} valueClassName="text-cyan-400" footerMuted="趨勢外推 RUL 誤差" />
          <MetricCard label="RMSE（小時）" value={m.metrics.rmse_hours.toFixed(1)} />
          <MetricCard label="R²" value={m.metrics.r2.toFixed(3)} valueClassName="text-emerald-400" />
          <MetricCard label="提前預警" value={m.lead_time_days.toFixed(1)} unit="天" valueClassName="text-amber-400" footerMuted={`FPT ${m.fpt_time.slice(0, 16)}`} />
        </section>
      )}

      {hi && hi.available && (
        <div className="rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-sm font-semibold">健康指標曲線（{hi.indicator}）</span>
            <span className="text-[11px] text-muted-foreground">
              FPT @ #{hi.fpt_index} · 告警線 {hi.alarm_health}
            </span>
          </div>
          <TrendChart data={curve} dataKey="health" xKey="t" height={280} color="var(--chart-3)" />
          <p className="mt-2 text-xs text-muted-foreground">
            健康分數隨運轉下降；越過 FPT 後退化加速，提供約 {m?.lead_time_days.toFixed(1)} 天的預警餘裕。
          </p>
        </div>
      )}
    </div>
  );
}
