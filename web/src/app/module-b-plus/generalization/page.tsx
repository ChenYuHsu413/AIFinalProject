"use client";

import { useEffect, useState } from "react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { Card, Note, PageTitle } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

interface PerBearing {
  condition: string;
  bearing: string;
  life_hours: number;
  lead_time_hours: number;
  mae_hours: number;
}
interface XjtuGen {
  per_bearing: PerBearing[];
  aggregate: {
    n_bearings: number;
    mean_lead_time_hours: number;
    mean_mae_hours: number;
    by_condition: Record<string, { n_bearings: number; mean_lead_time_hours: number; mean_mae_hours: number }>;
  };
}
interface DomainAdapt {
  summary: { baseline_r2: number; best_method: string; best_r2: number; ranking: [string, number][] };
}

export default function ModuleBPlusGenPage() {
  const [gen, setGen] = useState<XjtuGen | null>(null);
  const [da, setDa] = useState<DomainAdapt | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    Promise.all([
      apiGet<XjtuGen>("/xjtu/generalization"),
      apiGet<DomainAdapt>("/xjtu/domain_adapt"),
    ])
      .then(([g, d]) => {
        setGen(g);
        setDa(d);
      })
      .catch(() => setErr(true));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 B+ · 多軌跡泛化（XJTU-SY）"
        desc="以固定參數的趨勢外推，跨 15 顆軸承 / 3 種工況評估提前預警與 RUL 誤差的泛化能力。"
      />
      <Note tone="info" className="mb-6">
        相對 IMS 單軌跡，B+ 以<b>多軸承 / 多工況</b>檢驗泛化；不同工況的可預警程度差異甚大（如下）。
      </Note>
      {err && <Note tone="danger" className="mb-6">無法載入 XJTU 結果，請確認後端已啟動。</Note>}

      {gen && (
        <>
          <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-3">
            <MetricCard label="軸承數" value={gen.aggregate.n_bearings} unit="顆" footerMuted="15 軸承 / 3 工況" />
            <MetricCard label="平均提前預警" value={gen.aggregate.mean_lead_time_hours.toFixed(1)} unit="時" valueClassName="text-emerald-400" />
            <MetricCard label="平均 MAE" value={gen.aggregate.mean_mae_hours.toFixed(2)} unit="時" valueClassName="text-amber-400" />
          </section>

          <Card title="各工況彙整" className="mb-6">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[480px] text-sm">
                <thead>
                  <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <th className="py-2 pr-4 font-medium">工況</th>
                    <th className="py-2 pr-4 font-medium">軸承數</th>
                    <th className="py-2 pr-4 font-medium">平均提前預警（時）</th>
                    <th className="py-2 font-medium">平均 MAE（時）</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(gen.aggregate.by_condition).map(([cond, v]) => (
                    <tr key={cond} className="border-b border-border/40 last:border-0 hover:bg-muted/30">
                      <td className="py-2 pr-4 font-mono font-medium">{cond}</td>
                      <td className="py-2 pr-4 tabular-nums">{v.n_bearings}</td>
                      <td className="py-2 pr-4 tabular-nums text-emerald-400">{v.mean_lead_time_hours.toFixed(2)}</td>
                      <td className="py-2 tabular-nums">{v.mean_mae_hours.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="各軸承（15 顆）" className="mb-6">
            <div className="max-h-72 overflow-y-auto">
              <table className="w-full min-w-[520px] text-sm">
                <thead className="sticky top-0 bg-card">
                  <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <th className="py-2 pr-4 font-medium">軸承</th>
                    <th className="py-2 pr-4 font-medium">工況</th>
                    <th className="py-2 pr-4 font-medium">壽命（時）</th>
                    <th className="py-2 pr-4 font-medium">提前預警（時）</th>
                    <th className="py-2 font-medium">MAE（時）</th>
                  </tr>
                </thead>
                <tbody>
                  {gen.per_bearing.map((b) => (
                    <tr key={b.bearing} className="border-b border-border/40 last:border-0 hover:bg-muted/30">
                      <td className="py-2 pr-4 font-mono font-medium">{b.bearing}</td>
                      <td className="py-2 pr-4 text-muted-foreground">{b.condition}</td>
                      <td className="py-2 pr-4 tabular-nums">{b.life_hours.toFixed(1)}</td>
                      <td className="py-2 pr-4 tabular-nums">{b.lead_time_hours.toFixed(2)}</td>
                      <td className="py-2 tabular-nums">{b.mae_hours.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {da && (
        <Card title="領域自適應（跨工況 RUL）方法比較">
          <p className="mb-3 text-sm text-muted-foreground">
            Baseline R² <b className="text-foreground">{da.summary.baseline_r2.toFixed(3)}</b>；
            最佳方法 <b className="text-emerald-400">{da.summary.best_method}</b>（R² {da.summary.best_r2.toFixed(3)}）。
          </p>
          <div className="space-y-1.5">
            {da.summary.ranking.map(([name, val], i) => (
              <div key={i} className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/20 px-3 py-1.5 text-sm">
                <span className="font-mono">{name}</span>
                <span className="tabular-nums text-muted-foreground">
                  {typeof val === "number" ? `R² ${val.toFixed(3)}` : String(val)}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
