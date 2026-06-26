"use client";

import { useEffect, useState } from "react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { Card, Note, PageTitle } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

interface ModelInfo {
  model_name: string;
  feature_set: string;
  feature_columns: string[];
  metrics: { accuracy: number; precision: number; recall: number; f1: number; roc_auc: number; pr_auc: number };
}
interface MetricRow {
  model_name: string;
  feature_set: string;
  feature_count: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  recall: number;
}
interface FailureRow {
  failure_type: string;
  n_positives_test: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export default function ModuleAEvaluationPage() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [rows, setRows] = useState<MetricRow[]>([]);
  const [ftRows, setFtRows] = useState<FailureRow[]>([]);
  const [err, setErr] = useState(false);

  useEffect(() => {
    Promise.all([
      apiGet<ModelInfo>("/model_info"),
      apiGet<{ rows: MetricRow[] }>("/metrics"),
      apiGet<{ rows: FailureRow[] }>("/failure_type_metrics"),
    ])
      .then(([i, m, f]) => {
        setInfo(i);
        setRows([...m.rows].sort((a, b) => b.roc_auc - a.roc_auc).slice(0, 8));
        setFtRows(f.rows);
      })
      .catch(() => setErr(true));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 A · 模型評估結果"
        desc="最佳模型指標、各模型／特徵組比較，以及故障類型第二階段表現（AI4I 2020 合成資料）。"
      />
      {err && <Note tone="danger" className="mb-6">無法載入評估結果，請確認後端已啟動。</Note>}

      {info && (
        <>
          <p className="mb-3 text-sm text-muted-foreground">
            最佳模型：<b className="text-foreground">{info.model_name}</b>（特徵組 {info.feature_set}，
            {info.feature_columns.length} 維）
          </p>
          <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Accuracy" value={info.metrics.accuracy.toFixed(3)} />
            <MetricCard label="F1" value={info.metrics.f1.toFixed(3)} valueClassName="text-cyan-400" footerMuted={`Recall ${info.metrics.recall.toFixed(3)}`} />
            <MetricCard label="ROC-AUC" value={info.metrics.roc_auc.toFixed(3)} valueClassName="text-emerald-400" />
            <MetricCard label="PR-AUC" value={info.metrics.pr_auc.toFixed(3)} valueClassName="text-amber-400" />
          </section>
        </>
      )}

      <Card title="模型 / 特徵組比較（依 ROC-AUC 前 8）" className="mb-6">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="py-2 pr-4 font-medium">模型</th>
                <th className="py-2 pr-4 font-medium">特徵組</th>
                <th className="py-2 pr-4 font-medium">F1</th>
                <th className="py-2 pr-4 font-medium">Recall</th>
                <th className="py-2 pr-4 font-medium">ROC-AUC</th>
                <th className="py-2 font-medium">PR-AUC</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-b border-border/40 last:border-0 hover:bg-muted/30">
                  <td className="py-2 pr-4 font-medium">{r.model_name}</td>
                  <td className="py-2 pr-4 text-muted-foreground">{r.feature_set}（{r.feature_count}）</td>
                  <td className="py-2 pr-4 tabular-nums">{r.f1.toFixed(3)}</td>
                  <td className="py-2 pr-4 tabular-nums">{r.recall.toFixed(3)}</td>
                  <td className="py-2 pr-4 tabular-nums text-emerald-400">{r.roc_auc.toFixed(3)}</td>
                  <td className="py-2 tabular-nums">{r.pr_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="故障類型第二階段（罕見類別 PR-AUC 偏低為已知限制）">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="py-2 pr-4 font-medium">故障類型</th>
                <th className="py-2 pr-4 font-medium">測試正樣本</th>
                <th className="py-2 pr-4 font-medium">Recall</th>
                <th className="py-2 pr-4 font-medium">F1</th>
                <th className="py-2 font-medium">ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {ftRows.map((r) => (
                <tr key={r.failure_type} className="border-b border-border/40 last:border-0 hover:bg-muted/30">
                  <td className="py-2 pr-4 font-mono font-medium">{r.failure_type}</td>
                  <td className="py-2 pr-4 tabular-nums">{r.n_positives_test}</td>
                  <td className="py-2 pr-4 tabular-nums">{r.recall.toFixed(3)}</td>
                  <td className="py-2 pr-4 tabular-nums">{r.f1.toFixed(3)}</td>
                  <td className="py-2 tabular-nums">{r.roc_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
