"use client";

import { useEffect, useMemo, useState } from "react";

import { ConfusionMatrix } from "@/components/dashboard/ConfusionMatrix";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Card, Note, PageTitle, Stat } from "@/components/ui-kit";
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
interface TestPreds {
  y_true: number[];
  y_proba: number[];
}

export default function ModuleAEvaluationPage() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [rows, setRows] = useState<MetricRow[]>([]);
  const [ftRows, setFtRows] = useState<FailureRow[]>([]);
  const [tp, setTp] = useState<TestPreds | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    Promise.all([
      apiGet<ModelInfo>("/model_info"),
      apiGet<{ rows: MetricRow[] }>("/metrics"),
      apiGet<{ rows: FailureRow[] }>("/failure_type_metrics"),
      apiGet<TestPreds>("/metrics/test_predictions"),
    ])
      .then(([i, m, f, t]) => {
        setInfo(i);
        setRows([...m.rows].sort((a, b) => b.roc_auc - a.roc_auc).slice(0, 8));
        setFtRows(f.rows);
        setTp(t);
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

      {tp && tp.y_true.length > 0 && <ThresholdTuner data={tp} />}

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

// ===========================================================================
// Interactive decision-threshold tuner (recomputes the confusion matrix +
// precision/recall/F1 client-side from the held-out test probabilities)
// ===========================================================================
function ThresholdTuner({ data }: { data: TestPreds }) {
  const [thr, setThr] = useState(0.5);
  const total = data.y_true.length;
  const nPos = useMemo(() => data.y_true.reduce((s, v) => s + v, 0), [data]);

  const m = useMemo(() => {
    let tn = 0,
      fp = 0,
      fn = 0,
      tpos = 0;
    for (let i = 0; i < total; i++) {
      const pred = data.y_proba[i] >= thr ? 1 : 0;
      if (data.y_true[i] === 1) {
        if (pred === 1) tpos++;
        else fn++;
      } else if (pred === 1) {
        fp++;
      } else {
        tn++;
      }
    }
    const precision = tpos + fp ? tpos / (tpos + fp) : 0;
    const recall = tpos + fn ? tpos / (tpos + fn) : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { tn, fp, fn, tpos, precision, recall, f1 };
  }, [data, thr, total]);

  return (
    <Card title="互動式門檻調節器（即時重算）" className="mb-6">
      <Note tone="info" className="mb-4">
        測試集共 <b>{total}</b> 筆，其中故障樣本 <b>{nPos}</b> 筆（{((nPos / total) * 100).toFixed(2)}%）。
        往左拉門檻 → 預測為故障的樣本變多 → Recall 上升、漏報 FN 下降，但誤報 FP 通常上升。
        預測性維護情境通常偏好較低門檻以提高 Recall。
      </Note>

      <label className="block">
        <span className="text-xs text-muted-foreground">
          決策門檻 threshold（機率 ≥ threshold ⇒ 預測為故障）：
          <b className="ml-1 text-foreground tabular-nums">{thr.toFixed(2)}</b>
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={thr}
          onChange={(e) => setThr(Number(e.target.value))}
          className="mt-2 w-full accent-primary"
        />
      </label>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Precision" value={m.precision.toFixed(3)} sub="誤報越少越高" valueClass="text-cyan-400" />
        <Stat label="Recall" value={m.recall.toFixed(3)} sub="漏報越少越高" valueClass="text-emerald-400" />
        <Stat label="F1" value={m.f1.toFixed(3)} sub="兩者調和" valueClass="text-amber-400" />
        <Stat label="漏報 FN" value={String(m.fn)} sub={`誤報 FP = ${m.fp}`} valueClass="text-red-400" />
      </div>

      <div className="mt-4">
        <ConfusionMatrix
          labels={["健康", "故障"]}
          matrix={[
            [m.tn, m.fp],
            [m.fn, m.tpos],
          ]}
        />
      </div>
    </Card>
  );
}
