"use client";

import { useEffect, useState } from "react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { ConfusionMatrix } from "@/components/dashboard/ConfusionMatrix";
import { Card, Note, PageTitle } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

interface PaderbornSplit {
  model: string;
  accuracy: number;
  macro_f1: number;
  labels: string[];
  confusion_matrix: number[][];
  n: number;
}
interface PaderbornEval {
  method: string;
  features: string[];
  results: { baseline: PaderbornSplit; artificial_to_real: PaderbornSplit };
  summary: {
    best_model: string;
    baseline_macro_f1: number;
    generalization_macro_f1: number;
    gap: number;
  };
}

export default function ModuleCPage() {
  const [data, setData] = useState<PaderbornEval | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    apiGet<PaderbornEval>("/paderborn/eval").then(setData).catch(() => setErr(true));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 C · 馬達電流診斷（Paderborn）"
        desc="以定子電流（MCSA）+ 振動時域特徵做故障分類（健康 / 外環 / 內環）。頭條為「訓練人工故障、測真實損傷」的人工→真實泛化對照。"
      />

      <Note tone="warn" className="mb-6">
        <b>誠實性：</b>電流為真實 PMSM <b>試驗台</b>訊號（MCSA 主張成立），但屬試驗台、
        <b>非產線伺服馬達</b>；含人工（EDM/雕刻）與真實（加速壽命）兩種損傷，頭條實驗為
        「訓練人工、測真實」，<b>如實呈現泛化落差</b>；屬故障分類<b>非 RUL</b>；為子集 MVP。
      </Note>

      {err && (
        <Note tone="danger" className="mb-6">無法載入 Paderborn 評估，請確認後端已啟動。</Note>
      )}

      {data && (
        <>
          <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Baseline macro-F1"
              value={data.summary.baseline_macro_f1.toFixed(3)}
              valueClassName="text-emerald-400"
              footerMuted="健康+人工故障，訓練/測同分佈"
            />
            <MetricCard
              label="人工→真實 macro-F1"
              value={data.summary.generalization_macro_f1.toFixed(3)}
              valueClassName="text-red-400"
              footerMuted="訓練人工、測真實損傷"
            />
            <MetricCard
              label="泛化落差 gap"
              value={data.summary.gap.toFixed(3)}
              valueClassName="text-amber-400"
              footerMuted="baseline − 真實轉移"
            />
            <MetricCard
              label="最佳模型"
              value={data.summary.best_model}
              footerMuted={`${data.features.length} 維時域特徵`}
            />
          </section>

          <Note tone="info" className="mb-6">
            Baseline 近乎完美（macro-F1 {data.summary.baseline_macro_f1.toFixed(2)}），但換成真實損傷後
            掉到 {data.summary.generalization_macro_f1.toFixed(2)} —— 這正是「人工故障訓練無法直接泛化到
            真實劣化」的關鍵發現，不能只報 baseline。
          </Note>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title={`Baseline 混淆矩陣（${data.results.baseline.model}）`}>
              <ConfusionMatrix
                labels={data.results.baseline.labels}
                matrix={data.results.baseline.confusion_matrix}
              />
            </Card>
            <Card title="人工→真實 混淆矩陣">
              <ConfusionMatrix
                labels={data.results.artificial_to_real.labels}
                matrix={data.results.artificial_to_real.confusion_matrix}
              />
              <p className="mt-2 text-xs text-amber-300">
                真實損傷幾乎全被誤判 —— 印證泛化落差。
              </p>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
