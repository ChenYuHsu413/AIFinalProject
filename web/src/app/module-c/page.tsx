"use client";

import { useEffect, useState } from "react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { ConfusionMatrix } from "@/components/dashboard/ConfusionMatrix";
import { Bar, Card, Note, PageTitle } from "@/components/ui-kit";
import { apiGet, apiPost } from "@/lib/api";

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
interface Sample {
  bearing_code: string;
  condition: string;
  measurement: number;
  fault_class: string;
  damage_origin: string;
  features: Record<string, number>;
}
interface PredictOut {
  predicted_class: string;
  labels: string[];
  proba: Record<string, number>;
  confidence: number;
}

const ORIGIN_ZH: Record<string, string> = {
  healthy: "健康",
  artificial: "人工",
  real: "真實",
};
const CLASS_ZH: Record<string, string> = {
  healthy: "健康",
  outer: "外環故障",
  inner: "內環故障",
};

export default function ModuleCPage() {
  const [data, setData] = useState<PaderbornEval | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [err, setErr] = useState(false);

  useEffect(() => {
    apiGet<PaderbornEval>("/paderborn/eval").then(setData).catch(() => setErr(true));
    apiGet<Sample[]>("/paderborn/samples").then(setSamples).catch(() => {});
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

      {samples.length > 0 && <LiveInference samples={samples} />}
    </div>
  );
}

// ===========================================================================
// Live inference — pick a measurement, run the trained classifier server-side.
// Picking a REAL-damage measurement shows the artificial->real gap live.
// ===========================================================================
function LiveInference({ samples }: { samples: Sample[] }) {
  const [idx, setIdx] = useState(0);
  const [out, setOut] = useState<PredictOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(false);

  const s = samples[idx];

  // Async fetch only — resets live in the change handler so the effect body has
  // no synchronous setState (cascading-render lint rule).
  useEffect(() => {
    let cancelled = false;
    apiPost<PredictOut>("/paderborn/predict", { features: s.features })
      .then((o) => {
        if (!cancelled) setOut(o);
      })
      .catch(() => {
        if (!cancelled) setErr(true);
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [s]);

  const selectSample = (i: number) => {
    setBusy(true);
    setErr(false);
    setIdx(i);
  };

  const correct = out ? out.predicted_class === s.fault_class : null;
  const isReal = s.damage_origin === "real";

  return (
    <section className="mt-8">
      <h2 className="mb-1 text-lg font-semibold">即時推論（選一筆量測，模型即時分類）</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        以已訓練的分類器（健康 + 人工故障訓練）即時推論。選一筆<b>真實損傷</b>量測，常會看到模型誤判 ——
        這就是上方泛化落差在單筆量測上的現場呈現。
      </p>

      <Card>
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <select
            value={idx}
            onChange={(e) => selectSample(Number(e.target.value))}
            className="rounded-md border border-border bg-card px-2 py-1.5 text-sm"
          >
            {samples.map((x, i) => (
              <option key={`${x.bearing_code}-${x.measurement}`} value={i}>
                {x.bearing_code} · {ORIGIN_ZH[x.damage_origin] ?? x.damage_origin}損傷 · 真實標籤{" "}
                {CLASS_ZH[x.fault_class] ?? x.fault_class}
              </option>
            ))}
          </select>
          {isReal && (
            <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] text-amber-300 ring-1 ring-inset ring-amber-500/30">
              真實損傷（訓練未見過此分佈）
            </span>
          )}
        </div>

        {err && <Note tone="danger">推論失敗，請確認後端與模型已就緒。</Note>}

        {out && (
          <div className={busy ? "opacity-60 transition-opacity" : "transition-opacity"}>
            <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-xl border border-border/70 bg-card/70 p-4">
                <p className="text-xs text-muted-foreground">真實標籤</p>
                <p className="mt-1 text-lg font-bold">{CLASS_ZH[s.fault_class] ?? s.fault_class}</p>
              </div>
              <div className="rounded-xl border border-border/70 bg-card/70 p-4">
                <p className="text-xs text-muted-foreground">模型預測</p>
                <p
                  className={`mt-1 text-lg font-bold ${
                    correct ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {CLASS_ZH[out.predicted_class] ?? out.predicted_class} {correct ? "✓" : "✗"}
                </p>
              </div>
              <div className="rounded-xl border border-border/70 bg-card/70 p-4">
                <p className="text-xs text-muted-foreground">信心度</p>
                <p className="mt-1 text-lg font-bold text-cyan-400">
                  {(out.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="space-y-2">
              {out.labels.map((c) => (
                <Bar
                  key={c}
                  label={CLASS_ZH[c] ?? c}
                  right={`${((out.proba[c] ?? 0) * 100).toFixed(1)}%`}
                  value={out.proba[c] ?? 0}
                  colorClass={c === out.predicted_class ? "bg-cyan-400" : "bg-muted-foreground/40"}
                />
              ))}
            </div>

            {!correct && isReal && (
              <Note tone="warn" className="mt-4">
                模型把真實 {CLASS_ZH[s.fault_class] ?? s.fault_class} 誤判為{" "}
                {CLASS_ZH[out.predicted_class] ?? out.predicted_class} —— 人工故障訊號與真實疲勞劣化的分佈差異，
                正是「不能只報 baseline」的理由。
              </Note>
            )}
          </div>
        )}
      </Card>
    </section>
  );
}
