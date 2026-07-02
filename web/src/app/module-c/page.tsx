"use client";

import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

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
interface AdaptMethod {
  model: string;
  accuracy: number;
  macro_f1: number;
  binary_f1: number;
}
interface FewShotPoint {
  k_per_class: number;
  macro_f1_mean: number;
  macro_f1_std: number;
  binary_f1_mean: number;
  binary_f1_std: number;
  n_test_mean: number;
}
interface FeatureDiag {
  feature: string;
  importance: number;
  shift: number;
}
interface DomainAdaptData {
  results: {
    baseline: AdaptMethod;
    coral?: AdaptMethod;
    transductive_zscore?: AdaptMethod;
    few_shot?: {
      model: string;
      seeds: number;
      curve: FewShotPoint[];
      curve_coral?: FewShotPoint[];
    };
    diagnosis?: {
      per_feature: FeatureDiag[];
      spearman_importance_vs_shift: number;
      top_discriminative: FeatureDiag[];
    };
  };
  summary: {
    baseline_macro_f1: number;
    best_unsup_method: string | null;
    best_unsup_macro_f1: number | null;
    few_shot_best_k: number | null;
    few_shot_best_macro_f1: number | null;
    spearman_importance_vs_shift: number | null;
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
  const [adapt, setAdapt] = useState<DomainAdaptData | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [err, setErr] = useState(false);

  useEffect(() => {
    apiGet<PaderbornEval>("/paderborn/eval").then(setData).catch(() => setErr(true));
    apiGet<DomainAdaptData>("/paderborn/domain_adapt")
      .then((d) => setAdapt(d.results ? d : null))
      .catch(() => {});
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
              <p className="mt-2 text-xs text-amber-600 dark:text-amber-300">
                真實損傷幾乎全被誤判 —— 印證泛化落差。
              </p>
            </Card>
          </div>
        </>
      )}

      {adapt && <DomainAdapt data={adapt} />}

      {samples.length > 0 && <LiveInference samples={samples} />}
    </div>
  );
}

// ===========================================================================
// CE1 domain adaptation — can we close the artificial->real gap?
// Unsupervised feature alignment (CORAL / transductive z-score) uses target
// features only; few-shot admits a few REAL labels (disclosed) and traces a
// learning curve.  Honest finding: affine alignment doesn't help, a handful of
// real labels does.
// ===========================================================================
const METHOD_ZH: Record<string, string> = {
  baseline: "無自適應 (baseline)",
  coral: "CORAL 協方差對齊",
  transductive_zscore: "工況感知標準化",
};

function DomainAdapt({ data }: { data: DomainAdaptData }) {
  const { results, summary } = data;
  const unsup: { key: string; m: AdaptMethod }[] = [
    { key: "baseline", m: results.baseline },
    ...(results.coral ? [{ key: "coral", m: results.coral }] : []),
    ...(results.transductive_zscore
      ? [{ key: "transductive_zscore", m: results.transductive_zscore }]
      : []),
  ];
  const curve = results.few_shot?.curve ?? [];
  const curveCoral = results.few_shot?.curve_coral ?? [];
  // merge plain + CORAL few-shot series by k for a two-line chart
  const fsData = curve.map((p) => ({
    k: p.k_per_class,
    fs: p.macro_f1_mean,
    fsCoral: curveCoral.find((c) => c.k_per_class === p.k_per_class)?.macro_f1_mean ?? null,
  }));
  const diag = results.diagnosis;
  const bestUnsupBeatsBaseline =
    summary.best_unsup_macro_f1 != null &&
    summary.best_unsup_macro_f1 > summary.baseline_macro_f1 + 1e-9;

  return (
    <section className="mt-10">
      <h2 className="mb-1 text-lg font-semibold">CE1 · 領域自適應：能不能修補人工→真實落差？</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        在<b>同一</b>「人工→真實」切分上試三種補救：兩種<b>無監督</b>特徵對齊（僅用目標未標註特徵）
        與一種 <b>few-shot</b>（納入少量真實標籤）。
      </p>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="無監督特徵對齊 vs baseline（macro-F1）">
          <div className="space-y-3">
            {unsup.map(({ key, m }) => (
              <div key={key}>
                <Bar
                  label={METHOD_ZH[key] ?? key}
                  right={`${m.macro_f1.toFixed(3)}`}
                  value={m.macro_f1}
                  colorClass={key === "baseline" ? "bg-muted-foreground/50" : "bg-cyan-400"}
                />
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  outer/inner 二類 F1 {m.binary_f1.toFixed(3)}
                  {key !== "baseline" && " · 僅用目標未標註特徵（無監督）"}
                </p>
              </div>
            ))}
          </div>
          <Note tone={bestUnsupBeatsBaseline ? "info" : "warn"} className="mt-4">
            {bestUnsupBeatsBaseline ? (
              <>
                最佳無監督手段（{METHOD_ZH[summary.best_unsup_method ?? ""] ?? summary.best_unsup_method}）
                將 macro-F1 抬到 {summary.best_unsup_macro_f1?.toFixed(3)}。
              </>
            ) : (
              <>
                <b>無監督仿射對齊未能改善</b>（皆 ≤ baseline {summary.baseline_macro_f1.toFixed(3)}）——
                人工 EDM/雕刻故障與真實疲勞劣化的差異<b>不是單純的協方差/平移位移</b>，線性對齊修不動。
              </>
            )}
          </Note>
        </Card>

        <Card title="Few-shot 學習曲線（每類 k 筆真實標籤）">
          <div className="h-[240px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={fsData} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="currentColor"
                  className="text-border"
                  vertical={false}
                />
                <XAxis
                  dataKey="k"
                  tick={{ fontSize: 11, fill: "currentColor" }}
                  className="text-muted-foreground"
                  tickLine={false}
                  axisLine={false}
                  label={{ value: "k / 類", position: "insideBottomRight", fontSize: 10, offset: -2 }}
                />
                <YAxis
                  domain={[0, 1]}
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
                  labelFormatter={(k) => `k=${k} / 類`}
                  formatter={(v, name) => [v == null ? "—" : Number(v).toFixed(3), name as string]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="fs"
                  name="few-shot"
                  stroke="#22d3ee"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="fsCoral"
                  name="CORAL + few-shot"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  strokeDasharray="4 3"
                  dot={{ r: 3 }}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {summary.few_shot_best_k != null && (
            <p className="mt-2 text-xs text-muted-foreground">
              每類 {summary.few_shot_best_k} 筆真實標籤 → macro-F1{" "}
              <span className="font-semibold text-cyan-600 dark:text-cyan-300">
                {summary.few_shot_best_macro_f1?.toFixed(3)}
              </span>
              （baseline {summary.baseline_macro_f1.toFixed(3)}，{results.few_shot?.seeds} 次抽樣平均）。
              <b>先 CORAL 對齊再給標籤反而更差</b> —— 與下方診斷一致：判別軸已被破壞。
            </p>
          )}
        </Card>
      </div>

      {diag && (
        <Card title="機制診斷：判別特徵正是位移最大的（為何線性對齊修不動）" className="mt-6">
          <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
            <div className="h-[260px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 8, right: 12, bottom: 16, left: -8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" />
                  <XAxis
                    type="number"
                    dataKey="importance"
                    name="baseline 重要度"
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    className="text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                    label={{ value: "baseline 重要度 →", position: "insideBottom", fontSize: 10, offset: -8 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="shift"
                    name="人工→真實位移"
                    tick={{ fontSize: 11, fill: "currentColor" }}
                    className="text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                    width={40}
                    label={{ value: "位移 ↑", position: "insideLeft", angle: -90, fontSize: 10 }}
                  />
                  <ZAxis range={[50, 50]} />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    contentStyle={{
                      background: "var(--popover)",
                      border: "1px solid var(--border)",
                      borderRadius: 8,
                      fontSize: 12,
                      color: "var(--popover-foreground)",
                    }}
                    formatter={(v, name) => [Number(v).toFixed(3), name as string]}
                    labelFormatter={() => ""}
                  />
                  <Scatter data={diag.per_feature} fill="#22d3ee" fillOpacity={0.7} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div>
              <p className="text-sm">
                Spearman（重要度, 位移）={" "}
                <span className="font-semibold text-amber-600 dark:text-amber-300">
                  {diag.spearman_importance_vs_shift.toFixed(2)}
                </span>
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                正相關＝越被 baseline 倚重的判別特徵、artificial→real 位移越大。判別軸本身被破壞，
                故線性協方差對齊（CORAL）救不回——只有真實標籤（few-shot）能重建判別邊界。
              </p>
              <p className="mt-3 mb-1 text-xs font-semibold text-muted-foreground">
                最具判別力特徵（重要度 / 位移）
              </p>
              <ul className="space-y-1 text-xs">
                {diag.top_discriminative.map((f) => (
                  <li key={f.feature} className="flex justify-between gap-2">
                    <span className="font-mono">{f.feature}</span>
                    <span className="text-muted-foreground">
                      {f.importance.toFixed(3)} / 位移 {f.shift.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      )}

      <Note tone="warn" className="mt-4">
        <b>誠實性：</b>CORAL / 工況感知標準化<b>僅用目標未標註特徵</b>（合法無監督 DA）；
        few-shot <b>用了少量真實標籤</b>、屬 few-shot 非零樣本。改善幅度如實呈現 ——
        無監督修不動、需少量真實標籤才有效，量化了「要多少真實標籤才夠」。
      </Note>
    </section>
  );
}

// ===========================================================================
// Live inference — pick a measurement, run the trained classifier server-side.
// Picking a REAL-damage measurement shows the artificial->real gap live.
// ===========================================================================
function LiveInference({ samples }: { samples: Sample[] }) {
  const [idx, setIdx] = useState(0);
  const [out, setOut] = useState<PredictOut | null>(null);
  const [busy, setBusy] = useState(true); // mount fetches immediately
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
    setOut(null); // clear so the stale prediction's ✓/✗ can't flash against the new sample
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
            aria-label="選擇量測樣本"
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
            <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] text-amber-700 dark:text-amber-300 ring-1 ring-inset ring-amber-500/30">
              真實損傷（訓練未見過此分佈）
            </span>
          )}
        </div>

        {err && <Note tone="danger">推論失敗，請確認後端與模型已就緒。</Note>}

        {busy && !out && !err && (
          <p className="py-6 text-center text-sm text-muted-foreground">即時推論中…</p>
        )}

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
