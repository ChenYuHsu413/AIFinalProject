"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AlertTriangle, Bot, Loader2, Search, Wrench } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, Note, PageTitle } from "@/components/ui-kit";
import { HealthScoreGauge } from "@/components/dashboard/HealthScoreGauge";
import { FeatureImportancePanel } from "@/components/dashboard/FeatureImportancePanel";
import { TelemetryTrends } from "@/components/dashboard/TelemetryTrends";
import { HealthBadge, RiskBadge } from "@/components/dashboard/badges";
import {
  API_BASE,
  apiGet,
  apiPost,
  type ServoModelInfo,
  type ServoPrediction,
  type ServoSample,
} from "@/lib/api";
import { HEALTH_COLOR, HEALTH_ORDER, HEALTH_ZH } from "@/lib/servo";
import { TELEMETRY } from "@/lib/mock";

export default function ServoDashboardPage() {
  const [cols, setCols] = useState<string[] | null>(null);
  const [samples, setSamples] = useState<ServoSample[]>([]);
  const [idx, setIdx] = useState(0);
  const [pred, setPred] = useState<ServoPrediction | null>(null);
  const [loadErr, setLoadErr] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [info, rows] = await Promise.all([
          apiGet<ServoModelInfo>("/servo/model_info"),
          apiGet<ServoSample[]>("/servo/samples"),
        ]);
        setCols(info.feature_columns);
        setSamples(rows);
      } catch {
        setLoadErr(true);
      }
    })();
  }, []);

  async function predict() {
    if (!cols || !samples[idx]) return;
    setBusy(true);
    try {
      const row = samples[idx];
      const features: Record<string, number> = {};
      for (const c of cols) features[c] = Number(row[c]);
      setPred(await apiPost<ServoPrediction>("/servo/predict", { features }));
    } finally {
      setBusy(false);
    }
  }

  const trueLabel = samples[idx]?.["ylabel"] as string | undefined;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <PageTitle
        title="Servo 健康儀表板"
        desc="輸入一段運轉資料，估測健康狀態（LN/LO/MED/HI）、退化分數、風險等級、主要異常特徵與模型信心，並給出建議處置。"
      />

      <ProvenancePanel />

      {loadErr && (
        <Note tone="danger" className="mb-6">
          無法載入 demo 樣本或模型資訊。請確認後端已啟動。
        </Note>
      )}

      {/* sample picker */}
      <div className="mb-6 rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
        <h2 className="mb-3 text-sm font-semibold">選擇一筆運轉段</h2>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <label className="flex-1">
            <span className="mb-1 block text-xs text-muted-foreground">
              樣本（來自 demo 樣本筆）
            </span>
            <select
              value={idx}
              onChange={(e) => setIdx(Number(e.target.value))}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
            >
              {samples.map((s, i) => {
                const l = String(s["ylabel"] ?? "?");
                return (
                  <option key={i} value={i}>
                    #{i} · 真實標籤 {HEALTH_ZH[l] ?? l} ({l})
                  </option>
                );
              })}
            </select>
          </label>
          <Button onClick={predict} disabled={busy || !samples.length}>
            {busy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Search className="h-4 w-4" />
            )}
            估測健康狀態
          </Button>
        </div>
      </div>

      {pred ? (
        <Result pred={pred} trueLabel={trueLabel} sampleIdx={idx} />
      ) : (
        !loadErr && (
          <div className="rounded-xl border border-dashed border-border/70 bg-card/40 p-12 text-center text-sm text-muted-foreground">
            選一筆運轉段並按「估測健康狀態」即可看到健康分數、風險、異常特徵與建議處置。
          </div>
        )
      )}
    </div>
  );
}

function Result({
  pred,
  trueLabel,
  sampleIdx,
}: {
  pred: ServoPrediction;
  trueLabel?: string;
  sampleIdx: number;
}) {
  const c = HEALTH_COLOR[pred.predicted_health_state] ?? HEALTH_COLOR.MED;
  // Illustrative recent-telemetry window (mock until Servo Dataset streams real).
  const telemetry = TELEMETRY["servo-a02"];

  return (
    <div className="space-y-6">
      {pred.placeholder && (
        <Note tone="warn">
          目前模型以 <b>placeholder 合成資料</b> 訓練，僅供流程展示；下載真實 PHM
          資料並重訓後即為正式結果。
        </Note>
      )}

      {/* Top: gauge + headline metrics */}
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="flex flex-col items-center justify-center rounded-xl border border-border/70 bg-card/70 p-6 shadow-sm backdrop-blur-sm">
          <HealthScoreGauge score={pred.health_score} />
          <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
            <HealthBadge state={pred.predicted_health_state} />
            <RiskBadge level={pred.risk_level} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 lg:col-span-2">
          <MiniStat label="健康狀態" value={pred.health_state_zh} sub={`分類 ${pred.predicted_health_state}`} valueClass={c.text} />
          <MiniStat label="退化分數 DV" value={pred.degradation_score.toFixed(2)} sub="0=健康 · 1=高度退化" valueClass={c.text} />
          <MiniStat label="健康分數" value={pred.health_score.toFixed(0)} sub="(1−DV)×100" valueClass={c.text} />
          <MiniStat label="模型信心" value={`${(pred.model_confidence * 100).toFixed(0)}%`} sub="classifier confidence" />
        </div>
      </div>

      {pred.consistency_warning && (
        <Note tone="danger">
          <AlertTriangle className="mr-1 inline h-4 w-4" />
          {pred.consistency_warning}
        </Note>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* probability bars */}
        <Card title="各健康狀態機率">
          <div className="space-y-2.5">
            {HEALTH_ORDER.filter((k) => k in pred.health_state_proba).map((k) => {
              const v = pred.health_state_proba[k] ?? 0;
              const col = HEALTH_COLOR[k];
              return (
                <div key={k}>
                  <div className="mb-0.5 flex justify-between text-xs">
                    <span className="font-medium">
                      {HEALTH_ZH[k]} ({k})
                    </span>
                    <span className="text-muted-foreground">
                      {(v * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                    <div
                      className={`h-full rounded-full ${col.bar}`}
                      style={{ width: `${v * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          {trueLabel && (
            <p className="mt-3 text-xs text-muted-foreground">
              {trueLabel === pred.predicted_health_state
                ? "✅ 與真實標籤一致："
                : "⚠ 與真實標籤不同："}
              真實 {HEALTH_ZH[trueLabel] ?? trueLabel}（{trueLabel}）
            </p>
          )}
        </Card>

        {/* top anomalous features */}
        <Card title="主要異常特徵">
          <FeatureImportancePanel features={pred.top_features} />
        </Card>
      </div>

      {/* telemetry trends (mock) */}
      <div>
        <div className="mb-2 flex items-end justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide">
            感測器趨勢
          </h2>
          <span className="text-[11px] text-muted-foreground">
            示意 mock · 待真實遙測串流
          </span>
        </div>
        <TelemetryTrends data={telemetry} />
      </div>

      {/* maintenance advice */}
      <Card title="建議處置">
        <ul className="space-y-2">
          {pred.maintenance_advice.map((tip, i) => (
            <li
              key={i}
              className="flex gap-2 rounded-lg border border-border/60 bg-muted/30 p-3 text-sm"
            >
              <Wrench className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
              <span>{tip}</span>
            </li>
          ))}
        </ul>
      </Card>

      <Link
        href={`/servo/assistant?sample=${sampleIdx}`}
        className="flex items-start gap-2 rounded-xl border border-violet-500/30 bg-violet-500/10 p-4 text-sm text-violet-200 transition-colors hover:bg-violet-500/20"
      >
        <Bot className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          想要更完整的人話解釋與工單草稿？點此前往「LLM 維護助理」頁，它會接收<b>這筆</b>結果並產生維修建議。
        </span>
      </Link>
    </div>
  );
}

interface Provenance {
  is_simulation: boolean;
  source?: { n_files: number; total_uncompressed_gb: number };
  features?: { aggregated_segments: number };
  model?: { eval: string; placeholder: boolean; clf_macro_f1: number; reg_r2: number };
}

function ProvenancePanel() {
  const [p, setP] = useState<Provenance | null>(null);
  useEffect(() => {
    apiGet<Provenance>("/servo/provenance")
      .then(setP)
      .catch(() => {});
  }, []);
  if (!p?.source || !p.model) return null;
  return (
    <div className="mb-6 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4 shadow-sm">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm">
        <span className="font-semibold text-emerald-300">✅ 訓練於真實 PHM FMCRD 資料集</span>
        <span className="text-muted-foreground">
          {p.source.total_uncompressed_gb} GB · {p.source.n_files} 檔 ·{" "}
          {p.features?.aggregated_segments?.toLocaleString()} 段
        </span>
        <span className="text-muted-foreground">
          留出 macro-F1 {p.model.clf_macro_f1.toFixed(3)} · DV R² {p.model.reg_r2.toFixed(3)}
        </span>
        <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] text-emerald-300 ring-1 ring-inset ring-emerald-500/30">
          placeholder = {String(p.model.placeholder)}
        </span>
      </div>
      {p.is_simulation && (
        <p className="mt-1.5 text-[11px] text-muted-foreground">
          FMCRD 為高擬真<b>模擬</b>資料集（非真實工廠伺服馬達遙測）；「真實」指完整大型公開 PHM 資料集本身（相對於先前 placeholder 合成資料）。
        </p>
      )}
      <details className="mt-2">
        <summary className="cursor-pointer text-[11px] text-emerald-300/90 hover:text-emerald-200">
          查看資料溯源圖（DV 各類分布 + 留出測試混淆矩陣）
        </summary>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${API_BASE}/figures/servo_provenance.png`}
          alt="Servo 資料溯源：DV 各健康類別分布與留出測試混淆矩陣"
          className="mt-2 w-full rounded-lg border border-border/60 bg-white"
        />
      </details>
    </div>
  );
}

function MiniStat({
  label,
  value,
  sub,
  valueClass,
}: {
  label: string;
  value: string;
  sub?: string;
  valueClass?: string;
}) {
  return (
    <div className="rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`mt-1 text-2xl font-bold ${valueClass ?? ""}`}>{value}</p>
      {sub && <p className="mt-1 text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}
