"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Bot, Loader2, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  apiGet,
  apiPost,
  type ServoModelInfo,
  type ServoPrediction,
  type ServoSample,
} from "@/lib/api";
import {
  HEALTH_COLOR,
  HEALTH_ORDER,
  HEALTH_ZH,
  RISK_COLOR,
  RISK_ZH,
} from "@/lib/servo";

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
    <div className="mx-auto max-w-5xl px-6 py-8">
      <PageHeader />

      {loadErr && (
        <Note tone="danger">
          無法載入 demo 樣本或模型資訊。請確認後端已啟動。
        </Note>
      )}

      {/* sample picker */}
      <div className="mb-6 rounded-xl border bg-gradient-to-br from-sky-50 to-white p-5 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold">選擇一筆運轉段</h2>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <label className="flex-1">
            <span className="mb-1 block text-xs text-muted-foreground">
              樣本（來自 demo 樣本筆）
            </span>
            <select
              value={idx}
              onChange={(e) => setIdx(Number(e.target.value))}
              className="w-full rounded-lg border bg-white px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
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

      {pred && <Result pred={pred} trueLabel={trueLabel} />}
    </div>
  );
}

function PageHeader() {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-bold tracking-tight">Servo 健康狀態儀表板</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        輸入一段運轉資料，估測健康狀態（LN/LO/MED/HI）、退化分數、風險等級、
        主要異常特徵與模型信心，並給出建議處置。
      </p>
    </div>
  );
}

function Result({
  pred,
  trueLabel,
}: {
  pred: ServoPrediction;
  trueLabel?: string;
}) {
  const c = HEALTH_COLOR[pred.predicted_health_state] ?? HEALTH_COLOR.MED;

  return (
    <div className="space-y-6">
      {pred.placeholder && (
        <Note tone="warn">
          目前模型以 <b>placeholder 合成資料</b> 訓練，僅供流程展示；下載真實 PHM
          資料並重訓後即為正式結果。
        </Note>
      )}

      {/* KPI cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="健康狀態" value={pred.health_state_zh} sub={`分類：${pred.predicted_health_state}`} valueClass={c.text} />
        <Stat label="退化分數 DV" value={pred.degradation_score.toFixed(2)} sub="0=健康 · 1=高度退化" valueClass={c.text} />
        <Stat label="健康分數" value={pred.health_score.toFixed(0)} sub="(1−DV)×100" valueClass={c.text} />
        <div className="rounded-xl border bg-white p-4 shadow-sm">
          <p className="text-xs text-muted-foreground">風險等級</p>
          <span className={`mt-1 inline-block rounded-full px-3 py-1 text-sm font-semibold ${RISK_COLOR[pred.risk_level]}`}>
            {RISK_ZH[pred.risk_level]} · {pred.risk_level}
          </span>
          <p className="mt-2 text-xs text-muted-foreground">
            模型信心 <b className="text-foreground">{(pred.model_confidence * 100).toFixed(0)}%</b>
          </p>
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
                    <span className="text-muted-foreground">{(v * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                    <div className={`h-full rounded-full ${col.bar}`} style={{ width: `${v * 100}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
          {trueLabel && (
            <p className="mt-3 text-xs text-muted-foreground">
              {trueLabel === pred.predicted_health_state ? "✅ 與真實標籤一致：" : "⚠ 與真實標籤不同："}
              真實 {HEALTH_ZH[trueLabel] ?? trueLabel}（{trueLabel}）
            </p>
          )}
        </Card>

        {/* top anomalous features */}
        <Card title="主要異常特徵">
          <div className="space-y-3">
            {pred.top_features.map((t) => {
              const mag = Math.min(1, Math.abs(t.z) / 6);
              const bar = Math.abs(t.z) > 3 ? "bg-red-500" : Math.abs(t.z) > 1.5 ? "bg-amber-500" : "bg-emerald-500";
              return (
                <div key={t.feature}>
                  <div className="mb-0.5 flex justify-between text-xs">
                    <span className="font-mono font-medium">{t.feature}</span>
                    <span className="text-muted-foreground">z = {t.z}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <div className={`h-full rounded-full ${bar}`} style={{ width: `${mag * 100}%` }} />
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{t.hint}</p>
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      {/* maintenance advice */}
      <Card title="建議處置">
        <ul className="space-y-2">
          {pred.maintenance_advice.map((tip, i) => (
            <li key={i} className="flex gap-2 rounded-lg border bg-muted/30 p-3 text-sm">
              <span className="text-primary">•</span>
              <span>{tip}</span>
            </li>
          ))}
        </ul>
      </Card>

      <div className="flex items-start gap-2 rounded-xl border border-violet-200 bg-violet-50 p-4 text-sm text-violet-800">
        <Bot className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          想要更完整的人話解釋與工單草稿？到側邊欄「LLM 維護助理」頁，它會接收這筆結果並產生維修建議。
        </span>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  valueClass,
}: {
  label: string;
  value: string;
  sub: string;
  valueClass?: string;
}) {
  return (
    <div className="rounded-xl border bg-white p-4 shadow-sm">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`mt-1 text-2xl font-bold ${valueClass ?? ""}`}>{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{sub}</p>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border bg-white p-5 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold">{title}</h2>
      {children}
    </div>
  );
}

function Note({
  tone,
  children,
}: {
  tone: "warn" | "danger";
  children: React.ReactNode;
}) {
  const cls =
    tone === "danger"
      ? "border-red-200 bg-red-50 text-red-800"
      : "border-amber-200 bg-amber-50 text-amber-800";
  return (
    <div className={`mb-2 rounded-lg border px-4 py-2.5 text-sm ${cls}`}>
      {children}
    </div>
  );
}
