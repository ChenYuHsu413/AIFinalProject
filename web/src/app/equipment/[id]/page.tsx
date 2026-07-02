"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft, Bot, Loader2, MapPin, RefreshCw, Wrench } from "lucide-react";

import { Card, Note, PageTitle } from "@/components/ui-kit";
import { HealthScoreGauge } from "@/components/dashboard/HealthScoreGauge";
import { FeatureImportancePanel } from "@/components/dashboard/FeatureImportancePanel";
import { TelemetryTrends } from "@/components/dashboard/TelemetryTrends";
import { HealthBadge, RiskBadge, StatusDot } from "@/components/dashboard/badges";
import { TELEMETRY } from "@/lib/mock";
import { useFleet } from "@/lib/fleet";
import {
  apiGet,
  apiPost,
  type ServoModelInfo,
  type ServoPrediction,
  type ServoSample,
} from "@/lib/api";
import { HEALTH_COLOR, HEALTH_ORDER, HEALTH_ZH } from "@/lib/servo";

export default function EquipmentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { fleet } = useFleet();
  const unit = fleet.find((u) => u.id === id);

  if (!unit) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-12">
        <Note tone="warn">
          找不到設備 <code className="font-mono">{id}</code>。
          <Link href="/" className="ml-2 font-medium text-primary hover:underline">
            回總覽
          </Link>
        </Note>
      </div>
    );
  }

  const c = HEALTH_COLOR[unit.state];
  const telemetry = TELEMETRY[unit.id] ?? [];

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 lg:px-6">
      <Link
        href="/"
        className="mb-3 inline-flex items-center gap-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        回總覽
      </Link>

      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <PageTitle title={unit.name} />
          <p className="-mt-5 flex items-center gap-1.5 text-sm text-muted-foreground">
            <MapPin className="h-3.5 w-3.5" />
            {unit.location}
          </p>
        </div>
        <StatusDot status={unit.status} />
      </div>

      {/* mock health snapshot */}
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide">健康快照</h2>
        <span className="text-[11px] text-muted-foreground">示意 mock</span>
      </div>
      <div className="mb-6 grid gap-4 lg:grid-cols-3">
        <div className="flex flex-col items-center justify-center rounded-xl border border-border/70 bg-card/70 p-6 shadow-sm backdrop-blur-sm">
          <HealthScoreGauge score={unit.healthScore} />
          <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
            <HealthBadge state={unit.state} />
            <RiskBadge level={unit.risk} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 lg:col-span-2">
          <MiniStat label="退化分數 DV" value={unit.degradation.toFixed(2)} sub="0=健康 · 1=高度退化" valueClass={c.text} />
          <MiniStat label="模型信心" value={`${(unit.confidence * 100).toFixed(0)}%`} sub="confidence" />
          <MiniStat label="運轉時數" value={unit.uptimeHours.toLocaleString()} sub="hours" />
          <MiniStat label="狀態更新" value={unit.lastUpdated} sub="last telemetry" />
          <div className="col-span-2 rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm">
            <p className="text-xs text-muted-foreground">
              主要異常特徵{" "}
              <span className="font-mono font-medium text-foreground">
                {unit.topFeature.feature}
              </span>{" "}
              (z={unit.topFeature.z})
            </p>
            <p className="mt-1 text-sm">{unit.topFeature.hint}</p>
          </div>
        </div>
      </div>

      {/* telemetry trends (mock) */}
      <div className="mb-2 flex items-end justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide">感測器趨勢</h2>
        <span className="text-[11px] text-muted-foreground">示意 mock · 待真實遙測串流</span>
      </div>
      <div className="mb-6">
        <TelemetryTrends data={telemetry} />
      </div>

      {/* real model prediction */}
      <RealPredictionSection state={unit.state} />
    </div>
  );
}

/**
 * Bridges the mock fleet to the *real* reference model: picks a demo sample whose
 * ground-truth label matches this unit's state, runs it through POST /servo/predict,
 * and shows the actual model output. (Mock unit → representative real reading.)
 */
function RealPredictionSection({ state }: { state: string }) {
  const [cols, setCols] = useState<string[]>([]);
  const [matches, setMatches] = useState<{ row: ServoSample; idx: number }[]>([]);
  const [pos, setPos] = useState(0);
  const [pred, setPred] = useState<ServoPrediction | null>(null);
  const [busy, setBusy] = useState(true);
  const [err, setErr] = useState(false);

  // request sequence — the [state] effect can rerun (mock → model) while a
  // predictRow from the previous run is still in flight; the stale response
  // must not overwrite the newer one.
  const seqRef = useRef(0);

  async function predictRow(row: ServoSample, columns: string[]) {
    const seq = ++seqRef.current;
    setBusy(true);
    setErr(false); // clear any prior error so a successful retry recovers
    try {
      const features: Record<string, number> = {};
      for (const col of columns) features[col] = Number(row[col]);
      const p = await apiPost<ServoPrediction>("/servo/predict", { features });
      if (seq === seqRef.current) setPred(p);
    } catch {
      if (seq === seqRef.current) setErr(true);
    } finally {
      if (seq === seqRef.current) setBusy(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [info, rows] = await Promise.all([
          apiGet<ServoModelInfo>("/servo/model_info"),
          apiGet<ServoSample[]>("/servo/samples"),
        ]);
        if (cancelled) return;
        const matched = rows
          .map((row, idx) => ({ row, idx }))
          .filter((m) => String(m.row["ylabel"]) === state);
        const list = matched.length
          ? matched
          : [{ row: rows[Math.floor(rows.length / 2)], idx: Math.floor(rows.length / 2) }];
        if (!list[0]?.row) {
          setErr(true);
          setBusy(false);
          return;
        }
        setCols(info.feature_columns);
        setMatches(list);
        setPos(0);
        await predictRow(list[0].row, info.feature_columns);
      } catch {
        if (!cancelled) {
          setErr(true);
          setBusy(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [state]);

  function reroll() {
    if (matches.length < 2 || busy) return;
    const next = (pos + 1) % matches.length;
    setPos(next);
    predictRow(matches[next].row, cols);
  }

  const current = matches[pos];

  return (
    <>
      <div className="mb-2 flex items-end justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide">
          參考模型預測
        </h2>
        <span className="text-[11px] text-emerald-600 dark:text-emerald-300">
          接真 API · /servo/predict
        </span>
      </div>

      {busy && !pred ? (
        <div className="flex items-center gap-2 rounded-xl border border-border/70 bg-card/40 p-8 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> 以代表性運轉段估測中…
        </div>
      ) : err || !pred ? (
        <Note tone="danger">
          無法取得模型預測，請確認後端已啟動（/servo/predict）。
        </Note>
      ) : (
        <div className="space-y-4">
          {pred.placeholder && (
            <Note tone="warn">
              模型以 <b>placeholder 合成資料</b> 訓練，僅供流程展示。
            </Note>
          )}

          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs text-muted-foreground">
              以與本設備狀態相符的代表性 demo 運轉段
              {current && (
                <>
                  （demo #{current.idx} · 真實標籤 {HEALTH_ZH[state] ?? state}）
                </>
              )}
              送入參考模型，下方為實際模型輸出。
            </p>
            {matches.length > 1 && (
              <button
                type="button"
                onClick={reroll}
                disabled={busy}
                className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border bg-card/60 px-2.5 py-1 text-xs font-medium text-foreground transition-colors hover:bg-muted disabled:opacity-50"
              >
                {busy ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="h-3.5 w-3.5" />
                )}
                換一筆代表段重估（{matches.length} 筆）
              </button>
            )}
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="各健康狀態機率">
              <div className="space-y-2.5">
                {HEALTH_ORDER.filter((k) => k in pred.health_state_proba).map(
                  (k) => {
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
                  },
                )}
              </div>
            </Card>

            <Card title="主要異常特徵（模型）">
              <FeatureImportancePanel features={pred.top_features} />
            </Card>
          </div>

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
            href={current ? `/servo/assistant?sample=${current.idx}` : "/servo/assistant"}
            className="flex items-start gap-2 rounded-xl border border-violet-500/30 bg-violet-500/10 p-4 text-sm text-violet-700 dark:text-violet-200 transition-colors hover:bg-violet-500/15"
          >
            <Bot className="mt-0.5 h-4 w-4 shrink-0" />
            <span>到「LLM 維護助理」用這筆結果生成完整維修建議與工單草稿。</span>
          </Link>
        </div>
      )}
    </>
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
