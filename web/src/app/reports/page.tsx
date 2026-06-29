"use client";

import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import Link from "next/link";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { HealthBadge, RiskBadge } from "@/components/dashboard/badges";
import { Card, Note, PageTitle } from "@/components/ui-kit";
import {
  apiGet,
  type ServoCnnResults,
  type ServoModelInfo,
  type ServoReferenceMetrics,
} from "@/lib/api";
import { useFleet, type FleetSource } from "@/lib/fleet";
import { useFleetOps, type OpsSource } from "@/lib/ops";
import type { Equipment, FleetAlert } from "@/lib/mock";
import { HEALTH_COLOR, HEALTH_ZH } from "@/lib/servo";

export default function ReportsPage() {
  const [ref, setRef] = useState<ServoReferenceMetrics | null>(null);
  const [info, setInfo] = useState<ServoModelInfo | null>(null);
  const [cnn, setCnn] = useState<ServoCnnResults | null>(null);
  const [err, setErr] = useState(false);
  const { fleet, source: fleetSource } = useFleet();
  const { alerts, source: opsSource } = useFleetOps();

  useEffect(() => {
    (async () => {
      try {
        const [r, i] = await Promise.all([
          apiGet<ServoReferenceMetrics>("/servo/reference_metrics"),
          apiGet<ServoModelInfo>("/servo/model_info"),
        ]);
        setRef(r);
        setInfo(i);
      } catch {
        setErr(true);
      }
    })();
    apiGet<ServoCnnResults>("/servo/cnn_results")
      .then(setCnn)
      .catch(() => {/* CNN results optional */});
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <PageTitle
        title="報表中心"
        desc="參考模型評估指標、設備別健康比較與告警統計（接後端 /servo/reference_metrics、/servo/fleet、/servo/alerts）"
      />

      {err && (
        <Note tone="danger" className="mb-6">
          無法載入模型指標，請確認後端已啟動。
        </Note>
      )}

      {info?.placeholder && (
        <Note tone="warn" className="mb-6">
          目前指標來自 <b>placeholder 合成資料</b>，僅供流程展示；下載真實 PHM
          資料重訓後即為正式結果。
        </Note>
      )}

      <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-3">
        <MetricCard
          label="分類 macro-F1"
          value={ref?.clf.macro_f1?.toFixed(3) ?? "—"}
          footerMuted={
            <span className="flex flex-col gap-0.5">
              <span>各健康狀態平均表現，0~1，越接近 1 越好</span>
              <span className="text-xs opacity-70">{info?.clf_model ?? "reference clf"}</span>
            </span>
          }
        />
        <MetricCard
          label="回歸 R²"
          value={ref?.reg.r2?.toFixed(3) ?? "—"}
          footerMuted={
            <span className="flex flex-col gap-0.5">
              <span>模型解釋了多少變化，最高 1，越接近 1 越好</span>
              <span className="text-xs opacity-70">{info?.reg_model ?? "reference reg"}</span>
            </span>
          }
        />
        <MetricCard
          label="回歸 MAE"
          value={ref?.reg.mae?.toFixed(3) ?? "—"}
          footerMuted={
            <span className="flex flex-col gap-0.5">
              <span>退化值 DV 的平均誤差，越小越好</span>
              <span className="text-xs opacity-70">退化分數 DV 誤差</span>
            </span>
          }
        />
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="深度學習對照 (PyTorch MLP + 神經 AE)">
          {ref?.dl ? (
            <dl className="space-y-2 text-sm">
              <Row
                k="MLP 分類 macro-F1"
                v={ref.dl.mlp_classification_macro_f1?.toFixed(3) ?? "—"}
              />
              <Row k="MLP 回歸 R²" v={ref.dl.mlp_regression?.r2?.toFixed(3) ?? "—"} />
              <Row k="MLP 回歸 MAE" v={ref.dl.mlp_regression?.mae?.toFixed(3) ?? "—"} />
              <p className="pt-1 text-xs text-muted-foreground">
                macro-F1 / R² 越接近 1 越好、MAE 越小越好。
              </p>
              {ref.dl.note && (
                <p className="pt-1 text-xs text-muted-foreground">{ref.dl.note}</p>
              )}
            </dl>
          ) : (
            <p className="text-sm text-muted-foreground">載入中…</p>
          )}
        </Card>

        <Card title="特徵組合">
          <p className="text-sm text-muted-foreground">
            目前主線模型特徵組：
            <span className="ml-1 font-mono text-foreground">
              {info?.feature_set ?? "—"}
            </span>
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            共 {info?.feature_columns?.length ?? 0} 個特徵欄位
          </p>
          <Link
            href="/servo/simulator"
            className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-primary/15 px-3 py-1.5 text-xs font-medium text-primary ring-1 ring-inset ring-primary/30 transition-colors hover:bg-primary/25"
          >
            <FlaskConical className="h-3.5 w-3.5" />
            到訓練模擬器比較不同設定
          </Link>
        </Card>
      </div>

      <CnnReport cnn={cnn} />

      <EquipmentComparison fleet={fleet} source={fleetSource} />

      <AlarmStatistics alerts={alerts} source={opsSource} />

      <Note tone="info" className="mt-6">
        <b>時間區間彙整</b>（逐班次 / 逐日趨勢）需逐時遙測串流；目前遙測趨勢仍為示意 mock
        （見健康儀表板標示），待實場 / IoT 串流接入後補上。
      </Note>
    </div>
  );
}

function CnnReport({ cnn }: { cnn: ServoCnnResults | null }) {
  const clf = cnn?.classifier;
  if (!clf || clf.macro_f1 == null) return null; // not built (cloud may lack it)
  const rec = cnn?.autoencoder?.reconstruction_error_by_class ?? {};
  const recLabels = ["LN", "LO", "MED", "HI"].filter((l) => l in rec);
  const recMax = Math.max(1e-9, ...recLabels.map((l) => rec[l]));

  return (
    <section className="mt-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide">
        1D-CNN（原始波形）
      </h2>
      <Note tone="info" className="mb-4">
        真正的 <b>1D-CNN</b>（PyTorch）直接吃原始 FMCRD 波形的能量包絡——卷積分類 +
        conv-autoencoder。離線訓練、後端唯讀（<code className="font-mono">/servo/cnn_results</code>）。
      </Note>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <MetricCard
          label="CNN 分類準確率"
          value={clf.accuracy.toFixed(3)}
          footerMuted={
            <span className="flex flex-col gap-0.5">
              <span>整體答對的比例，0~1，越接近 1 越好</span>
              <span className="text-xs opacity-70">留出測試（依檔分離）</span>
            </span>
          }
        />
        <MetricCard
          label="CNN 分類 macro-F1"
          value={clf.macro_f1.toFixed(3)}
          footerMuted={
            <span className="flex flex-col gap-0.5">
              <span>各健康狀態平均表現，越接近 1 越好</span>
              <span className="text-xs opacity-70">{cnn?.architecture?.cnn ?? "1D-CNN"}</span>
            </span>
          }
        />
        <MetricCard
          label="輸入"
          value={`${cnn?.window?.channels?.length ?? 8}×${cnn?.window?.len ?? 256}`}
          footerMuted={
            <span className="flex flex-col gap-0.5">
              <span>輸入波形大小：通道數 × 每段取樣點（非分數）</span>
              <span className="text-xs opacity-70">{`${cnn?.window?.n_train ?? 0}/${cnn?.window?.n_test ?? 0} 段（train/test）`}</span>
            </span>
          }
        />
      </div>

      {recLabels.length > 0 && (
        <Card title="conv-autoencoder 重建誤差（健康擬合，退化越重越大）" className="mt-4">
          <div className="space-y-2">
            {recLabels.map((l) => {
              const pct = Math.round((rec[l] / recMax) * 100);
              return (
                <div key={l} className="text-sm">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-muted-foreground">
                      {HEALTH_ZH[l]} ({l})
                    </span>
                    <span className="font-medium tabular-nums">{rec[l].toFixed(3)}</span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className={`h-full rounded-full ${HEALTH_COLOR[l]?.bar ?? "bg-primary"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          {cnn?.note && (
            <p className="mt-3 text-xs text-muted-foreground">{cnn.note}</p>
          )}
        </Card>
      )}
    </section>
  );
}

const SEVERITY_META: Record<
  "critical" | "warning" | "info",
  { label: string; hex: string; cls: string }
> = {
  critical: { label: "嚴重", hex: "#f87171", cls: "text-red-400" },
  warning: { label: "警示", hex: "#fbbf24", cls: "text-amber-400" },
  info: { label: "提示", hex: "#60a5fa", cls: "text-sky-400" },
};

function EquipmentComparison({
  fleet,
  source,
}: {
  fleet: Equipment[];
  source: FleetSource;
}) {
  const data = fleet.map((u) => ({
    name: u.name,
    health: u.healthScore,
    state: u.state,
    degradation: u.degradation,
    risk: u.risk,
  }));

  return (
    <section className="mt-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide">設備別比較</h2>
      {source !== "mock" ? (
        <Note tone="info" className="mb-4">
          各設備健康分數 / 退化值 (DV) 由<b>真實參考模型</b>在代表性 demo 運轉段上即時計算
          （後端 <code className="font-mono">/servo/fleet</code>）；設備識別為示意。
        </Note>
      ) : (
        <Note tone="warn" className="mb-4">
          後端未連線，以下為 mock fallback；連線後改由 <code className="font-mono">/servo/fleet</code> 即時衍生。
        </Note>
      )}

      <Card title="健康分數（依設備）">
        <div className="h-[240px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="currentColor"
                className="text-border"
                vertical={false}
              />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 11, fill: "currentColor" }}
                className="text-muted-foreground"
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 11, fill: "currentColor" }}
                className="text-muted-foreground"
                tickLine={false}
                axisLine={false}
                width={36}
              />
              <Tooltip
                cursor={{ fill: "var(--muted)", opacity: 0.3 }}
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  fontSize: 12,
                  color: "var(--popover-foreground)",
                }}
                labelStyle={{ color: "var(--muted-foreground)" }}
                formatter={(value) => [`${value}`, "健康分數"]}
              />
              <Bar dataKey="health" radius={[4, 4, 0, 0]} maxBarSize={64}>
                {data.map((d) => (
                  <Cell key={d.name} fill={HEALTH_COLOR[d.state]?.hex ?? "#94a3b8"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 space-y-1.5">
          {fleet.map((u) => (
            <div
              key={u.id}
              className="flex flex-wrap items-center justify-between gap-2 border-b border-border/40 pb-1.5 text-sm last:border-0"
            >
              <span className="font-medium">{u.name}</span>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">DV {u.degradation.toFixed(2)}</span>
                <HealthBadge state={u.state} />
                <RiskBadge level={u.risk} />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </section>
  );
}

function AlarmStatistics({
  alerts,
  source,
}: {
  alerts: FleetAlert[];
  source: OpsSource;
}) {
  const sev = { critical: 0, warning: 0, info: 0 };
  for (const a of alerts) sev[a.severity] += 1;
  const active = alerts.filter((a) => a.status !== "resolved").length;

  const byType = Object.entries(
    alerts.reduce<Record<string, number>>((m, a) => {
      m[a.type] = (m[a.type] ?? 0) + 1;
      return m;
    }, {}),
  ).sort((a, b) => b[1] - a[1]);

  return (
    <section className="mt-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide">告警統計</h2>
      {source !== "mock" ? (
        <Note tone="info" className="mb-4">
          告警由<b>真實模型驅動的機群</b>衍生（後端 <code className="font-mono">/servo/alerts</code>）；
          事件 ID / 時間屬示意性運維包裝。
        </Note>
      ) : (
        <Note tone="warn" className="mb-4">
          後端未連線，以下為 mock fallback；連線後改由 <code className="font-mono">/servo/alerts</code> 即時衍生。
        </Note>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {(["critical", "warning", "info"] as const).map((k) => (
          <MetricCard
            key={k}
            label={`${SEVERITY_META[k].label}告警`}
            value={sev[k]}
            unit="筆"
            valueClassName={sev[k] > 0 ? SEVERITY_META[k].cls : "text-emerald-400"}
            footerMuted={`severity = ${k}`}
          />
        ))}
      </div>

      <Card title="告警類型分布" className="mt-4">
        <p className="mb-3 text-xs text-muted-foreground">
          共 {alerts.length} 筆告警，其中 {active} 筆未結案。
        </p>
        {byType.length > 0 ? (
          <div className="space-y-2">
            {byType.map(([type, n]) => {
              const pct = Math.round((n / Math.max(1, alerts.length)) * 100);
              return (
                <div key={type} className="text-sm">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-muted-foreground">{type}</span>
                    <span className="font-medium tabular-nums">{n} 筆 · {pct}%</span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">目前無告警。</p>
        )}
      </Card>
    </section>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between border-b border-border/40 pb-1.5 last:border-0">
      <span className="text-muted-foreground">{k}</span>
      <span className="font-medium tabular-nums">{v}</span>
    </div>
  );
}
