"use client";

import Link from "next/link";
import { Activity, ArrowRight, Dna, HeartPulse, Target, Zap } from "lucide-react";

import { PlantStatusHeader } from "@/components/dashboard/PlantStatusHeader";
import { ActionRequiredPanel } from "@/components/dashboard/ActionRequiredPanel";
import { ProductionLineMap } from "@/components/dashboard/ProductionLineMap";
import { MotorHealthGrid } from "@/components/dashboard/MotorHealthCard";
import { AlertWorkOrderQueue } from "@/components/dashboard/AlertWorkOrderQueue";
import { HealthTrendPanel } from "@/components/dashboard/HealthTrendPanel";
import { MaintenanceBriefCard } from "@/components/dashboard/MaintenanceBriefCard";
import { SystemStatusPanels } from "@/components/dashboard/SystemStatusPanels";
import { DashboardLoadingSkeleton } from "@/components/dashboard/skeletons";
import {
  LegacyModelCard,
  type LegacyModel,
} from "@/components/dashboard/LegacyModelCard";
import { useFleet } from "@/lib/fleet";
import { useFleetOps } from "@/lib/ops";
import { plantSummary, toMotorViews } from "@/lib/dashboard";

const LEGACY: LegacyModel[] = [
  { code: "模組 C", name: "電流診斷", dataset: "Paderborn", task: "MCSA 故障分類", href: "/module-c", icon: Zap, accent: "rose" },
  { code: "模組 B", name: "動態健康度", dataset: "IMS 軸承", task: "RUL / 健康度", href: "/module-b/overview", icon: HeartPulse, accent: "emerald" },
  { code: "模組 B+", name: "多軌跡泛化", dataset: "XJTU-SY", task: "跨軸承泛化", href: "/module-b-plus/generalization", icon: Dna, accent: "amber" },
  // 合成資料、最不貼近伺服 — 置於最後並以灰階淡化，定位為方法學基礎對照。
  { code: "模組 A", name: "靜態風險", dataset: "AI4I 2020 (合成·基礎)", task: "故障分類", href: "/module-a/predict", icon: Target, accent: "slate" },
];

export default function Overview() {
  const { fleet, source } = useFleet();
  const { alerts, workOrders } = useFleetOps();
  const loading = source === "loading";
  const live = source === "model" || source === "cache";

  // Single derived view-model for the whole page (operator language lives here).
  const views = toMotorViews(fleet, alerts, workOrders).sort(
    (a, b) => a.healthScore - b.healthScore,
  );
  const summary = plantSummary(views, alerts, workOrders);
  const worst = views[0];

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 lg:px-6">
      {/* 1 · 全廠狀態 — is anything wrong right now? */}
      <PlantStatusHeader summary={summary} loading={loading} />

      {/* 2–5 · model-driven sections — skeleton on first visit (no cached data
          yet) so the page never paints the mock placeholder. */}
      {loading ? (
        <DashboardLoadingSkeleton />
      ) : (
        <>
          {/* 2 · 立即處理 — which unit is worst, and what do I do? */}
          <ActionRequiredPanel views={views} />

          {/* 3 · 產線地圖 + 維護摘要 */}
          <section className="grid gap-4 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <ProductionLineMap views={views} />
            </div>
            <MaintenanceBriefCard views={views} />
          </section>

          {/* 4 · 設備健康卡片 — operation-oriented */}
          <div>
            <SectionHeader
              title="設備健康總覽"
              desc={
                live
                  ? "健康分數由參考模型即時計算；RUL / 建議為衍生推估"
                  : "mock fallback（後端未連線）；健康為示意資料"
              }
              action={{ label: "Servo 健康儀表板", href: "/servo/dashboard" }}
            />
            <MotorHealthGrid views={views} />
          </div>

          {/* 5 · 告警 / 工單佇列 */}
          <div>
            <SectionHeader
              title="告警 / 工單佇列"
              desc="作用中事件依嚴重度與 SLA 待派工"
              action={{ label: "前往告警 / 工單", href: "/alerts" }}
            />
            <AlertWorkOrderQueue
              alerts={alerts.slice(0, 5)}
              workOrders={workOrders}
              views={views}
            />
          </div>
        </>
      )}

      {/* 6 · 健康 / 告警趨勢 */}
      <HealthTrendPanel />

      {/* 7 · 系統狀態（真實 API） */}
      <div>
        <SectionHeader title="系統狀態" desc="參考模型 / LLM 助理 / 知識庫 / 最新預測" />
        <SystemStatusPanels
          worst={{ name: worst.name, state: worst.state, score: worst.healthScore }}
          loading={loading}
        />
      </div>

      {/* 8 · Legacy / 對照實驗 */}
      <div>
        <SectionHeader
          title="Legacy / 對照實驗"
          desc="模組 A / B / B+ / C — 保留為對照與歷史補充，非主線"
        />
        <section className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {LEGACY.map((m) => (
            <LegacyModelCard key={m.code} model={m} />
          ))}
        </section>
      </div>
    </div>
  );
}

function SectionHeader({
  title,
  desc,
  action,
}: {
  title: string;
  desc?: string;
  action?: { label: string; href: string };
}) {
  return (
    <div className="mb-3 flex items-end justify-between gap-4">
      <div>
        <h2 className="flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide">
          <Activity className="h-3.5 w-3.5 text-muted-foreground" />
          {title}
        </h2>
        {desc && <p className="mt-0.5 text-xs text-muted-foreground">{desc}</p>}
      </div>
      {action && (
        <Link
          href={action.href}
          className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-primary hover:underline"
        >
          {action.label}
          <ArrowRight className="h-3 w-3" />
        </Link>
      )}
    </div>
  );
}
