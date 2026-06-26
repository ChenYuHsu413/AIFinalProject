"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  BookOpen,
  Bot,
  Dna,
  HeartPulse,
  ShieldCheck,
  Target,
  Zap,
} from "lucide-react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { EquipmentHealthCard } from "@/components/dashboard/EquipmentHealthCard";
import { FleetHealthChart } from "@/components/dashboard/FleetHealthChart";
import { AlertTable } from "@/components/dashboard/AlertTable";
import {
  LegacyModelCard,
  type LegacyModel,
} from "@/components/dashboard/LegacyModelCard";
import { HealthBadge } from "@/components/dashboard/badges";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ALERTS, FLEET, fleetSummary } from "@/lib/mock";
import { HEALTH_COLOR } from "@/lib/servo";
import {
  apiGet,
  type AssistantProviders,
  type KnowledgeDoc,
  type ServoModelInfo,
  type ServoReferenceMetrics,
} from "@/lib/api";

const LEGACY: LegacyModel[] = [
  { code: "模組 A", name: "靜態風險", dataset: "AI4I 2020 (合成)", task: "故障分類", href: "/module-a/predict", icon: Target, accent: "blue" },
  { code: "模組 B", name: "動態健康度", dataset: "IMS 軸承", task: "RUL / 健康度", href: "/module-b/overview", icon: HeartPulse, accent: "emerald" },
  { code: "模組 B+", name: "多軌跡泛化", dataset: "XJTU-SY", task: "跨軸承泛化", href: "/module-b-plus/generalization", icon: Dna, accent: "amber" },
  { code: "模組 C", name: "電流診斷", dataset: "Paderborn", task: "MCSA 故障分類", href: "/module-c", icon: Zap, accent: "rose" },
];

export default function Overview() {
  const s = fleetSummary();
  const worst = [...FLEET].sort((a, b) => a.healthScore - b.healthScore)[0];

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 lg:px-6">
      {/* heading row */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="mb-1 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-cyan-300">
            <Activity className="h-3.5 w-3.5" />
            Predictive Maintenance · 指揮中心
          </div>
          <h1 className="text-2xl font-bold tracking-tight">
            AI Servo Motor Health Command Center
          </h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            伺服馬達健康監控與智慧維護指揮中心
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Badge variant="success">
            <span className="size-1.5 rounded-full bg-emerald-400" />
            即時監控
          </Badge>
          <Badge variant="outline">Demo · mock fleet</Badge>
        </div>
      </div>

      {/* KPI grid — shadcn dashboard-01 gradient cards */}
      <section className="grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="監控設備"
          value={s.total}
          unit="台"
          trend={s.trends.total}
          footerStrong="本班次新增 1 台"
          footerMuted="Demo 機群 + 實驗台"
        />
        <MetricCard
          label="平均健康分數"
          value={s.avgHealth}
          unit="/100"
          trend={s.trends.avgHealth}
          valueClassName={s.avgHealth >= 70 ? "text-emerald-400" : "text-amber-400"}
          footerStrong="較上班次下降"
          footerMuted="全機群加權平均"
        />
        <MetricCard
          label="高風險設備"
          value={s.highRisk}
          unit="台"
          trend={s.trends.highRisk}
          valueClassName={s.highRisk > 0 ? "text-red-400" : "text-emerald-400"}
          footerStrong="新增 1 台高風險"
          footerMuted="風險等級 = High"
        />
        <MetricCard
          label="作用中告警"
          value={s.activeAlerts}
          unit="筆"
          trend={s.trends.activeAlerts}
          valueClassName={s.activeAlerts > 0 ? "text-amber-400" : "text-emerald-400"}
          footerStrong="新增 2 筆告警"
          footerMuted="未解決告警事件"
        />
      </section>

      {/* hero chart + ranking */}
      <section className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <FleetHealthChart />
        </div>
        <EquipmentRankCard />
      </section>

      {/* fleet health */}
      <div>
        <SectionHeader
          title="設備健康總覽"
          desc="各伺服馬達即時健康狀態（mock，待 Servo Dataset 模組接真實遙測）"
          action={{ label: "Servo 健康儀表板", href: "/servo/dashboard" }}
        />
        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {FLEET.map((u) => (
            <EquipmentHealthCard key={u.id} unit={u} />
          ))}
        </section>
      </div>

      {/* system status islands (real API) */}
      <div>
        <SectionHeader title="系統狀態" desc="參考模型 / LLM 助理 / 知識庫 / 最新預測" />
        <section className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
          <LatestPredictionPanel
            unit={worst.name}
            state={worst.state}
            score={worst.healthScore}
          />
          <ReferenceModelPanel />
          <AssistantPanel />
          <KnowledgePanel />
        </section>
      </div>

      {/* active alerts preview */}
      <div>
        <SectionHeader
          title="最新告警"
          desc="作用中與近期事件"
          action={{ label: "前往告警 / 工單", href: "/alerts" }}
        />
        <AlertTable alerts={ALERTS.slice(0, 4)} />
      </div>

      {/* legacy modules */}
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
        <h2 className="text-sm font-semibold uppercase tracking-wide">{title}</h2>
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

/** "Recent sales"-style ranking list adapted to equipment health. */
function EquipmentRankCard() {
  const ranked = [...FLEET].sort((a, b) => a.healthScore - b.healthScore);
  return (
    <Card className="@container/card">
      <CardHeader>
        <CardDescription>需關注設備</CardDescription>
        <CardTitle className="text-lg">健康分數排行（由低到高）</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {ranked.map((u) => {
          const c = HEALTH_COLOR[u.state];
          return (
            <Link
              key={u.id}
              href={`/equipment/${u.id}`}
              className="flex items-center gap-3 rounded-lg px-1 py-1 transition-colors hover:bg-muted/40"
            >
              <span className={`flex h-9 w-9 items-center justify-center rounded-lg ${c.chip}`}>
                <Activity className="h-4 w-4" />
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{u.name}</p>
                <p className="truncate text-xs text-muted-foreground">{u.location}</p>
              </div>
              <div className="flex flex-col items-end gap-0.5">
                <span className={`text-sm font-bold tabular-nums ${c.text}`}>
                  {u.healthScore}
                </span>
                <HealthBadge state={u.state} />
              </div>
            </Link>
          );
        })}
      </CardContent>
    </Card>
  );
}

/** Generic status panel shell. */
function StatusPanel({
  icon: Icon,
  title,
  children,
  tone = "cyan",
}: {
  icon: typeof ShieldCheck;
  title: string;
  children: React.ReactNode;
  tone?: "cyan" | "emerald" | "amber" | "violet";
}) {
  const ring = {
    cyan: "text-cyan-300 bg-cyan-500/15 ring-cyan-500/30",
    emerald: "text-emerald-300 bg-emerald-500/15 ring-emerald-500/30",
    amber: "text-amber-300 bg-amber-500/15 ring-amber-500/30",
    violet: "text-violet-300 bg-violet-500/15 ring-violet-500/30",
  }[tone];
  return (
    <div className="rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm">
      <div className="mb-2 flex items-center gap-2">
        <span className={`flex h-7 w-7 items-center justify-center rounded-lg ring-1 ring-inset ${ring}`}>
          <Icon className="h-4 w-4" />
        </span>
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

function LatestPredictionPanel({
  unit,
  state,
  score,
}: {
  unit: string;
  state: "LN" | "LO" | "MED" | "HI";
  score: number;
}) {
  return (
    <StatusPanel icon={Activity} title="最新預測" tone="violet">
      <p className="text-sm font-semibold">{unit}</p>
      <div className="mt-1.5 flex items-center gap-2">
        <HealthBadge state={state} />
        <span className="text-xs text-muted-foreground">分數 {score}/100</span>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        最近一次估測為機群最低分設備
      </p>
    </StatusPanel>
  );
}

function ReferenceModelPanel() {
  const [ref, setRef] = useState<ServoReferenceMetrics | null>(null);
  const [info, setInfo] = useState<ServoModelInfo | null>(null);
  const [err, setErr] = useState(false);

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
  }, []);

  return (
    <StatusPanel icon={ShieldCheck} title="參考模型" tone="cyan">
      {err ? (
        <p className="text-xs text-muted-foreground">後端未連線</p>
      ) : !ref ? (
        <p className="text-xs text-muted-foreground">載入中…</p>
      ) : (
        <>
          <p className="text-sm font-semibold">{info?.clf_model ?? "—"}</p>
          <dl className="mt-1.5 space-y-0.5 text-xs text-muted-foreground">
            <div className="flex justify-between">
              <dt>分類 macro-F1</dt>
              <dd className="font-medium tabular-nums text-foreground">
                {ref.clf.macro_f1?.toFixed(3) ?? "—"}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt>回歸 R²</dt>
              <dd className="font-medium tabular-nums text-foreground">
                {ref.reg.r2?.toFixed(3) ?? "—"}
              </dd>
            </div>
          </dl>
          {info?.placeholder && (
            <p className="mt-2 inline-block rounded bg-amber-500/15 px-1.5 py-0.5 text-[11px] font-medium text-amber-300 ring-1 ring-inset ring-amber-500/30">
              placeholder 合成資料
            </p>
          )}
        </>
      )}
    </StatusPanel>
  );
}

function AssistantPanel() {
  const [providers, setProviders] = useState<string[] | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const r = await apiGet<AssistantProviders>("/servo/assistant/providers");
        setProviders(r.providers);
      } catch {
        setErr(true);
      }
    })();
  }, []);

  const online = (providers?.length ?? 0) > 0;
  return (
    <StatusPanel icon={Bot} title="LLM 維護助理" tone="violet">
      {err ? (
        <p className="text-xs text-muted-foreground">後端未連線</p>
      ) : providers === null ? (
        <p className="text-xs text-muted-foreground">載入中…</p>
      ) : (
        <>
          <div className="flex items-center gap-1.5">
            <span
              className={`h-2 w-2 rounded-full ${online ? "bg-emerald-400" : "bg-amber-400"}`}
            />
            <span className="text-sm font-semibold">
              {online ? "供應商就緒" : "本地 fallback"}
            </span>
          </div>
          <p className="mt-1.5 text-xs text-muted-foreground">
            {online ? providers.join(" · ") : "未設定 API 金鑰，使用規則式建議"}
          </p>
          <Link
            href="/servo/assistant"
            className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
          >
            開啟助理 <ArrowRight className="h-3 w-3" />
          </Link>
        </>
      )}
    </StatusPanel>
  );
}

function KnowledgePanel() {
  const [docs, setDocs] = useState<KnowledgeDoc[] | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const r = await apiGet<KnowledgeDoc[]>("/knowledge/documents");
        setDocs(r);
      } catch {
        setErr(true);
      }
    })();
  }, []);

  return (
    <StatusPanel icon={BookOpen} title="維修知識庫" tone="emerald">
      {err ? (
        <p className="text-xs text-muted-foreground">後端未連線</p>
      ) : docs === null ? (
        <p className="text-xs text-muted-foreground">載入中…</p>
      ) : (
        <>
          <p className="text-2xl font-bold tabular-nums">{docs.length}</p>
          <p className="text-xs text-muted-foreground">已收錄維修文件</p>
          <Link
            href="/servo/knowledge"
            className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
          >
            瀏覽知識庫 <ArrowRight className="h-3 w-3" />
          </Link>
        </>
      )}
    </StatusPanel>
  );
}
