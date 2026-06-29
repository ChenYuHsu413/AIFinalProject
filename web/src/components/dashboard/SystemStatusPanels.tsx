"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, ArrowRight, BookOpen, Bot, ShieldCheck } from "lucide-react";

import {
  apiGet,
  type AssistantProviders,
  type KnowledgeDoc,
  type ServoModelInfo,
  type ServoReferenceMetrics,
} from "@/lib/api";
import { HealthBadge } from "./badges";
import { Skeleton } from "./skeletons";

/**
 * Real-API system status islands (reference model · LLM assistant · knowledge
 * base · latest prediction). Moved out of the homepage so the page file stays an
 * orchestrator; behaviour is unchanged from the original inline panels.
 */
export function SystemStatusPanels({
  worst,
  loading = false,
}: {
  worst: { name: string; state: "LN" | "LO" | "MED" | "HI"; score: number };
  /** First visit with no cached fleet — skeleton the worst-unit panel instead
   *  of showing the mock placeholder. */
  loading?: boolean;
}) {
  return (
    <section className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
      <LatestPredictionPanel
        unit={worst.name}
        state={worst.state}
        score={worst.score}
        loading={loading}
      />
      <ReferenceModelPanel />
      <AssistantPanel />
      <KnowledgePanel />
    </section>
  );
}

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
    cyan: "text-cyan-700 dark:text-cyan-300 bg-cyan-500/15 ring-cyan-500/30",
    emerald: "text-emerald-700 dark:text-emerald-300 bg-emerald-500/15 ring-emerald-500/30",
    amber: "text-amber-700 dark:text-amber-300 bg-amber-500/15 ring-amber-500/30",
    violet: "text-violet-700 dark:text-violet-300 bg-violet-500/15 ring-violet-500/30",
  }[tone];
  return (
    <div className="rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm">
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
  loading = false,
}: {
  unit: string;
  state: "LN" | "LO" | "MED" | "HI";
  score: number;
  loading?: boolean;
}) {
  return (
    <StatusPanel icon={Activity} title="最新預測" tone="violet">
      {loading ? (
        <>
          <Skeleton className="h-4 w-24" />
          <Skeleton className="mt-2 h-5 w-32 rounded-full" />
          <Skeleton className="mt-2.5 h-3 w-40" />
        </>
      ) : (
        <>
          <p className="text-sm font-semibold">{unit}</p>
          <div className="mt-1.5 flex items-center gap-2">
            <HealthBadge state={state} />
            <span className="text-xs text-muted-foreground">分數 {score}/100</span>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">最近一次估測為機群最低分設備</p>
        </>
      )}
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
            <p className="mt-2 inline-block rounded bg-amber-500/15 px-1.5 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-300 ring-1 ring-inset ring-amber-500/30">
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
            <span className={`h-2 w-2 rounded-full ${online ? "bg-emerald-400" : "bg-amber-400"}`} />
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
