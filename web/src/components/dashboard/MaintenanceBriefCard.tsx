"use client";

import { useState } from "react";
import Link from "next/link";
import { Bot, Check, FileDown, Sparkles } from "lucide-react";

import type { MotorView } from "@/lib/dashboard";
import { maintenanceBrief } from "@/lib/dashboard";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * AI Maintenance Brief — condenses the current fleet anomalies into one
 * operator-facing paragraph. Today it renders a deterministic, offline-safe
 * summary from the adapter; the `詢問助理` CTA links to the real LLM assistant.
 *
 * Integration point: when the streaming brief endpoint is wired, replace the
 * `brief` const with a fetch to POST /servo/assistant/report (fall back to
 * maintenanceBrief() on error so the card is never empty).
 */
export function MaintenanceBriefCard({ views }: { views: MotorView[] }) {
  const brief = maintenanceBrief(views);
  const [copied, setCopied] = useState(false);

  async function copyBrief() {
    try {
      await navigator.clipboard.writeText(`AI 維護摘要\n${brief}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard unavailable — no-op */
    }
  }

  return (
    <Card className="border-primary/30 bg-gradient-to-br from-primary/5 to-card">
      <CardHeader>
        <CardDescription className="flex items-center gap-1.5 text-primary">
          <Sparkles className="h-3.5 w-3.5" />
          AI 維護摘要 · Maintenance Brief
        </CardDescription>
        <CardTitle className="text-lg">今日維護重點</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="rounded-lg border border-border/60 bg-card/50 p-4">
          <p className="text-sm leading-relaxed text-foreground/90">{brief}</p>
          <p className="mt-2 text-[11px] text-muted-foreground/70">
            摘要由設備健康與告警自動彙整（離線 fallback）；點「詢問助理」取得即時 LLM 建議。
          </p>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <Link
            href="/servo/assistant"
            className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/80"
          >
            <Bot className="h-3.5 w-3.5" />
            詢問 LLM 助理
          </Link>
          <Link
            href="/servo/assistant"
            className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card/60 px-3 py-1.5 text-sm font-medium transition-colors hover:bg-muted"
          >
            <Sparkles className="h-3.5 w-3.5" />
            產生維修建議
          </Link>
          <button
            type="button"
            onClick={copyBrief}
            className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card/60 px-3 py-1.5 text-sm font-medium transition-colors hover:bg-muted"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-400" /> 已複製
              </>
            ) : (
              <>
                <FileDown className="h-3.5 w-3.5" /> 匯出今日摘要
              </>
            )}
          </button>
        </div>
      </CardContent>
    </Card>
  );
}
