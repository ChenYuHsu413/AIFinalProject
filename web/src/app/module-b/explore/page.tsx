"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

import { TrendChart } from "@/components/dashboard/TrendChart";
import { Note, PageTitle } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

interface ImsHi {
  available: boolean;
  indicator: string;
  candidates: string[];
  fpt_index: number;
  lead_time_days: number;
  alarm_health: number;
  points: { timestamp: string; health: number }[];
}

export default function ModuleBExplorePage() {
  const [hi, setHi] = useState<ImsHi | null>(null);
  const [indicator, setIndicator] = useState<string>("");
  const [busy, setBusy] = useState(true);
  const [err, setErr] = useState(false);
  // request sequence — rapid indicator switches must not let an older, slower
  // response overwrite the newer one.
  const seqRef = useRef(0);

  function load(ind?: string) {
    const seq = ++seqRef.current;
    setBusy(true);
    setErr(false);
    const q = ind ? `?indicator=${encodeURIComponent(ind)}` : "";
    apiGet<ImsHi>(`/ims/health_indicator${q}`)
      .then((h) => {
        if (seq !== seqRef.current) return;
        setHi(h);
        setIndicator(h.indicator);
      })
      .catch(() => {
        if (seq === seqRef.current) setErr(true);
      })
      .finally(() => {
        if (seq === seqRef.current) setBusy(false);
      });
  }

  // initial load (set state only inside async callbacks; load() is for the picker)
  useEffect(() => {
    const seq = ++seqRef.current;
    apiGet<ImsHi>("/ims/health_indicator")
      .then((h) => {
        if (seq !== seqRef.current) return;
        setHi(h);
        setIndicator(h.indicator);
      })
      .catch(() => {
        if (seq === seqRef.current) setErr(true);
      })
      .finally(() => {
        if (seq === seqRef.current) setBusy(false);
      });
  }, []);

  const curve =
    hi?.points?.map((p) => ({ t: p.timestamp.slice(5, 16), health: p.health })) ?? [];

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 B · 互動探索（IMS）"
        desc="切換不同健康指標，比較各自的健康曲線與退化起點（FPT）。"
      />
      <Note tone="warn" className="mb-6">單軌跡 IMS Set 2；指標切換僅供觀察，不代表可泛化。</Note>
      {err && <Note tone="danger" className="mb-6">無法載入指標，請確認後端已啟動。</Note>}

      {hi && !hi.available && (
        <Note tone="info">
          互動健康指標需<b>原始振動資料</b>即時重算；雲端 demo 未打包該資料（約 21 GB）。
          完整版（可切換指標、即時重算 FPT）請在本機（下載資料後）執行。
        </Note>
      )}

      {hi && hi.available && (
        <>
          <div className="mb-4 flex flex-wrap items-end gap-3 rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
            <label className="block">
              <span className="mb-1 block text-xs text-muted-foreground">健康指標</span>
              <select
                value={indicator}
                onChange={(e) => load(e.target.value)}
                disabled={busy}
                className="rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
              >
                {hi.candidates.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <div className="text-xs text-muted-foreground">
              FPT @ #{hi.fpt_index} · 提前預警 {hi.lead_time_days.toFixed(1)} 天 · 告警線 {hi.alarm_health}
            </div>
            {busy && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          </div>

          <div className="rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
            <span className="text-sm font-semibold">健康曲線（{hi.indicator}）</span>
            <TrendChart data={curve} dataKey="health" xKey="t" height={280} color="var(--chart-1)" />
          </div>
        </>
      )}
    </div>
  );
}
