"use client";

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { apiGet, API_BASE, type Health } from "@/lib/api";

type Status = "loading" | "ok" | "error";

export default function Home() {
  const [status, setStatus] = useState<Status>("loading");
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function check() {
    setStatus("loading");
    setError(null);
    try {
      setHealth(await apiGet<Health>("/health"));
      setStatus("ok");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }

  useEffect(() => {
    check();
  }, []);

  return (
    <main className="mx-auto flex min-h-full max-w-2xl flex-col justify-center gap-6 px-6 py-16">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          伺服馬達健康監測與智慧維護
        </h1>
        <p className="mt-2 text-muted-foreground">
          AI 故障風險預測與預測性維護建議系統 · Next.js 前端
        </p>
      </div>

      <div className="rounded-lg border p-5">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">後端連線狀態</span>
          <StatusBadge status={status} health={health} />
        </div>
        <p className="mt-2 break-all font-mono text-xs text-muted-foreground">
          {API_BASE}/health
        </p>
        {status === "error" && (
          <p className="mt-2 text-sm text-red-600">
            無法連線後端：{error}。請確認 uvicorn 已啟動、且 .env.local 的
            NEXT_PUBLIC_API_BASE_URL 指向正確位址。
          </p>
        )}
        {status === "ok" && health && (
          <p className="mt-2 text-sm text-muted-foreground">
            model_loaded：{String(health.model_loaded)}
          </p>
        )}
        <Button onClick={check} variant="outline" size="sm" className="mt-4">
          重新檢查
        </Button>
      </div>
    </main>
  );
}

function StatusBadge({
  status,
  health,
}: {
  status: Status;
  health: Health | null;
}) {
  const map = {
    loading: { text: "檢查中…", cls: "bg-muted text-muted-foreground" },
    ok: {
      text: health?.status === "ok" ? "已連線 · 模型就緒" : "已連線 · 模型未載入",
      cls:
        health?.status === "ok"
          ? "bg-emerald-100 text-emerald-700"
          : "bg-amber-100 text-amber-700",
    },
    error: { text: "未連線", cls: "bg-red-100 text-red-700" },
  }[status];

  return (
    <span className={`rounded-full px-3 py-1 text-xs font-medium ${map.cls}`}>
      {map.text}
    </span>
  );
}
