"use client";

import { useEffect, useState } from "react";
import { FileText, Loader2, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, Note, PageTitle } from "@/components/ui-kit";
import {
  apiGet,
  type KnowledgeDoc,
  type KnowledgeHit,
} from "@/lib/api";

export default function KnowledgePage() {
  const [docs, setDocs] = useState<KnowledgeDoc[]>([]);
  const [loadErr, setLoadErr] = useState(false);

  const [q, setQ] = useState("位置誤差 變大 卡滯");
  const [hits, setHits] = useState<KnowledgeHit[] | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        setDocs(await apiGet<KnowledgeDoc[]>("/knowledge/documents"));
      } catch {
        setLoadErr(true);
      }
    })();
  }, []);

  async function searchKb() {
    if (!q.trim()) return;
    setBusy(true);
    try {
      setHits(
        await apiGet<KnowledgeHit[]>(
          `/knowledge/search?q=${encodeURIComponent(q)}&top_k=5`,
        ),
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <PageTitle
        title="維修知識庫 / RAG"
        desc="伺服馬達與滾珠螺桿維修知識庫，支援關鍵字檢索（TF-IDF），可依模型異常特徵檢索相關片段供 LLM 引用。離線可用。"
      />

      {loadErr && <Note tone="danger">無法載入知識庫文件，請確認後端已啟動。</Note>}

      <p className="mb-3 text-sm text-muted-foreground">
        目前收錄 {docs.length} 份離線知識文件。
      </p>
      <div className="mb-6 grid gap-3 sm:grid-cols-2">
        {docs.map((d) => (
          <div
            key={d.source}
            className="flex items-start gap-3 rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm"
          >
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-violet-500/15 text-violet-300 ring-1 ring-inset ring-violet-500/30">
              <FileText className="h-5 w-5" />
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{d.title}</p>
              <p className="truncate text-xs text-muted-foreground">
                {d.source} · {d.chars} 字
              </p>
            </div>
          </div>
        ))}
      </div>

      <Card title="關鍵字檢索（TF-IDF）">
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && searchKb()}
            placeholder="輸入症狀或關鍵字…"
            className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
          />
          <Button onClick={searchKb} disabled={busy}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            檢索
          </Button>
        </div>

        {hits && hits.length === 0 && (
          <Note tone="warn" className="mt-4">
            沒有檢索到相關片段，換個關鍵字試試。
          </Note>
        )}

        {hits && hits.length > 0 && (
          <div className="mt-4 space-y-2">
            {hits.map((h, i) => (
              <details key={i} className="group rounded-lg border border-border/70 bg-card/70 backdrop-blur-sm">
                <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2.5 text-sm">
                  <span className="font-medium">{h.title || h.source}</span>
                  <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-300 ring-1 ring-inset ring-emerald-500/30">
                    相關度 {h.score}
                  </span>
                  <span className="ml-auto text-muted-foreground transition-transform group-open:rotate-90">
                    ›
                  </span>
                </summary>
                <p className="border-t border-border/70 px-4 py-3 text-sm leading-relaxed text-muted-foreground">
                  {h.text}
                </p>
              </details>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
