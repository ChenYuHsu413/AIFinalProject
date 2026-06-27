"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Bot, Loader2, MessageCircleQuestion } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { Card, Note, PageTitle, Stat } from "@/components/ui-kit";
import {
  apiGet,
  apiPost,
  type AssistantProviders,
  type AssistantResponse,
  type ServoModelInfo,
  type ServoPrediction,
  type ServoSample,
} from "@/lib/api";
import { HEALTH_ZH, RISK_ZH } from "@/lib/servo";

const PROVIDER_LABEL: Record<string, string> = {
  groq: "Groq",
  openrouter: "OpenRouter",
  gemini: "Gemini",
  anthropic: "Anthropic",
};

export default function AssistantPage() {
  const [cols, setCols] = useState<string[]>([]);
  const [samples, setSamples] = useState<ServoSample[]>([]);
  const [providers, setProviders] = useState<string[]>([]);
  const [idx, setIdx] = useState(0);
  const [pred, setPred] = useState<ServoPrediction | null>(null);
  const [predBusy, setPredBusy] = useState(false);
  const [loadErr, setLoadErr] = useState(false);

  const [report, setReport] = useState<AssistantResponse | null>(null);
  const [reportBusy, setReportBusy] = useState(false);
  const [question, setQuestion] = useState("目前狀況要先檢查什麼？");
  const [answer, setAnswer] = useState<AssistantResponse | null>(null);
  const [qaBusy, setQaBusy] = useState(false);

  // Load options, then predict the sample the dashboard handed over via
  // ?sample=<idx> (the "到 LLM 維護助理" link). Falls back to a default MED
  // sample so the page is usable when opened directly.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [info, rows, prov] = await Promise.all([
          apiGet<ServoModelInfo>("/servo/model_info"),
          apiGet<ServoSample[]>("/servo/samples"),
          apiGet<AssistantProviders>("/servo/assistant/providers"),
        ]);
        if (cancelled) return;
        setCols(info.feature_columns);
        setSamples(rows);
        setProviders(prov.providers);
        const param = new URLSearchParams(window.location.search).get("sample");
        // require a non-empty numeric param (Number("")===0 would wrongly pick #0)
        const handed = param != null && param.trim() !== "" ? Number(param) : NaN;
        const di = rows.findIndex((r) => r["ylabel"] === "MED");
        const fallback = di < 0 ? Math.floor(rows.length / 2) : di;
        const start =
          Number.isInteger(handed) && handed >= 0 && handed < rows.length
            ? handed
            : fallback;
        setIdx(start);
        await runPredict(start, info.feature_columns, rows);
      } catch {
        if (!cancelled) setLoadErr(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function runPredict(i: number, c: string[], rows: ServoSample[]) {
    if (!rows[i] || !c.length) return;
    setPredBusy(true);
    setReport(null);
    setAnswer(null);
    try {
      const features: Record<string, number> = {};
      for (const col of c) features[col] = Number(rows[i][col]);
      setPred(await apiPost<ServoPrediction>("/servo/predict", { features }));
    } finally {
      setPredBusy(false);
    }
  }

  async function genReport() {
    if (!pred) return;
    setReportBusy(true);
    try {
      setReport(
        await apiPost<AssistantResponse>("/servo/assistant/report", {
          prediction: pred,
        }),
      );
    } finally {
      setReportBusy(false);
    }
  }

  async function askQa() {
    if (!pred || !question.trim()) return;
    setQaBusy(true);
    try {
      setAnswer(
        await apiPost<AssistantResponse>("/servo/assistant/qa", {
          question,
          prediction: pred,
        }),
      );
    } finally {
      setQaBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <PageTitle
        title="LLM 維護助理"
        desc="接收模型的結構化輸出，生成保守的維修建議、可能原因、檢查項目、工單草稿與報告摘要；無 API Key 時自動使用離線範本。"
      />

      {loadErr && <Note tone="danger">無法載入助理資料，請確認後端已啟動。</Note>}

      {/* prediction context */}
      <div className="mb-4 rounded-xl border border-border/70 bg-gradient-to-br from-violet-500/10 to-card/50 p-5 shadow-sm backdrop-blur-sm">
        <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <h2 className="text-sm font-semibold">目前的模型結果（助理輸入）</h2>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            樣本
            <select
              value={idx}
              onChange={(e) => {
                const i = Number(e.target.value);
                setIdx(i);
                runPredict(i, cols, samples);
              }}
              className="rounded-lg border border-border bg-background px-2 py-1.5 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
            >
              {samples.map((s, i) => {
                const l = String(s["ylabel"] ?? "?");
                return (
                  <option key={i} value={i}>
                    #{i} · {HEALTH_ZH[l] ?? l} ({l})
                  </option>
                );
              })}
            </select>
          </label>
        </div>

        {predBusy || !pred ? (
          <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> 估測中…
          </div>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Stat label="健康狀態" value={`${pred.health_state_zh} (${pred.predicted_health_state})`} />
              <Stat label="風險等級" value={RISK_ZH[pred.risk_level] ?? pred.risk_level} />
              <Stat label="退化分數" value={pred.degradation_score.toFixed(2)} />
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              主要異常特徵：{pred.top_features.map((t) => t.feature).join("、")}
            </p>
            {pred.consistency_warning && (
              <Note tone="danger" className="mt-3">
                <AlertTriangle className="mr-1 inline h-4 w-4" />
                {pred.consistency_warning}
              </Note>
            )}
          </>
        )}
      </div>

      {/* providers */}
      {providers.length > 0 ? (
        <Note className="mb-6">
          已偵測到 LLM 供應商：
          <b>{providers.map((p) => PROVIDER_LABEL[p] ?? p).join("、")}</b>
          （依序嘗試，失敗才退回離線範本）。
        </Note>
      ) : (
        <Note className="mb-6">
          未偵測到 LLM 供應商金鑰，將使用<b>離線 fallback 範本</b>。可設定
          <code className="mx-1 rounded bg-muted px-1">GROQ_API_KEY</code>/
          <code className="mx-1 rounded bg-muted px-1">OPENROUTER_API_KEY</code>/
          <code className="mx-1 rounded bg-muted px-1">GEMINI_API_KEY</code>
          任一後改用 LLM 生成。
        </Note>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* report */}
        <Card title="生成維護報告">
          <Button onClick={genReport} disabled={!pred || reportBusy} className="w-full">
            {reportBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Bot className="h-4 w-4" />}
            產生維護報告（含工單草稿）
          </Button>
          {report && (
            <div className="mt-4">
              <SourceBadge source={report.source} />
              <Markdown text={report.text} />
            </div>
          )}
        </Card>

        {/* Q&A */}
        <Card title="維修問答">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && askQa()}
            placeholder="輸入問題…"
            className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
          />
          <Button onClick={askQa} disabled={!pred || qaBusy} variant="secondary" className="mt-2 w-full">
            {qaBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <MessageCircleQuestion className="h-4 w-4" />}
            詢問助理
          </Button>
          {answer && (
            <div className="mt-4">
              <SourceBadge source={answer.source} />
              <Markdown text={answer.text} />
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function SourceBadge({ source }: { source: string }) {
  const fallback = source === "fallback";
  return (
    <span
      className={`mb-2 inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${
        fallback
          ? "bg-slate-500/15 text-slate-300 ring-1 ring-inset ring-slate-500/30"
          : "bg-emerald-500/15 text-emerald-300 ring-1 ring-inset ring-emerald-500/30"
      }`}
    >
      {fallback ? "⚪ 離線範本" : `🟢 ${PROVIDER_LABEL[source] ?? source}`}
    </span>
  );
}

function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm prose-invert max-w-none prose-headings:font-semibold prose-p:my-2 prose-li:my-0.5">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
