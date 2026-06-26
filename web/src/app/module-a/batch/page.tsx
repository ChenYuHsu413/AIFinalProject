"use client";

import { useState } from "react";
import { Loader2, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Note, PageTitle } from "@/components/ui-kit";
import { API_BASE } from "@/lib/api";
import { RISK_COLOR, RISK_ZH } from "@/lib/servo";

interface PredictResp {
  failure_probability: number;
  predicted_class: number;
  health_score: number;
  risk_level: "Low" | "Medium" | "High";
}
interface BatchResp {
  count: number;
  results: PredictResp[];
}

export default function ModuleABatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [res, setRes] = useState<BatchResp | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function upload() {
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API_BASE}/batch_predict`, { method: "POST", body: fd });
      if (!r.ok) {
        setErr(`上傳失敗（${r.status}）。請確認 CSV 欄位符合 AI4I 格式。`);
        return;
      }
      setRes((await r.json()) as BatchResp);
    } catch {
      setErr("上傳失敗，請確認後端已啟動。");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 A · 批次 CSV 上傳"
        desc="上傳一份 CSV（AI4I 欄位），一次取得每列的故障機率與風險等級。"
      />
      <Note tone="info" className="mb-6">
        CSV 需含欄位：<code className="font-mono">Type</code>、
        <code className="font-mono">Air temperature [K]</code>、
        <code className="font-mono">Process temperature [K]</code>、
        <code className="font-mono">Rotational speed [rpm]</code>、
        <code className="font-mono">Torque [Nm]</code>、
        <code className="font-mono">Tool wear [min]</code>。
      </Note>

      <div className="mb-6 flex flex-col gap-3 rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm sm:flex-row sm:items-center">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="flex-1 text-sm text-muted-foreground file:mr-3 file:rounded-md file:border-0 file:bg-primary/15 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-primary hover:file:bg-primary/25"
        />
        <Button onClick={upload} disabled={busy || !file}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          上傳預測
        </Button>
      </div>

      {err && <Note tone="danger" className="mb-6">{err}</Note>}

      {res && (
        <>
          <p className="mb-3 text-sm text-muted-foreground">
            共 <b className="text-foreground">{res.count}</b> 列，顯示前 50 列。
          </p>
          <div className="overflow-x-auto rounded-xl border border-border/70 bg-card/70 shadow-sm backdrop-blur-sm">
            <table className="w-full min-w-[480px] text-sm">
              <thead>
                <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th className="px-4 py-3 font-medium">#</th>
                  <th className="px-4 py-3 font-medium">故障機率</th>
                  <th className="px-4 py-3 font-medium">類別</th>
                  <th className="px-4 py-3 font-medium">健康分數</th>
                  <th className="px-4 py-3 font-medium">風險</th>
                </tr>
              </thead>
              <tbody>
                {res.results.slice(0, 50).map((r, i) => (
                  <tr key={i} className="border-b border-border/40 last:border-0 hover:bg-muted/30">
                    <td className="px-4 py-2 font-mono text-xs text-muted-foreground">{i}</td>
                    <td className="px-4 py-2 tabular-nums">{(r.failure_probability * 100).toFixed(1)}%</td>
                    <td className="px-4 py-2">{r.predicted_class === 1 ? "故障" : "正常"}</td>
                    <td className="px-4 py-2 tabular-nums">{r.health_score.toFixed(0)}</td>
                    <td className="px-4 py-2">
                      <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${RISK_COLOR[r.risk_level]}`}>
                        {RISK_ZH[r.risk_level]}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
