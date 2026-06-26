"use client";

import { useState } from "react";
import { Loader2, Target, Wrench } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, Note, PageTitle, Stat } from "@/components/ui-kit";
import { apiPost } from "@/lib/api";
import { RISK_COLOR, RISK_ZH } from "@/lib/servo";

interface PredictReq {
  type: "L" | "M" | "H";
  air_temperature_K: number;
  process_temperature_K: number;
  rotational_speed_rpm: number;
  torque_Nm: number;
  tool_wear_min: number;
}
interface PredictResp {
  failure_probability: number;
  predicted_class: number;
  health_score: number;
  risk_level: "Low" | "Medium" | "High";
  maintenance_advice: string[];
}

const DEFAULTS: PredictReq = {
  type: "L",
  air_temperature_K: 298.1,
  process_temperature_K: 308.6,
  rotational_speed_rpm: 1551,
  torque_Nm: 42.8,
  tool_wear_min: 108,
};

const FIELDS: { key: keyof PredictReq; label: string; step?: number }[] = [
  { key: "air_temperature_K", label: "氣溫 (K)", step: 0.1 },
  { key: "process_temperature_K", label: "製程溫度 (K)", step: 0.1 },
  { key: "rotational_speed_rpm", label: "轉速 (rpm)", step: 1 },
  { key: "torque_Nm", label: "扭矩 (Nm)", step: 0.1 },
  { key: "tool_wear_min", label: "刀具磨耗 (min)", step: 1 },
];

export default function ModuleAPredictPage() {
  const [form, setForm] = useState<PredictReq>(DEFAULTS);
  const [res, setRes] = useState<PredictResp | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(false);

  async function predict() {
    setBusy(true);
    setErr(false);
    try {
      setRes(await apiPost<PredictResp>("/predict", form));
    } catch {
      setErr(true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 A · 手動單筆預測"
        desc="輸入一組運轉條件，估測故障機率、健康分數與風險等級（AI4I 2020 合成資料）。"
      />
      <Note tone="warn" className="mb-6">
        AI4I 2020 為<b>合成資料</b>，非真實伺服馬達資料；本頁為靜態風險對照模組。
      </Note>

      <div className="mb-6 rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
        <h2 className="mb-3 text-sm font-semibold">運轉條件</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <label className="block">
            <span className="mb-1 block text-xs text-muted-foreground">產品品質</span>
            <select
              value={form.type}
              onChange={(e) => setForm({ ...form, type: e.target.value as PredictReq["type"] })}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
            >
              <option value="L">L（低）</option>
              <option value="M">M（中）</option>
              <option value="H">H（高）</option>
            </select>
          </label>
          {FIELDS.map((f) => (
            <label key={f.key} className="block">
              <span className="mb-1 block text-xs text-muted-foreground">{f.label}</span>
              <input
                type="number"
                step={f.step}
                value={form[f.key] as number}
                onChange={(e) => setForm({ ...form, [f.key]: Number(e.target.value) })}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
              />
            </label>
          ))}
        </div>
        <Button onClick={predict} disabled={busy} className="mt-4">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Target className="h-4 w-4" />}
          估測風險
        </Button>
      </div>

      {err && <Note tone="danger">預測失敗，請確認後端已啟動。</Note>}

      {res && (
        <div className="space-y-6">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="故障機率" value={`${(res.failure_probability * 100).toFixed(1)}%`} valueClass="text-amber-400" />
            <Stat label="健康分數" value={res.health_score.toFixed(0)} />
            <Stat label="預測類別" value={res.predicted_class === 1 ? "故障" : "正常"} valueClass={res.predicted_class === 1 ? "text-red-400" : "text-emerald-400"} />
            <div className="rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm">
              <p className="text-xs text-muted-foreground">風險等級</p>
              <span className={`mt-1 inline-block rounded-full px-3 py-1 text-sm font-semibold ${RISK_COLOR[res.risk_level]}`}>
                {RISK_ZH[res.risk_level]} · {res.risk_level}
              </span>
            </div>
          </div>
          <Card title="建議處置">
            <ul className="space-y-2">
              {res.maintenance_advice.map((t, i) => (
                <li key={i} className="flex gap-2 rounded-lg border border-border/60 bg-muted/30 p-3 text-sm">
                  <Wrench className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  );
}
