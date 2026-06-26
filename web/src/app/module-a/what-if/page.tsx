"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import { TrendChart } from "@/components/dashboard/TrendChart";
import { Note, PageTitle } from "@/components/ui-kit";
import { apiPost } from "@/lib/api";

interface PredictReq {
  type: "L" | "M" | "H";
  air_temperature_K: number;
  process_temperature_K: number;
  rotational_speed_rpm: number;
  torque_Nm: number;
  tool_wear_min: number;
}
interface BatchResp {
  count: number;
  results: { failure_probability: number }[];
}

const BASE: PredictReq = {
  type: "L",
  air_temperature_K: 298.1,
  process_temperature_K: 308.6,
  rotational_speed_rpm: 1551,
  torque_Nm: 42.8,
  tool_wear_min: 108,
};

const SWEEPS: Record<
  string,
  { key: keyof PredictReq; label: string; from: number; to: number; step: number; unit: string }
> = {
  tool_wear_min: { key: "tool_wear_min", label: "刀具磨耗", from: 0, to: 250, step: 10, unit: "min" },
  torque_Nm: { key: "torque_Nm", label: "扭矩", from: 10, to: 80, step: 3, unit: "Nm" },
  rotational_speed_rpm: { key: "rotational_speed_rpm", label: "轉速", from: 1200, to: 2800, step: 64, unit: "rpm" },
};

export default function ModuleAWhatIfPage() {
  const [varKey, setVarKey] = useState<keyof typeof SWEEPS>("tool_wear_min");
  const [data, setData] = useState<{ x: number; p: number }[]>([]);
  const [busy, setBusy] = useState(true);
  const [err, setErr] = useState(false);

  useEffect(() => {
    const sw = SWEEPS[varKey];
    const xs: number[] = [];
    for (let v = sw.from; v <= sw.to; v += sw.step) xs.push(Number(v.toFixed(2)));
    const records: PredictReq[] = xs.map((v) => ({ ...BASE, [sw.key]: v }));
    apiPost<BatchResp>("/predict/batch", records)
      .then((r) => {
        setErr(false);
        setData(xs.map((x, i) => ({ x, p: +(r.results[i].failure_probability * 100).toFixed(2) })));
      })
      .catch(() => setErr(true))
      .finally(() => setBusy(false));
  }, [varKey]);

  const sw = SWEEPS[varKey];

  return (
    <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 A · What-if 敏感度分析"
        desc="固定其餘條件，掃描單一變數，觀察故障機率如何隨之變化（一次批次預測取代多次單點呼叫）。"
      />
      <Note tone="warn" className="mb-6">AI4I 2020 合成資料；其餘條件固定於典型值。</Note>

      <div className="mb-4 rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
        <label className="block max-w-xs">
          <span className="mb-1 block text-xs text-muted-foreground">掃描變數</span>
          <select
            value={varKey}
            onChange={(e) => setVarKey(e.target.value as keyof typeof SWEEPS)}
            className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
          >
            {Object.entries(SWEEPS).map(([k, v]) => (
              <option key={k} value={k}>{v.label}（{v.unit}）</option>
            ))}
          </select>
        </label>
      </div>

      {err ? (
        <Note tone="danger">批次預測失敗，請確認後端已啟動。</Note>
      ) : (
        <div className="rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-semibold">故障機率 vs {sw.label}（{sw.unit}）</span>
            {busy && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          </div>
          <TrendChart data={data} dataKey="p" xKey="x" unit="%" height={260} color="var(--chart-4)" />
        </div>
      )}
    </div>
  );
}
