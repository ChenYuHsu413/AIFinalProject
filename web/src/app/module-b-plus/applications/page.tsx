"use client";

import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Note, PageTitle } from "@/components/ui-kit";
import { apiGet } from "@/lib/api";

interface Curve {
  condition: string;
  bearing: string;
  life_pct: number[];
  hi: number[];
}
interface Overlay {
  available: boolean;
  curves: Curve[];
}

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

export default function ModuleBPlusApplicationsPage() {
  const [ov, setOv] = useState<Overlay | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    apiGet<Overlay>("/xjtu/health_overlay").then(setOv).catch(() => setErr(true));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
      <PageTitle
        title="模組 B+ · 延伸應用（XJTU-SY）"
        desc="多軸承健康指標（HI）對齊壽命百分比的重疊圖：觀察不同軸承的退化軌跡形狀與離散度。"
      />
      <Note tone="info" className="mb-6">
        每條曲線為一顆軸承；x 軸為壽命百分比（0→失效），y 軸為健康指標。多軌跡重疊有助於評估泛化的可行性與限制。
      </Note>
      {err && <Note tone="danger" className="mb-6">無法載入 HI 重疊圖，請確認後端已啟動。</Note>}

      {ov && ov.available && (
        <div className="rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm backdrop-blur-sm">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-semibold">HI 重疊圖（{ov.curves.length} 顆軸承）</span>
            <span className="text-[11px] text-muted-foreground">x：壽命 % · y：健康指標</span>
          </div>
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ top: 8, right: 8, bottom: 4, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" vertical={false} />
                <XAxis
                  type="number"
                  dataKey="x"
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                  tick={{ fontSize: 11, fill: "currentColor" }}
                  className="text-muted-foreground"
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "currentColor" }}
                  className="text-muted-foreground"
                  tickLine={false}
                  axisLine={false}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--popover)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "var(--popover-foreground)",
                  }}
                />
                {ov.curves.map((c, i) => (
                  <Line
                    key={c.bearing}
                    type="monotone"
                    dataKey="y"
                    data={c.life_pct.map((x, j) => ({ x, y: c.hi[j] }))}
                    name={c.bearing}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={1.5}
                    strokeOpacity={0.55}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {ov && !ov.available && (
        <Note tone="info">
          HI 重疊圖需<b>原始振動資料</b>即時重算；雲端 demo 未打包該資料（約 21 GB）。
          多軌跡泛化的彙整指標見「多軌跡泛化」頁（由已提交結果呈現）；完整重疊圖請在本機
          （下載資料後）執行。
        </Note>
      )}
    </div>
  );
}
