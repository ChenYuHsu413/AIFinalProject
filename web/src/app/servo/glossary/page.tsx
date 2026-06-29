"use client";

import { useEffect, useState } from "react";

import { Card, Note, PageTitle } from "@/components/ui-kit";
import {
  apiGet,
  type GlossaryEntry,
  type ServoFeatureSets,
} from "@/lib/api";

export default function GlossaryPage() {
  const [docs, setDocs] = useState<GlossaryEntry[]>([]);
  const [featureSets, setFeatureSets] = useState<ServoFeatureSets>({});
  const [loadErr, setLoadErr] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [g, f] = await Promise.all([
          apiGet<GlossaryEntry[]>("/servo/glossary"),
          apiGet<ServoFeatureSets>("/servo/feature_sets"),
        ]);
        setDocs(g);
        setFeatureSets(f);
      } catch {
        setLoadErr(true);
      }
    })();
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <PageTitle
        title="馬達欄位解釋 / 資料教學"
        desc="把伺服馬達常見訊號（扭矩、轉速、三相電流、D/Q 軸、位置誤差…）用白話說明，並說明各特徵組的組成。"
      />

      {loadErr && <Note tone="danger">無法載入欄位資料，請確認後端已啟動。</Note>}

      <Card title="馬達訊號欄位解釋" className="mb-6 overflow-x-auto">
        <table className="w-full min-w-[680px] text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-2 pr-3">欄位</th>
              <th className="py-2 pr-3">中文</th>
              <th className="py-2 pr-3">說明</th>
              <th className="py-2 pr-3">對伺服馬達的意義</th>
              <th className="py-2">異常時可能代表</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.name} className="border-b align-top last:border-0">
                <td className="py-2 pr-3 font-mono text-xs font-medium">{d.name}</td>
                <td className="py-2 pr-3 whitespace-nowrap">{d.zh}</td>
                <td className="py-2 pr-3 text-muted-foreground">{d.desc}</td>
                <td className="py-2 pr-3 text-muted-foreground">{d.meaning}</td>
                <td className="py-2 text-muted-foreground">{d.anomaly}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <h2 className="mb-3 text-sm font-semibold">特徵組說明</h2>
      <div className="space-y-2">
        {Object.entries(featureSets).map(([key, spec]) => (
          <details key={key} className="group rounded-xl border border-border/70 bg-card/70 shadow-sm backdrop-blur-sm">
            <summary className="flex cursor-pointer list-none items-center gap-2 px-5 py-3 text-sm font-medium">
              <span className="text-violet-600 dark:text-violet-300">{spec.label}</span>
              <span className="text-xs text-muted-foreground">
                （{key}）— {spec.columns.length} 個特徵
              </span>
              <span className="ml-auto text-muted-foreground transition-transform group-open:rotate-90">
                ›
              </span>
            </summary>
            <div className="border-t border-border/70 px-5 py-3">
              <p className="text-sm text-muted-foreground">{spec.desc}</p>
              <p className="mt-2 rounded-lg bg-muted px-3 py-2 font-mono text-xs">
                {spec.columns.join(", ") || "（運動 + 電流 + 位置追隨的聯集）"}
              </p>
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}
