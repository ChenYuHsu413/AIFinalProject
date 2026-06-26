"use client";

import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import Link from "next/link";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { Card, Note, PageTitle } from "@/components/ui-kit";
import {
  apiGet,
  type ServoModelInfo,
  type ServoReferenceMetrics,
} from "@/lib/api";

export default function ReportsPage() {
  const [ref, setRef] = useState<ServoReferenceMetrics | null>(null);
  const [info, setInfo] = useState<ServoModelInfo | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [r, i] = await Promise.all([
          apiGet<ServoReferenceMetrics>("/servo/reference_metrics"),
          apiGet<ServoModelInfo>("/servo/model_info"),
        ]);
        setRef(r);
        setInfo(i);
      } catch {
        setErr(true);
      }
    })();
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <PageTitle
        title="報表中心"
        desc="參考模型評估指標與訓練結果彙整（接後端 /servo/reference_metrics）"
      />

      {err && (
        <Note tone="danger" className="mb-6">
          無法載入模型指標，請確認後端已啟動。
        </Note>
      )}

      {info?.placeholder && (
        <Note tone="warn" className="mb-6">
          目前指標來自 <b>placeholder 合成資料</b>，僅供流程展示；下載真實 PHM
          資料重訓後即為正式結果。
        </Note>
      )}

      <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-3">
        <MetricCard
          label="分類 macro-F1"
          value={ref?.clf.macro_f1?.toFixed(3) ?? "—"}
          footerMuted={info?.clf_model ?? "reference clf"}
        />
        <MetricCard
          label="回歸 R²"
          value={ref?.reg.r2?.toFixed(3) ?? "—"}
          footerMuted={info?.reg_model ?? "reference reg"}
        />
        <MetricCard
          label="回歸 MAE"
          value={ref?.reg.mae?.toFixed(3) ?? "—"}
          footerMuted="退化分數 DV 誤差"
        />
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="深度學習對照 (1D-CNN AE / MLP)">
          {ref?.dl ? (
            <dl className="space-y-2 text-sm">
              <Row
                k="MLP 分類 macro-F1"
                v={ref.dl.mlp_classification_macro_f1?.toFixed(3) ?? "—"}
              />
              <Row k="MLP 回歸 R²" v={ref.dl.mlp_regression?.r2?.toFixed(3) ?? "—"} />
              <Row k="MLP 回歸 MAE" v={ref.dl.mlp_regression?.mae?.toFixed(3) ?? "—"} />
              {ref.dl.note && (
                <p className="pt-1 text-xs text-muted-foreground">{ref.dl.note}</p>
              )}
            </dl>
          ) : (
            <p className="text-sm text-muted-foreground">載入中…</p>
          )}
        </Card>

        <Card title="特徵組合">
          <p className="text-sm text-muted-foreground">
            目前主線模型特徵組：
            <span className="ml-1 font-mono text-foreground">
              {info?.feature_set ?? "—"}
            </span>
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            共 {info?.feature_columns?.length ?? 0} 個特徵欄位
          </p>
          <Link
            href="/servo/simulator"
            className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-primary/15 px-3 py-1.5 text-xs font-medium text-primary ring-1 ring-inset ring-primary/30 transition-colors hover:bg-primary/25"
          >
            <FlaskConical className="h-3.5 w-3.5" />
            到訓練模擬器比較不同設定
          </Link>
        </Card>
      </div>

      <Note tone="info" className="mt-6">
        更多報表（時間區間彙整、設備別比較、告警統計）將於 Servo Dataset
        模組接真實資料後擴充。
      </Note>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between border-b border-border/40 pb-1.5 last:border-0">
      <span className="text-muted-foreground">{k}</span>
      <span className="font-medium tabular-nums">{v}</span>
    </div>
  );
}
