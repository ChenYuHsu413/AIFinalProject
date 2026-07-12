"use client";

import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";

import { Card, Note, PageTitle, Stat } from "@/components/ui-kit";
import { Badge } from "@/components/ui/badge";
import { apiGet, type MlopsEvent, type MlopsStatus } from "@/lib/api";

// Causal-chain event styling — mirrors the Streamlit _EVENT_STYLE terminology.
const EVENT_STYLE: Record<string, { color: string; label: string }> = {
  drift_detected: { color: "#a855f7", label: "🟣 資料漂移偵測" },
  drift_cleared: { color: "#14b8a6", label: "🟦 漂移解除" },
  retrain_started: { color: "#3b82f6", label: "🔄 觸發自動重訓" },
  retrain_finished: { color: "#6366f1", label: "✅ 重訓完成" },
  retrain_error: { color: "#ef4444", label: "❌ 重訓失敗" },
};

const GATE_ICON: Record<string, string> = {
  PASS: "✅",
  FAIL: "❌",
  SKIP: "⏭",
  BLOCKED: "🚫",
};

function eventDetail(e: MlopsEvent): string {
  switch (e.type) {
    case "drift_detected":
      return `重建誤差 ${e.rolling_recon_error ?? "—"} > 基線 P95 ${e.baseline_p95 ?? "—"}（資料不像訓練分布；退化不會觸發）`;
    case "drift_cleared":
      return "訊號回落基線內（漂移已被重訓消化或情境結束）";
    case "retrain_started":
      return `模式=${e.mode ?? "—"}；納入漂移工況資料重訓（背景執行，不阻塞監控）`;
    case "retrain_finished":
      return `驗證閘門 ${e.gate_passed ? "PASS" : "FAIL"}${
        e.new_version ? ` → 新版本 ${e.new_version}` : "（dry-run，未切換）"
      }`;
    case "retrain_error":
      return e.error ?? "";
    default:
      return "";
  }
}

export default function ServoMlopsPage() {
  const [data, setData] = useState<MlopsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      setData(await apiGet<MlopsStatus>("/servo/mlops"));
    } catch {
      setErr("讀取 MLOps 狀態失敗——請確認後端已啟動，且模型 registry 已遷移。");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // load() sets loading state synchronously — expected for a fetch-on-mount.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  const reg = data?.registry;
  const gate = data?.gate_report ?? null;
  const timeline = data?.timeline ?? [];

  return (
    <div className="space-y-6">
      <PageTitle
        title="MLOps 狀態面板"
        desc="模型版本 registry、驗證閘門結果、漂移→重訓→切版因果鏈——全唯讀。"
      />

      <div className="flex items-center gap-3">
        <button
          onClick={load}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-md bg-muted px-3 py-1.5 text-sm font-medium text-foreground hover:bg-muted/70 disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          重新整理
        </button>
        {reg?.updated && (
          <span className="text-xs text-muted-foreground">
            registry 更新於 {reg.updated}
          </span>
        )}
      </div>

      <Note tone="info">
        本頁為<strong>唯讀狀態面板</strong>，不提供任何觸發按鈕——重訓／切版 demo 由{" "}
        <code>scripts/run_drift_demo.py</code> 驅動，避免線上誤觸。
      </Note>

      {err && <Note tone="danger">{err}</Note>}

      {/* 1. Registry version history */}
      <Card title="模型版本 Registry（版本歷史）">
        {!reg ? (
          <p className="text-sm text-muted-foreground">{loading ? "載入中…" : "—"}</p>
        ) : (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <span>
                active 版本：
                <strong className="text-foreground">{reg.active_version ?? "—"}</strong>
              </span>
              <span
                className={
                  reg.outputs_consistent === false
                    ? "text-amber-600 dark:text-amber-400"
                    : "text-muted-foreground"
                }
              >
                outputs/models 與 active CRC32：
                {reg.outputs_consistent === null
                  ? "無法比對"
                  : reg.outputs_consistent
                    ? "一致 ✓"
                    : "不一致（outputs/models 已過時，部署以 registry 為準）"}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/60 text-left text-xs text-muted-foreground">
                    <th className="py-2 pr-3">版本</th>
                    <th className="py-2 pr-3">macro-F1</th>
                    <th className="py-2 pr-3">DV R²</th>
                    <th className="py-2 pr-3">特徵組</th>
                    <th className="py-2 pr-3">評估</th>
                    <th className="py-2 pr-3">CRC32（clf / reg）</th>
                    <th className="py-2 pr-3">備註</th>
                  </tr>
                </thead>
                <tbody>
                  {reg.versions.map((v) => (
                    <tr key={v.version} className="border-b border-border/40 align-top">
                      <td className="py-2 pr-3 font-medium">
                        <span className="flex items-center gap-1.5">
                          {v.version}
                          {v.active && <Badge variant="success">active</Badge>}
                          {v.placeholder && <Badge variant="warning">placeholder</Badge>}
                        </span>
                      </td>
                      <td className="py-2 pr-3 tabular-nums">
                        {v.macro_f1 != null ? v.macro_f1.toFixed(4) : "—"}
                      </td>
                      <td className="py-2 pr-3 tabular-nums">
                        {v.dv_r2 != null ? v.dv_r2.toFixed(4) : "—"}
                      </td>
                      <td className="py-2 pr-3">{v.feature_set ?? "—"}</td>
                      <td className="py-2 pr-3 text-xs text-muted-foreground">
                        {v.eval_mode ?? "—"}
                      </td>
                      <td className="py-2 pr-3 font-mono text-xs text-muted-foreground">
                        {v.crc32
                          ? `${v.crc32["servo_clf.joblib"] ?? "—"} / ${v.crc32["servo_reg.joblib"] ?? "—"}`
                          : "—"}
                      </td>
                      <td className="py-2 pr-3 text-xs text-muted-foreground">
                        {v.note ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>

      {/* 2. Latest validation gate report */}
      <Card title="最近一次驗證閘門（gate_report）">
        {gate ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <Badge variant={gate.passed ? "success" : "danger"}>
                {gate.passed ? "PASS ✅" : "FAIL ❌"}
              </Badge>
              <span className="text-xs text-muted-foreground">
                候選 {gate._source ?? gate.candidate_dir} · active 基準 {gate.active_version ?? "—"} ·{" "}
                {gate.created}
              </span>
            </div>
            <ul className="space-y-1.5">
              {gate.checks.map((c) => (
                <li
                  key={c.name}
                  className="rounded-md border border-border/50 bg-muted/20 px-3 py-2 text-sm"
                >
                  <span className="font-medium">
                    {GATE_ICON[c.status] ?? "•"} {c.name}
                  </span>
                  <span className="ml-2 text-xs text-muted-foreground">{c.status}</span>
                  <p className="mt-0.5 text-xs text-muted-foreground">{c.detail}</p>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <Note tone="info">
            尚無 gate 紀錄。目前的 <code>v1</code> 由初始部署直接遷移（未經閘門）；跑{" "}
            <code>run_drift_demo.py</code> 產生候選後，此處會顯示最近一次 PASS/FAIL 明細。
          </Note>
        )}
      </Card>

      {/* 3. Drift → retrain → promote causal timeline */}
      <Card title="漂移 → 重訓 → 切版 因果鏈時間線（最新在上）">
        {timeline.length === 0 ? (
          <Note tone="info">
            尚未觸發閉環。跑 <code>scripts/run_drift_demo.py</code> 後，drift → retrain →
            gate → （切版）事件會依序出現在此。
          </Note>
        ) : (
          <ul className="space-y-2">
            {timeline.map((e, i) => {
              const style = EVENT_STYLE[e.type] ?? { color: "#94a3b8", label: e.type };
              return (
                <li
                  // ids restart each demo run (retrain-0001…), so they repeat across
                  // runs in the accumulated log — combine with position for a unique key.
                  key={`${e.id}-${e.ts ?? ""}-${i}`}
                  className="rounded-md border-l-4 bg-muted/30 px-3 py-2"
                  style={{ borderLeftColor: style.color }}
                >
                  <div className="flex flex-wrap items-center gap-2 text-sm">
                    <span className="font-semibold" style={{ color: style.color }}>
                      {style.label}
                    </span>
                    <span className="text-xs text-muted-foreground">{e.id}</span>
                    {e.trigger && (
                      <span className="text-xs text-muted-foreground">
                        ← 由 {e.trigger} 觸發
                      </span>
                    )}
                    {e.ts && (
                      <span className="ml-auto text-xs tabular-nums text-muted-foreground">
                        {e.ts}
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-xs text-muted-foreground">{eventDetail(e)}</p>
                </li>
              );
            })}
          </ul>
        )}
      </Card>

      {reg && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="已登記版本" value={String(reg.versions.length)} />
          <Stat label="active 版本" value={reg.active_version ?? "—"} />
          <Stat
            label="最近閘門"
            value={gate ? (gate.passed ? "PASS" : "FAIL") : "—"}
            valueClass={gate ? (gate.passed ? "text-emerald-500" : "text-red-500") : undefined}
          />
          <Stat label="因果鏈事件" value={String(timeline.length)} />
        </div>
      )}
    </div>
  );
}
