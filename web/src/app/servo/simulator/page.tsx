"use client";

import { useEffect, useMemo, useState } from "react";
import { Brain, ChevronRight, Loader2, Rocket } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Bar, Card, Note, PageTitle, Stat } from "@/components/ui-kit";
import {
  apiGet,
  apiPost,
  type ServoFeatureSets,
  type ServoModelInfo,
  type ServoReferenceMetrics,
  type ServoSimResult,
  type ServoSimulateOptions,
} from "@/lib/api";
import { HEALTH_ZH } from "@/lib/servo";
import { cn } from "@/lib/utils";

const SIZES = [100, 500, 1000, 5000];

export default function SimulatorPage() {
  const [opts, setOpts] = useState<ServoSimulateOptions | null>(null);
  const [featureSets, setFeatureSets] = useState<ServoFeatureSets>({});
  const [ref, setRef] = useState<ServoReferenceMetrics | null>(null);
  const [placeholder, setPlaceholder] = useState(false);
  const [loadErr, setLoadErr] = useState(false);

  const [n, setN] = useState(500);
  const [task, setTask] = useState<"clf" | "reg">("clf");
  const [fs, setFs] = useState("engineered");
  const [algo, setAlgo] = useState("");

  const [res, setRes] = useState<ServoSimResult | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [o, f, r, m] = await Promise.all([
          apiGet<ServoSimulateOptions>("/servo/simulate/options"),
          apiGet<ServoFeatureSets>("/servo/feature_sets"),
          apiGet<ServoReferenceMetrics>("/servo/reference_metrics"),
          apiGet<ServoModelInfo>("/servo/model_info"),
        ]);
        setOpts(o);
        setFeatureSets(f);
        setRef(r);
        setPlaceholder(!!m.placeholder);
      } catch {
        setLoadErr(true);
      }
    })();
  }, []);

  const algos = useMemo(
    () => (task === "clf" ? opts?.classifiers : opts?.regressors) ?? [],
    [task, opts],
  );

  // keep algo valid whenever the task (and thus the list) changes
  useEffect(() => {
    if (algos.length && !algos.includes(algo)) setAlgo(algos[0]);
  }, [algos, algo]);

  async function train() {
    if (!algo) return;
    setBusy(true);
    try {
      setRes(
        await apiPost<ServoSimResult>("/servo/simulate", {
          task,
          feature_set: fs,
          algo,
          n,
        }),
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <PageTitle
        title="AI 訓練模擬器"
        desc="選資料量 / 特徵組 / 演算法，在後端即時訓練小模型，比較訓練時間、指標，並對照離線 Reference Model。"
      />

      {loadErr && <Note tone="danger">無法載入模擬器選項，請確認後端已啟動。</Note>}
      {placeholder && (
        <Note tone="warn" className="mb-4">
          目前模型以 <b>placeholder 合成資料</b> 訓練，僅供流程展示。
        </Note>
      )}

      {/* controls */}
      <div className="mb-4 rounded-xl border bg-gradient-to-br from-emerald-50 to-white p-5 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold">選擇訓練設定</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="資料量">
            <Select value={n} onChange={(v) => setN(Number(v))}>
              {SIZES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="任務">
            <Select value={task} onChange={(v) => setTask(v as "clf" | "reg")}>
              <option value="clf">分類（健康狀態）</option>
              <option value="reg">回歸（退化值 DV）</option>
            </Select>
          </Field>
          <Field label="特徵組">
            <Select value={fs} onChange={setFs}>
              {Object.entries(featureSets).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.label}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="演算法">
            <Select value={algo} onChange={setAlgo}>
              {algos.map((a) => (
                <option key={a} value={a}>
                  {opts?.algo_labels[a] ?? a}
                </option>
              ))}
            </Select>
          </Field>
        </div>
        {featureSets[fs] && (
          <p className="mt-3 text-xs text-muted-foreground">
            特徵組「{featureSets[fs].label}」：{featureSets[fs].desc}
          </p>
        )}
        <Button onClick={train} disabled={busy || !algo} className="mt-4 w-full">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4" />}
          開始訓練（後端小模型）
        </Button>
      </div>

      <DlPanel dl={ref?.dl} />

      {res ? (
        <div className="mt-6">
          {res.confusion_matrix ? (
            <ClfResult res={res} ref_={ref?.clf} algoLabel={opts?.algo_labels[res.algo] ?? res.algo} />
          ) : (
            <RegResult res={res} ref_={ref?.reg} algoLabel={opts?.algo_labels[res.algo] ?? res.algo} />
          )}
          {res.explanation?.length > 0 && (
            <Card title="為什麼會這樣？" className="mt-6">
              <ul className="space-y-1.5 text-sm text-muted-foreground">
                {res.explanation.map((e, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-primary">•</span>
                    <span>{e}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      ) : (
        <Note className="mt-6">選好設定後按「開始訓練」，比較小模型與離線 Reference Model 的差異。</Note>
      )}
    </div>
  );
}

function ClfResult({
  res,
  ref_,
  algoLabel,
}: {
  res: ServoSimResult;
  ref_?: ServoReferenceMetrics["clf"];
  algoLabel: string;
}) {
  const refF1 = ref_?.macro_f1;
  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="資料量" value={String(res.n_samples)} />
        <Stat label="特徵數" value={String(res.n_features)} />
        <Stat label="Accuracy" value={res.accuracy!.toFixed(3)} />
        <Stat label="Macro-F1" value={res.macro_f1!.toFixed(3)} valueClass="text-primary" />
      </div>
      <p className="text-xs text-muted-foreground">
        訓練時間 {res.train_time_s.toFixed(3)} 秒 · 演算法 {algoLabel}
      </p>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="混淆矩陣（測試集 vs 真實標籤）">
          <ConfusionMatrix labels={res.labels!} cm={res.confusion_matrix!} />
        </Card>
        <Card title="小模型 vs Reference Model">
          <div className="space-y-3">
            <Bar label="你的小模型" right={res.macro_f1!.toFixed(3)} value={res.macro_f1!} colorClass="bg-primary" />
            {refF1 != null && (
              <Bar
                label={`Reference（${ref_?.model ?? "?"}）`}
                right={refF1.toFixed(3)}
                value={refF1}
                colorClass="bg-slate-400"
              />
            )}
          </div>
          {refF1 != null && (
            <p className="mt-3 text-xs text-muted-foreground">
              Reference 以完整資料離線訓練（macro-F1 {refF1.toFixed(3)}）；你的小模型差距約{" "}
              {(refF1 - res.macro_f1!).toFixed(3)}。資料越多、特徵越貼切，差距越小。
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}

function RegResult({
  res,
  ref_,
  algoLabel,
}: {
  res: ServoSimResult;
  ref_?: ServoReferenceMetrics["reg"];
  algoLabel: string;
}) {
  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="資料量" value={String(res.n_samples)} />
        <Stat label="MAE" value={res.mae!.toFixed(3)} />
        <Stat label="RMSE" value={res.rmse!.toFixed(3)} />
        <Stat label="R²" value={res.r2!.toFixed(3)} valueClass="text-primary" />
      </div>
      <p className="text-xs text-muted-foreground">
        訓練時間 {res.train_time_s.toFixed(3)} 秒 · 演算法 {algoLabel}
      </p>
      <Card title="小模型 vs Reference Model">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-1.5">模型</th>
              <th className="py-1.5">R²</th>
              <th className="py-1.5">MAE</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b">
              <td className="py-1.5 font-medium">你的小模型</td>
              <td>{res.r2!.toFixed(3)}</td>
              <td>{res.mae!.toFixed(3)}</td>
            </tr>
            {ref_?.r2 != null && (
              <tr>
                <td className="py-1.5">Reference（{ref_.model ?? "?"}）</td>
                <td>{ref_.r2.toFixed(3)}</td>
                <td>{ref_.mae?.toFixed(3) ?? "—"}</td>
              </tr>
            )}
          </tbody>
        </table>
        {ref_?.r2 != null && (
          <p className="mt-3 text-xs text-muted-foreground">
            Reference（離線、完整資料）R²={ref_.r2.toFixed(3)}。小模型資料量小，R² 通常較低、MAE 較高。
          </p>
        )}
      </Card>
    </div>
  );
}

function ConfusionMatrix({ labels, cm }: { labels: string[]; cm: number[][] }) {
  const max = Math.max(1, ...cm.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="p-1 text-muted-foreground">真\預</th>
            {labels.map((l) => (
              <th key={l} className="p-1 font-medium text-muted-foreground">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <td className="p-1 font-medium text-muted-foreground">{labels[i]}</td>
              {row.map((v, j) => {
                const a = v / max;
                return (
                  <td
                    key={j}
                    className="h-9 w-9 rounded text-center font-medium"
                    style={{
                      backgroundColor: `rgba(79,70,229,${a * 0.85})`,
                      color: a > 0.5 ? "white" : "#334155",
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted-foreground">列＝真實標籤，欄＝模型預測。對角線越集中越好。</p>
    </div>
  );
}

function DlPanel({ dl }: { dl?: ServoReferenceMetrics["dl"] }) {
  const [open, setOpen] = useState(false);
  if (!dl || dl.mlp_classification_macro_f1 == null) return null;
  const rec = dl.reconstruction_error_by_class ?? {};
  const recOrder = ["LN", "LO", "MED", "HI"].filter((l) => l in rec);
  const recMax = Math.max(1e-9, ...recOrder.map((l) => rec[l]));

  return (
    <div className="rounded-xl border bg-white shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-5 py-3 text-sm font-medium"
      >
        <Brain className="h-4 w-4 text-fuchsia-500" />
        深度學習離線結果（唯讀）
        <ChevronRight className={cn("ml-auto h-4 w-4 transition-transform", open && "rotate-90")} />
      </button>
      {open && (
        <div className="border-t px-5 py-4">
          {dl.note && <p className="mb-3 text-xs text-muted-foreground">{dl.note}</p>}
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="MLP 分類 macro-F1" value={dl.mlp_classification_macro_f1.toFixed(3)} />
            <Stat label="MLP 回歸 R²" value={(dl.mlp_regression?.r2 ?? 0).toFixed(3)} />
            <Stat label="MLP 回歸 MAE" value={(dl.mlp_regression?.mae ?? 0).toFixed(3)} />
          </div>
          {recOrder.length > 0 && (
            <div className="mt-4">
              <p className="mb-2 text-xs text-muted-foreground">
                PCA 重建誤差（以健康資料擬合）— 退化越嚴重，重建誤差應越大：
              </p>
              <div className="space-y-2">
                {recOrder.map((l) => (
                  <Bar
                    key={l}
                    label={`${HEALTH_ZH[l]} (${l})`}
                    right={rec[l].toFixed(2)}
                    value={rec[l] / recMax}
                    colorClass="bg-fuchsia-500"
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}

function Select({
  value,
  onChange,
  children,
}: {
  value: string | number;
  onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg border bg-white px-3 py-2 text-sm shadow-sm outline-none focus:ring-2 focus:ring-primary/40"
    >
      {children}
    </select>
  );
}
