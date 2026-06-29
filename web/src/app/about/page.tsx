import { Card, Note, PageTitle } from "@/components/ui-kit";

const TRACKS = [
  {
    code: "Servo（主線）",
    name: "伺服馬達健康估測",
    data: "PHM FMCRD 伺服馬達退化（高擬真模擬）",
    task: "健康狀態分類 + 退化值 DV 回歸",
    tone: "text-violet-300",
  },
  {
    code: "模組 C（對照 · 最貼近馬達）",
    name: "馬達電流診斷",
    data: "Paderborn PMSM 試驗台",
    task: "MCSA 故障分類（人工→真實泛化）",
    tone: "text-rose-300",
  },
  {
    code: "模組 B（對照）",
    name: "動態健康度",
    data: "IMS 軸承（單軌跡 Set 2）",
    task: "趨勢外推 RUL / 健康度",
    tone: "text-emerald-300",
  },
  {
    code: "模組 B+（對照）",
    name: "多軌跡泛化",
    data: "XJTU-SY（15 軸承 / 3 工況）",
    task: "跨軸承 / 跨工況泛化",
    tone: "text-amber-300",
  },
  {
    code: "模組 A（對照 · 合成基礎）",
    name: "靜態風險",
    data: "AI4I 2020（合成）",
    task: "單點故障分類",
    tone: "text-slate-400",
  },
];

const DISCLAIMERS = [
  "Servo 主線已以完整真實 PHM FMCRD 資料集（106.66 GB）重訓；FMCRD 為高擬真模擬資料集，非真實工廠伺服馬達遙測——「真實」指完整大型公開 PHM 資料集本身（相對於先前 placeholder 合成）。機群健康為參考模型在代表性 demo 運轉段上的即時輸出，設備識別與遙測趨勢 / 告警排程為示意包裝。",
  "AI4I 2020 為合成資料，不得宣稱為真實伺服馬達資料。",
  "IMS Set 2 為單軌跡，其結果不可泛化到其他軸承 / 馬達；不在單軌跡上做深度 RUL 回歸。",
  "Paderborn 為真實 PMSM 試驗台訊號（MCSA 成立），但屬試驗台、非產線伺服馬達；含人工與真實兩種損傷，須如實呈現泛化落差；屬故障分類非 RUL；為子集 MVP。",
  "ESP32 定位為未來實場接入 / IoT demo，非現階段訓練資料來源。",
  "本系統提供維護「建議」，為決策輔助，不直接控制馬達。",
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
      <PageTitle
        title="關於本專案"
        desc="AI 伺服馬達健康監控與智慧維護指揮中心 —— 以 Servo 為主線、A / B / B+ / C 對照（依貼近馬達程度 C > B/B+ > A）與誠實性聲明。"
      />

      <Card title="三軌定位" className="mb-6">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="py-2 pr-4 font-medium">軌道</th>
                <th className="py-2 pr-4 font-medium">主題</th>
                <th className="py-2 pr-4 font-medium">資料</th>
                <th className="py-2 font-medium">任務</th>
              </tr>
            </thead>
            <tbody>
              {TRACKS.map((t) => (
                <tr key={t.code} className="border-b border-border/40 align-top last:border-0">
                  <td className={`py-2 pr-4 font-semibold ${t.tone}`}>{t.code}</td>
                  <td className="py-2 pr-4">{t.name}</td>
                  <td className="py-2 pr-4 text-muted-foreground">{t.data}</td>
                  <td className="py-2 text-muted-foreground">{t.task}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide">誠實性聲明（報告防禦）</h2>
      <div className="space-y-2">
        {DISCLAIMERS.map((d, i) => (
          <Note key={i} tone={i === DISCLAIMERS.length - 1 ? "info" : "warn"}>
            {d}
          </Note>
        ))}
      </div>

      <p className="mt-6 text-xs text-muted-foreground">
        前端建構於 FastAPI 契約之上（Next.js App Router + TypeScript + Tailwind v4 + shadcn）。
        機群健康與告警由參考模型在 demo 運轉段上即時計算；遙測趨勢為示意 mock。
      </p>
    </div>
  );
}
