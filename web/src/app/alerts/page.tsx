import { MetricCard } from "@/components/dashboard/MetricCard";
import { AlertTable } from "@/components/dashboard/AlertTable";
import { WorkOrderCard } from "@/components/dashboard/WorkOrderCard";
import { PageTitle, Note } from "@/components/ui-kit";
import { ALERTS, WORK_ORDERS } from "@/lib/mock";

export default function AlertsPage() {
  const open = ALERTS.filter((a) => a.status === "open").length;
  const critical = ALERTS.filter((a) => a.severity === "critical").length;
  const wipOrders = WORK_ORDERS.filter((w) => w.status === "in_progress").length;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <PageTitle
        title="告警 / 工單中心"
        desc="設備告警事件與維修工單追蹤（目前為 mock 示意，待 Servo Dataset 模組產生真實告警）"
      />

      <Note tone="warn" className="mb-6">
        此頁資料為示意用 mock；後端尚無告警 / 工單 endpoint。介面已預先對齊未來 API
        形狀，屆時抽換 <code className="font-mono">lib/mock.ts</code> 即可接真資料。
      </Note>

      <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-3">
        <MetricCard
          label="未處理告警"
          value={open}
          unit="筆"
          valueClassName={open > 0 ? "text-red-400" : "text-emerald-400"}
          footerMuted="需立即指派工單"
        />
        <MetricCard
          label="嚴重告警"
          value={critical}
          unit="筆"
          valueClassName={critical > 0 ? "text-amber-400" : "text-emerald-400"}
          footerMuted="severity = critical"
        />
        <MetricCard
          label="處理中工單"
          value={wipOrders}
          unit="張"
          footerMuted="維修班執行中"
        />
      </section>

      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide">告警事件</h2>
      <section className="mb-8">
        <AlertTable alerts={ALERTS} />
      </section>

      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide">維修工單</h2>
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {WORK_ORDERS.map((w) => (
          <WorkOrderCard key={w.id} order={w} />
        ))}
      </section>
    </div>
  );
}
