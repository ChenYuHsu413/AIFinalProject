import Link from "next/link";

import { NAV_GROUPS } from "@/lib/nav";

export default function Home() {
  // All groups that have a heading (skip the ungrouped home/about items).
  const sections = NAV_GROUPS.filter((g) => g.title);

  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <section className="mb-10">
        <p className="text-sm font-medium text-muted-foreground">
          Servo Health · Smart Maintenance
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight">
          AI 伺服馬達健康狀態估測與智慧維護助理
        </h1>
        <p className="mt-3 max-w-2xl text-muted-foreground">
          以伺服馬達退化資料為主線，結合 ML、訓練模擬器、LLM 維護助理與知識庫；
          模組 A / B / B+ / C 作為對照與歷史補充。前端建構於 FastAPI 契約之上。
        </p>
      </section>

      <div className="space-y-8">
        {sections.map((group) => (
          <section key={group.title}>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              {group.title}
            </h2>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {group.items.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="group flex items-center gap-3 rounded-lg border p-4 transition-colors hover:border-foreground/30 hover:bg-muted/50"
                  >
                    <Icon className="h-5 w-5 shrink-0 text-muted-foreground group-hover:text-foreground" />
                    <span className="text-sm font-medium">{item.label}</span>
                  </Link>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
