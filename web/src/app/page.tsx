import Link from "next/link";

import { ACCENTS, NAV_GROUPS } from "@/lib/nav";

export default function Home() {
  const sections = NAV_GROUPS.filter((g) => g.title);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <section className="mb-10 overflow-hidden rounded-2xl bg-gradient-to-br from-indigo-500 via-violet-500 to-fuchsia-500 p-8 text-white shadow-lg">
        <p className="text-sm font-medium text-white/80">
          Servo Health · Smart Maintenance
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight">
          AI 伺服馬達健康狀態估測與智慧維護助理
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-white/90">
          以伺服馬達退化資料為主線，結合 ML、訓練模擬器、LLM 維護助理與知識庫；
          模組 A / B / B+ / C 作為對照與歷史補充。前端建構於 FastAPI 契約之上。
        </p>
      </section>

      <div className="space-y-8">
        {sections.map((group) => {
          const accent = ACCENTS[group.accent];
          return (
            <section key={group.title}>
              <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                <span className={`h-2 w-2 rounded-full ${accent.dot}`} />
                {group.title}
              </h2>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`group flex items-center gap-3 rounded-xl border bg-white p-4 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md ${accent.hover}`}
                    >
                      <span
                        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${accent.chip}`}
                      >
                        <Icon className="h-5 w-5" />
                      </span>
                      <span className="text-sm font-medium">{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
