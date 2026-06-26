"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ACCENTS, NAV_GROUPS } from "@/lib/nav";
import { cn } from "@/lib/utils";

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r bg-white md:flex">
      <div className="flex items-center gap-2.5 border-b px-5 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-violet-500 to-fuchsia-500 text-sm font-bold text-white shadow-sm">
          S
        </div>
        <div className="leading-tight">
          <p className="text-sm font-semibold">伺服馬達健康監測</p>
          <p className="text-xs text-muted-foreground">智慧維護助理系統</p>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {NAV_GROUPS.map((group, gi) => {
          const accent = ACCENTS[group.accent];
          return (
            <div key={gi} className="mb-4">
              {group.title && (
                <p className="flex items-center gap-1.5 px-2 pb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  <span className={cn("h-1.5 w-1.5 rounded-full", accent.dot)} />
                  {group.title}
                </p>
              )}
              <ul className="space-y-0.5">
                {group.items.map((item) => {
                  const active = pathname === item.href;
                  const Icon = item.icon;
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        className={cn(
                          "flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-sm transition-colors",
                          active
                            ? "bg-primary font-medium text-primary-foreground shadow-sm"
                            : "text-foreground/75 hover:bg-muted hover:text-foreground",
                        )}
                      >
                        <Icon
                          className={cn(
                            "h-4 w-4 shrink-0",
                            !active && accent.text,
                          )}
                        />
                        <span className="truncate">{item.label}</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </nav>

      <div className="border-t px-4 py-3 text-xs text-muted-foreground">
        <div className="flex flex-wrap gap-1.5">
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 font-medium text-emerald-700">
            決策輔助
          </span>
          <span className="rounded-full bg-amber-100 px-2 py-0.5 font-medium text-amber-700">
            不控制馬達
          </span>
        </div>
        <p className="mt-2">本系統提供維護建議，不直接控制馬達。</p>
        <a
          href="https://github.com/ChenYuHsu413/AIFinalProject"
          target="_blank"
          rel="noreferrer"
          className="mt-1 inline-block font-medium text-primary hover:underline"
        >
          GitHub Repo →
        </a>
      </div>
    </aside>
  );
}
