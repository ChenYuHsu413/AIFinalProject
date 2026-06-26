"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { NAV_GROUPS } from "@/lib/nav";
import { cn } from "@/lib/utils";

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r bg-muted/30 md:flex">
      <div className="border-b px-5 py-4">
        <p className="text-sm font-semibold leading-tight">伺服馬達健康監測</p>
        <p className="text-xs text-muted-foreground">智慧維護助理系統</p>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {NAV_GROUPS.map((group, gi) => (
          <div key={gi} className="mb-4">
            {group.title && (
              <p className="px-2 pb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
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
                        "flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors",
                        active
                          ? "bg-primary text-primary-foreground"
                          : "text-foreground/80 hover:bg-muted hover:text-foreground",
                      )}
                    >
                      <Icon className="h-4 w-4 shrink-0" />
                      <span className="truncate">{item.label}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      <div className="border-t px-4 py-3 text-xs text-muted-foreground">
        <div className="flex flex-wrap gap-1">
          <span className="rounded bg-muted px-1.5 py-0.5 font-medium">決策輔助</span>
          <span className="rounded bg-muted px-1.5 py-0.5 font-medium">不控制馬達</span>
        </div>
        <p className="mt-2">本系統提供維護建議，不直接控制馬達。</p>
        <a
          href="https://github.com/ChenYuHsu413/AIFinalProject"
          target="_blank"
          rel="noreferrer"
          className="mt-1 inline-block underline hover:text-foreground"
        >
          GitHub Repo
        </a>
      </div>
    </aside>
  );
}
