"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight } from "lucide-react";

import { ACCENTS, NAV_GROUPS, type NavGroup } from "@/lib/nav";
import { cn } from "@/lib/utils";

export function Sidebar() {
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
        {NAV_GROUPS.map((group, gi) => (
          <SidebarGroup key={gi} group={group} />
        ))}
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

function SidebarGroup({ group }: { group: NavGroup }) {
  const pathname = usePathname();
  const accent = ACCENTS[group.accent];
  const containsActive = group.items.some((i) => i.href === pathname);

  const collapsible = group.collapsible ?? false;
  const [open, setOpen] = useState(
    !collapsible || (group.defaultOpen ?? false) || containsActive,
  );

  // Auto-open when navigating into a child route (e.g. from a home card).
  useEffect(() => {
    if (containsActive) setOpen(true);
  }, [containsActive]);

  const items = (
    <ul className="space-y-0.5 pt-0.5">
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
                  ? accent.active
                  : cn("text-foreground/75", accent.tint),
              )}
            >
              <Icon
                className={cn("h-4 w-4 shrink-0", !active && accent.text)}
              />
              <span className="truncate">{item.label}</span>
            </Link>
          </li>
        );
      })}
    </ul>
  );

  // Ungrouped single items (home / about): render directly, no header.
  if (!group.title) {
    return <div className="mb-2">{items}</div>;
  }

  if (!collapsible) {
    return (
      <div className="mb-4">
        <p className="flex items-center gap-1.5 px-2 pb-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          <span className={cn("h-1.5 w-1.5 rounded-full", accent.dot)} />
          {group.title}
        </p>
        {items}
      </div>
    );
  }

  return (
    <div className="mb-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "flex w-full items-center gap-1.5 rounded-lg px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground transition-colors",
          accent.tint,
        )}
      >
        <span className={cn("h-1.5 w-1.5 rounded-full", accent.dot)} />
        <span className="flex-1 text-left">{group.title}</span>
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 transition-transform duration-200",
            open && "rotate-90",
          )}
        />
      </button>
      <div
        className={cn(
          "grid transition-[grid-template-rows] duration-200 ease-out",
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        )}
      >
        <div className="overflow-hidden">{items}</div>
      </div>
    </div>
  );
}
