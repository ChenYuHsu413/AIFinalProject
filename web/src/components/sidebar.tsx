"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, ChevronRight, Package } from "lucide-react";

import { ACCENTS, NAV_GROUPS, type NavGroup } from "@/lib/nav";
import { cn } from "@/lib/utils";

export function Sidebar() {
  // Lay out: home + Servo (main axis) → one "補充模組" expander (A/B/B+/C) → about.
  const blocks: React.ReactNode[] = [];
  let supp: NavGroup[] = [];
  NAV_GROUPS.forEach((g, i) => {
    if (g.supplementary) {
      supp.push(g);
      return;
    }
    if (supp.length) {
      blocks.push(<SupplementaryExpander key={`supp-${i}`} groups={supp} />);
      supp = [];
    }
    blocks.push(<PlainGroup key={i} group={g} />);
  });
  if (supp.length) blocks.push(<SupplementaryExpander key="supp" groups={supp} />);

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground md:flex">
      <div className="flex items-center gap-2.5 border-b border-sidebar-border px-5 py-4">
        <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-sky-600 text-sm font-bold text-slate-950 shadow-[0_0_18px_-2px] shadow-cyan-500/50">
          <Activity className="h-5 w-5" />
        </div>
        <div className="leading-tight">
          <p className="text-sm font-semibold tracking-tight">Command Center</p>
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
            Servo Motor Health
          </p>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">{blocks}</nav>

      <div className="border-t border-sidebar-border px-4 py-3 text-xs text-muted-foreground">
        <div className="flex flex-wrap gap-1.5">
          <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 font-medium text-emerald-300 ring-1 ring-inset ring-emerald-500/30">
            決策輔助
          </span>
          <span className="rounded-full bg-amber-500/15 px-2 py-0.5 font-medium text-amber-300 ring-1 ring-inset ring-amber-500/30">
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

function GroupItems({ group }: { group: NavGroup }) {
  const pathname = usePathname();
  const accent = ACCENTS[group.accent];
  return (
    <ul className="space-y-0.5 pt-0.5">
      {group.items.map((item) => {
        const active = pathname === item.href;
        const Icon = item.icon;
        return (
          <li key={item.href}>
            <Link
              href={item.href}
              className={cn(
                "flex h-8 items-center gap-2.5 rounded-md px-2 text-sm transition-colors",
                active
                  ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/75 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
              )}
            >
              <Icon
                className={cn(
                  "h-4 w-4 shrink-0",
                  active ? "text-primary" : accent.text,
                )}
              />
              <span className="truncate">{item.label}</span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

function GroupHeading({ group }: { group: NavGroup }) {
  const accent = ACCENTS[group.accent];
  return (
    <p className="flex items-center gap-1.5 px-2 pb-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
      <span className={cn("h-1.5 w-1.5 rounded-full", accent.dot)} />
      {group.title}
    </p>
  );
}

/** Home / Servo / About — always visible. */
function PlainGroup({ group }: { group: NavGroup }) {
  if (!group.title) {
    return (
      <div className="mb-2">
        <GroupItems group={group} />
      </div>
    );
  }
  return (
    <div className="mb-4">
      <GroupHeading group={group} />
      <GroupItems group={group} />
    </div>
  );
}

/** Supplementary modules A/B/B+/C, tucked into one collapsible expander. */
function SupplementaryExpander({ groups }: { groups: NavGroup[] }) {
  const pathname = usePathname();
  const containsActive = groups.some((g) =>
    g.items.some((i) => i.href === pathname),
  );
  const [open, setOpen] = useState(containsActive);

  useEffect(() => {
    if (containsActive) setOpen(true);
  }, [containsActive]);

  return (
    <div className="mb-2 mt-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground transition-colors hover:bg-sidebar-accent/60"
      >
        <Package className="h-3.5 w-3.5" />
        <span className="flex-1 text-left">補充模組（對照 / 歷史）</span>
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
        <div className="overflow-hidden">
          <div className="space-y-3 pt-1.5 pl-1">
            {groups.map((g, i) => (
              <div key={i}>
                <GroupHeading group={g} />
                <GroupItems group={g} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
