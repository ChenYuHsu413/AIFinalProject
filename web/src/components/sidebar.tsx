"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  ChevronRight,
  Package,
  PanelLeftClose,
  PanelLeftOpen,
  X,
} from "lucide-react";

import { ACCENTS, NAV_GROUPS, type NavGroup, type NavItem } from "@/lib/nav";
import { cn } from "@/lib/utils";
import { useSidebar } from "@/components/sidebar-context";

export function Sidebar() {
  const { collapsed, toggleCollapsed, mobileOpen, setMobileOpen } =
    useSidebar();

  return (
    <>
      {/* Desktop rail */}
      <aside
        className={cn(
          "hidden shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground transition-[width] duration-200 md:flex",
          collapsed ? "w-16" : "w-64",
        )}
      >
        <SidebarInner
          collapsed={collapsed}
          onToggleCollapse={toggleCollapsed}
        />
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <button
            type="button"
            aria-label="關閉選單"
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="absolute inset-y-0 left-0 flex w-64 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground shadow-2xl">
            <SidebarInner
              collapsed={false}
              onNavigate={() => setMobileOpen(false)}
              onClose={() => setMobileOpen(false)}
            />
          </aside>
        </div>
      )}
    </>
  );
}

function SidebarInner({
  collapsed,
  onNavigate,
  onToggleCollapse,
  onClose,
}: {
  collapsed: boolean;
  onNavigate?: () => void;
  onToggleCollapse?: () => void;
  onClose?: () => void;
}) {
  // home + Servo (main axis) → one collapsible "補充模組" expander (A/B/B+/C) → about.
  const blocks: React.ReactNode[] = [];
  let supp: NavGroup[] = [];
  NAV_GROUPS.forEach((g, i) => {
    if (g.supplementary) {
      supp.push(g);
      return;
    }
    if (supp.length) {
      blocks.push(
        <SupplementaryExpander
          key={`supp-${i}`}
          groups={supp}
          collapsed={collapsed}
          onNavigate={onNavigate}
        />,
      );
      supp = [];
    }
    blocks.push(
      <PlainGroup
        key={i}
        group={g}
        collapsed={collapsed}
        onNavigate={onNavigate}
      />,
    );
  });
  if (supp.length)
    blocks.push(
      <SupplementaryExpander
        key="supp"
        groups={supp}
        collapsed={collapsed}
        onNavigate={onNavigate}
      />,
    );

  return (
    <>
      {/* brand */}
      <div
        className={cn(
          "flex items-center gap-2.5 border-b border-sidebar-border px-5 py-4",
          collapsed && "justify-center px-0",
        )}
      >
        <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-sky-600 text-sm font-bold text-slate-950 shadow-[0_0_18px_-2px] shadow-cyan-500/50">
          <Activity className="h-5 w-5" />
        </div>
        {!collapsed && (
          <div className="min-w-0 leading-tight">
            <p className="truncate text-sm font-semibold tracking-tight">
              Command Center
            </p>
            <p className="truncate text-[11px] uppercase tracking-wide text-muted-foreground">
              Servo Motor Health
            </p>
          </div>
        )}
        {onClose && (
          <button
            type="button"
            aria-label="關閉選單"
            onClick={onClose}
            className="ml-auto flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">{blocks}</nav>

      {/* footer */}
      <div className="border-t border-sidebar-border px-3 py-3">
        {!collapsed && (
          <div className="px-1 text-xs text-muted-foreground">
            <div className="flex flex-wrap gap-1.5">
              <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 font-medium text-emerald-700 dark:text-emerald-300 ring-1 ring-inset ring-emerald-500/30">
                決策輔助
              </span>
              <span className="rounded-full bg-amber-500/15 px-2 py-0.5 font-medium text-amber-700 dark:text-amber-300 ring-1 ring-inset ring-amber-500/30">
                不控制馬達
              </span>
            </div>
            <p className="mt-2">本系統提供維護建議，不直接控制馬達。</p>
          </div>
        )}
        {onToggleCollapse && (
          <button
            type="button"
            onClick={onToggleCollapse}
            title={collapsed ? "展開側欄" : "收合側欄"}
            className={cn(
              "mt-2 flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-foreground",
              collapsed && "justify-center",
            )}
          >
            {collapsed ? (
              <PanelLeftOpen className="h-4 w-4" />
            ) : (
              <>
                <PanelLeftClose className="h-4 w-4" />
                收合側欄
              </>
            )}
          </button>
        )}
      </div>
    </>
  );
}

function NavRow({
  item,
  accentKey,
  collapsed,
  onNavigate,
}: {
  item: NavItem;
  accentKey: NavGroup["accent"];
  collapsed: boolean;
  onNavigate?: () => void;
}) {
  const pathname = usePathname();
  const accent = ACCENTS[accentKey];
  const active = pathname === item.href;
  const Icon = item.icon;
  return (
    <Link
      href={item.href}
      onClick={onNavigate}
      title={collapsed ? item.label : undefined}
      className={cn(
        "flex h-8 items-center gap-2.5 rounded-md text-sm transition-colors",
        collapsed ? "justify-center px-0" : "px-2",
        active
          ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
          : "text-sidebar-foreground/75 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
      )}
    >
      <Icon
        className={cn("h-4 w-4 shrink-0", active ? "text-primary" : accent.text)}
      />
      {!collapsed && <span className="truncate">{item.label}</span>}
    </Link>
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

/** Home / Servo / 運維 / About — always visible. */
function PlainGroup({
  group,
  collapsed,
  onNavigate,
}: {
  group: NavGroup;
  collapsed: boolean;
  onNavigate?: () => void;
}) {
  const items = (
    <ul className="space-y-0.5 pt-0.5">
      {group.items.map((item) => (
        <li key={item.href}>
          <NavRow
            item={item}
            accentKey={group.accent}
            collapsed={collapsed}
            onNavigate={onNavigate}
          />
        </li>
      ))}
    </ul>
  );

  if (!group.title) return <div className="mb-2">{items}</div>;
  return (
    <div className="mb-4">
      {!collapsed && <GroupHeading group={group} />}
      {items}
    </div>
  );
}

/** Supplementary modules A/B/B+/C. Expander when expanded, flat icons when collapsed. */
function SupplementaryExpander({
  groups,
  collapsed,
  onNavigate,
}: {
  groups: NavGroup[];
  collapsed: boolean;
  onNavigate?: () => void;
}) {
  const pathname = usePathname();
  const containsActive = groups.some((g) =>
    g.items.some((i) => i.href === pathname),
  );
  const [open, setOpen] = useState(containsActive);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (containsActive) setOpen(true);
  }, [containsActive]);

  // Collapsed rail: just show the icons flat, no expander chrome.
  if (collapsed) {
    return (
      <div className="mb-2 mt-1 space-y-0.5 border-t border-sidebar-border pt-2">
        {groups.flatMap((g) =>
          g.items.map((item) => (
            <NavRow
              key={item.href}
              item={item}
              accentKey={g.accent}
              collapsed
              onNavigate={onNavigate}
            />
          )),
        )}
      </div>
    );
  }

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
                <ul className="space-y-0.5 pt-0.5">
                  {g.items.map((item) => (
                    <li key={item.href}>
                      <NavRow
                        item={item}
                        accentKey={g.accent}
                        collapsed={false}
                        onNavigate={onNavigate}
                      />
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
