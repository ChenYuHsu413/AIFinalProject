"use client";

import { usePathname } from "next/navigation";

import { ACCENTS, NAV_GROUPS } from "@/lib/nav";

/** Temporary placeholder for routes not yet migrated (Phase 2 · T19). */
export default function StubPage() {
  const pathname = usePathname();
  const group = NAV_GROUPS.find((g) =>
    g.items.some((i) => i.href === pathname),
  );
  const item = group?.items.find((i) => i.href === pathname);
  const accent = ACCENTS[group?.accent ?? "slate"];
  const Icon = item?.icon;

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <div className="flex items-center gap-3">
        {Icon && (
          <span
            className={`flex h-11 w-11 items-center justify-center rounded-xl ${accent.chip}`}
          >
            <Icon className="h-6 w-6" />
          </span>
        )}
        <h1 className="text-2xl font-bold tracking-tight">
          {item?.label ?? "頁面"}
        </h1>
      </div>
      <div className="mt-6 rounded-xl border border-dashed border-border/70 bg-card/50 p-10 text-center text-muted-foreground shadow-sm backdrop-blur-sm">
        <p className="text-sm">
          此頁規劃於 Phase 2（T19）由 Streamlit 逐步搬移；後端 API 已就緒。
        </p>
      </div>
    </div>
  );
}
