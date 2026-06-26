"use client";

import { usePathname } from "next/navigation";

import { NAV_GROUPS } from "@/lib/nav";

/** Temporary placeholder for routes not yet migrated (Phase 2 · T19). */
export default function StubPage() {
  const pathname = usePathname();
  const item = NAV_GROUPS.flatMap((g) => g.items).find(
    (i) => i.href === pathname,
  );
  const Icon = item?.icon;

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <div className="flex items-center gap-3">
        {Icon && <Icon className="h-7 w-7 text-muted-foreground" />}
        <h1 className="text-2xl font-bold tracking-tight">
          {item?.label ?? "頁面"}
        </h1>
      </div>
      <div className="mt-6 rounded-lg border border-dashed p-8 text-center text-muted-foreground">
        <p className="text-sm">
          此頁規劃於 Phase 2（T19）由 Streamlit 逐步搬移；後端 API 已就緒。
        </p>
      </div>
    </div>
  );
}
