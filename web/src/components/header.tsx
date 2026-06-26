"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight, ExternalLink } from "lucide-react";

import { apiGet, type Health, type ServoModelInfo } from "@/lib/api";
import { NAV_GROUPS } from "@/lib/nav";

/** Derive a [section, page] breadcrumb trail from the current path. */
function useCrumbs(): { label: string; href?: string }[] {
  const pathname = usePathname();
  if (pathname === "/") return [{ label: "總覽 Overview" }];
  for (const g of NAV_GROUPS) {
    const item = g.items.find((i) => i.href === pathname);
    if (item) {
      return [
        { label: g.title ?? "總覽", href: "/" },
        { label: item.label },
      ];
    }
  }
  return [{ label: "總覽 Overview", href: "/" }];
}

export function Header() {
  const crumbs = useCrumbs();
  const [health, setHealth] = useState<Health | null>(null);
  const [servo, setServo] = useState<ServoModelInfo | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [h, s] = await Promise.all([
          apiGet<Health>("/health"),
          apiGet<ServoModelInfo>("/servo/model_info"),
        ]);
        if (!alive) return;
        setHealth(h);
        setServo(s);
      } catch {
        if (alive) setErr(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const connected = !err && health !== null;

  return (
    <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center justify-between gap-2 border-b border-border/70 bg-background/60 backdrop-blur-md">
      {/* breadcrumbs */}
      <nav className="flex items-center gap-1.5 px-4 text-sm">
        {crumbs.map((c, i) => {
          const last = i === crumbs.length - 1;
          return (
            <span key={i} className="flex items-center gap-1.5">
              {i > 0 && (
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/50" />
              )}
              {c.href && !last ? (
                <Link
                  href={c.href}
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  {c.label}
                </Link>
              ) : (
                <span className={last ? "font-medium text-foreground" : "text-muted-foreground"}>
                  {c.label}
                </span>
              )}
            </span>
          );
        })}
      </nav>

      {/* actions / status */}
      <div className="flex items-center gap-3 px-4 text-xs">
        {servo?.placeholder && (
          <span
            className="hidden rounded-md bg-amber-500/15 px-2 py-0.5 font-medium text-amber-300 ring-1 ring-inset ring-amber-500/30 sm:inline"
            title="目前以 placeholder 合成資料訓練，非真實 PHM 伺服馬達資料"
          >
            placeholder 合成資料
          </span>
        )}
        {servo && (
          <span className="hidden text-muted-foreground md:inline">
            主線模型{" "}
            <span className="font-medium text-foreground">
              {servo.clf_model ?? "—"}
            </span>
            {servo.clf_macro_f1 != null && (
              <span className="font-mono"> · F1 {servo.clf_macro_f1.toFixed(3)}</span>
            )}
          </span>
        )}

        <span className="flex items-center gap-1.5 rounded-md border border-border/70 bg-card/50 px-2 py-1">
          <span className="relative flex h-2 w-2">
            {connected && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
            )}
            <span
              className={`relative inline-flex h-2 w-2 rounded-full ${
                err ? "bg-red-500" : connected ? "bg-emerald-500" : "bg-amber-400"
              }`}
            />
          </span>
          <span className="text-muted-foreground">
            {err ? "離線" : connected ? "已連線" : "連線中"}
          </span>
        </span>

        <a
          href="https://github.com/ChenYuHsu413/AIFinalProject"
          target="_blank"
          rel="noreferrer"
          className="flex h-7 w-7 items-center justify-center rounded-md border border-border/70 bg-card/50 text-muted-foreground transition-colors hover:text-foreground"
          title="GitHub Repo"
        >
          <ExternalLink className="h-4 w-4" />
        </a>
      </div>
    </header>
  );
}
