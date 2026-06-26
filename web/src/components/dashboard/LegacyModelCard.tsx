import Link from "next/link";
import { ArrowRight, type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export interface LegacyModel {
  code: string;
  name: string;
  dataset: string;
  task: string;
  href: string;
  icon: LucideIcon;
  accent: "blue" | "emerald" | "amber" | "rose";
}

const ACCENT: Record<LegacyModel["accent"], { icon: string; ring: string }> = {
  blue: { icon: "text-sky-300 bg-sky-500/15", ring: "ring-sky-500/25" },
  emerald: { icon: "text-emerald-300 bg-emerald-500/15", ring: "ring-emerald-500/25" },
  amber: { icon: "text-amber-300 bg-amber-500/15", ring: "ring-amber-500/25" },
  rose: { icon: "text-rose-300 bg-rose-500/15", ring: "ring-rose-500/25" },
};

/** Small entry card for legacy/comparison modules A / B / B+ / C. */
export function LegacyModelCard({ model }: { model: LegacyModel }) {
  const a = ACCENT[model.accent];
  const Icon = model.icon;
  return (
    <Link
      href={model.href}
      className="group flex items-center gap-3 rounded-lg border border-border/60 bg-card/50 p-3 text-sm shadow-sm backdrop-blur-sm transition-colors hover:border-primary/40 hover:bg-card/80"
    >
      <span
        className={cn(
          "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ring-1 ring-inset",
          a.icon,
          a.ring,
        )}
      >
        <Icon className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="font-medium leading-tight">
          <span className="text-muted-foreground">{model.code} · </span>
          {model.name}
        </p>
        <p className="truncate text-xs text-muted-foreground">
          {model.dataset} · {model.task}
        </p>
      </div>
      <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
    </Link>
  );
}
