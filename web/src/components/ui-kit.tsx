import { cn } from "@/lib/utils";

/** Page title + description block. */
export function PageTitle({ title, desc }: { title: string; desc?: string }) {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
      {desc && <p className="mt-1 text-sm text-muted-foreground">{desc}</p>}
    </div>
  );
}

/** White rounded card with optional title. */
export function Card({
  title,
  children,
  className,
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("rounded-xl border bg-white p-5 shadow-sm", className)}>
      {title && <h2 className="mb-3 text-sm font-semibold">{title}</h2>}
      {children}
    </div>
  );
}

/** Compact KPI stat tile. */
export function Stat({
  label,
  value,
  sub,
  valueClass,
}: {
  label: string;
  value: string;
  sub?: string;
  valueClass?: string;
}) {
  return (
    <div className="rounded-xl border bg-white p-4 shadow-sm">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={cn("mt-1 text-2xl font-bold", valueClass)}>{value}</p>
      {sub && <p className="mt-1 text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}

/** Inline callout. */
export function Note({
  tone = "info",
  children,
  className,
}: {
  tone?: "info" | "warn" | "danger";
  children: React.ReactNode;
  className?: string;
}) {
  const cls = {
    info: "border-slate-200 bg-slate-50 text-slate-700",
    warn: "border-amber-200 bg-amber-50 text-amber-800",
    danger: "border-red-200 bg-red-50 text-red-800",
  }[tone];
  return (
    <div className={cn("rounded-lg border px-4 py-2.5 text-sm", cls, className)}>
      {children}
    </div>
  );
}

/** Labeled horizontal bar (value 0..1). */
export function Bar({
  label,
  right,
  value,
  colorClass = "bg-primary",
  sub,
}: {
  label: React.ReactNode;
  right?: React.ReactNode;
  value: number;
  colorClass?: string;
  sub?: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-0.5 flex justify-between text-xs">
        <span className="font-medium">{label}</span>
        {right != null && <span className="text-muted-foreground">{right}</span>}
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full", colorClass)}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
      </div>
      {sub && <p className="mt-1 text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}
