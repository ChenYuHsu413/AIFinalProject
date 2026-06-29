import { cn } from "@/lib/utils";

/** Pulsing placeholder block. Shared across the dashboard loading states. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <span
      className={cn("block animate-pulse rounded bg-muted/70", className)}
      aria-hidden
    />
  );
}

function CardSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "rounded-xl border border-border/70 bg-card/60 p-4 shadow-sm",
        className,
      )}
    >
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-12 rounded-full" />
      </div>
      <Skeleton className="mt-3 h-8 w-16" />
      <Skeleton className="mt-3 h-1.5 w-full rounded-full" />
      <Skeleton className="mt-4 h-3 w-full" />
      <Skeleton className="mt-1.5 h-3 w-4/5" />
      <Skeleton className="mt-4 h-12 w-full rounded-lg" />
    </div>
  );
}

/**
 * Loading placeholder for the home dashboard's model-driven sections (Action
 * Required, line map + brief, motor grid, alert/work-order queue). Shown on the
 * first visit when there is no cached fleet yet, so the page never paints the
 * mock placeholder data.
 */
export function DashboardLoadingSkeleton() {
  return (
    <>
      {/* Action Required */}
      <section>
        <Skeleton className="mb-3 h-4 w-48" />
        <div className="rounded-2xl border border-border/70 bg-card/60 p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:justify-between">
            <div className="flex-1">
              <Skeleton className="h-6 w-40" />
              <Skeleton className="mt-2 h-3 w-32" />
              <Skeleton className="mt-4 h-3 w-3/4" />
              <Skeleton className="mt-3 h-14 w-full rounded-lg" />
            </div>
            <div className="grid shrink-0 grid-cols-2 gap-3 lg:w-64">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-14 w-full rounded-lg" />
              ))}
            </div>
          </div>
          <div className="mt-4 flex gap-2">
            <Skeleton className="h-8 w-28 rounded-lg" />
            <Skeleton className="h-8 w-24 rounded-lg" />
            <Skeleton className="h-8 w-28 rounded-lg" />
          </div>
        </div>
      </section>

      {/* Line map + maintenance brief */}
      <section className="grid gap-4 lg:grid-cols-3">
        <Skeleton className="h-64 rounded-xl lg:col-span-2" />
        <Skeleton className="h-64 rounded-xl" />
      </section>

      {/* Motor health grid */}
      <div>
        <Skeleton className="mb-3 h-4 w-40" />
        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </section>
      </div>

      {/* Alert / work-order queue */}
      <div>
        <Skeleton className="mb-3 h-4 w-36" />
        <Skeleton className="h-44 w-full rounded-xl" />
      </div>
    </>
  );
}
