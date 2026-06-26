import type { ServoTopFeature } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Ranked anomalous features with z-score magnitude bars. */
export function FeatureImportancePanel({
  features,
}: {
  features: ServoTopFeature[];
}) {
  return (
    <div className="space-y-3">
      {features.map((t) => {
        const mag = Math.min(1, Math.abs(t.z) / 6);
        const tone =
          Math.abs(t.z) > 3
            ? "bg-red-500"
            : Math.abs(t.z) > 1.5
              ? "bg-amber-500"
              : "bg-emerald-500";
        return (
          <div key={t.feature}>
            <div className="mb-0.5 flex items-center justify-between text-xs">
              <span className="font-mono font-medium">{t.feature}</span>
              <span
                className={cn(
                  "tabular-nums",
                  Math.abs(t.z) > 3
                    ? "text-red-300"
                    : Math.abs(t.z) > 1.5
                      ? "text-amber-300"
                      : "text-muted-foreground",
                )}
              >
                z = {t.z}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div
                className={cn("h-full rounded-full", tone)}
                style={{ width: `${mag * 100}%` }}
              />
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{t.hint}</p>
          </div>
        );
      })}
    </div>
  );
}
