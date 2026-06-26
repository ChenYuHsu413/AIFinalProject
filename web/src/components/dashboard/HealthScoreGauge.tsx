import { HEALTH_COLOR, scoreToState } from "@/lib/servo";
import { cn } from "@/lib/utils";

/**
 * Radial health-score gauge (pure SVG, no chart lib).
 * 270° sweep; arc colour follows the derived health state.
 */
export function HealthScoreGauge({
  score,
  size = 168,
  className,
}: {
  score: number;
  size?: number;
  className?: string;
}) {
  const clamped = Math.max(0, Math.min(100, score));
  const { state } = scoreToState(clamped);
  const color = HEALTH_COLOR[state].hex;

  const stroke = 12;
  const r = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const sweep = 270; // degrees
  const start = 135; // start angle (bottom-left)
  const circ = 2 * Math.PI * r;
  const arcLen = (sweep / 360) * circ;
  const filled = (clamped / 100) * arcLen;

  return (
    <div className={cn("relative inline-flex", className)} style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-[0deg]">
        {/* track */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="currentColor"
          className="text-muted/40"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${arcLen} ${circ}`}
          transform={`rotate(${start} ${cx} ${cy})`}
        />
        {/* value */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circ}`}
          transform={`rotate(${start} ${cx} ${cy})`}
          style={{ filter: `drop-shadow(0 0 6px ${color}66)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold tabular-nums" style={{ color }}>
          {Math.round(clamped)}
        </span>
        <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Health Score
        </span>
      </div>
    </div>
  );
}
