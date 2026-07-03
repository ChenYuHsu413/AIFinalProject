// Shared constants + small components for the Live Monitor v3 pages
// (replay + live). Keeps the subsystem/stage vocabulary in one place.

export const SUB_ZH: Record<string, string> = {
  temperature: "溫度",
  current: "電流",
  vibration: "振動",
  encoder: "編碼器",
  motion: "運動追隨",
  communication: "通訊",
};

export const STAGE = [
  { key: "normal", zh: "正常", color: "#10b981", tone: "text-emerald-500" },
  { key: "early_degradation", zh: "早期退化", color: "#38bdf8", tone: "text-sky-500" },
  { key: "warning", zh: "警告", color: "#f59e0b", tone: "text-amber-500" },
  { key: "alarm", zh: "告警", color: "#f97316", tone: "text-orange-500" },
  { key: "trip", zh: "跳機停機", color: "#ef4444", tone: "text-red-500" },
];

export const SPEEDS = [0.5, 1, 2, 4, 8];

export function healthColor(h: number): string {
  if (h >= 70) return "#10b981";
  if (h >= 45) return "#f59e0b";
  return "#ef4444";
}

export function severityColorClass(v: number): string {
  return v >= 0.66 ? "bg-red-500" : v >= 0.33 ? "bg-amber-500" : "bg-emerald-500";
}

/** Circular health gauge (inline SVG). */
export function HealthRing({ value, size = 140 }: { value: number; size?: number }) {
  const r = size * 0.37;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value)) / 100;
  const color = healthColor(value);
  return (
    <div className="relative" style={{ height: size, width: size }}>
      <svg viewBox="0 0 140 140" className="h-full w-full -rotate-90">
        <circle cx="70" cy="70" r={140 * 0.37} fill="none" strokeWidth="12" className="stroke-muted" />
        <circle
          cx="70"
          cy="70"
          r={140 * 0.37}
          fill="none"
          strokeWidth="12"
          stroke={color}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct)}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-black tabular-nums" style={{ color }}>
          {value.toFixed(0)}
        </span>
        <span className="text-xs text-muted-foreground">健康分數</span>
      </div>
    </div>
  );
}
