/** Shared label / colour maps for the Servo health states and risk levels. */

export const HEALTH_ORDER = ["LN", "LO", "MED", "HI"] as const;

export const HEALTH_ZH: Record<string, string> = {
  LN: "健康",
  LO: "輕度退化",
  MED: "中度退化",
  HI: "高度退化",
};

export const HEALTH_EN: Record<string, string> = {
  LN: "Nominal",
  LO: "Low",
  MED: "Medium",
  HI: "High",
};

/** Literal Tailwind classes per health state, tuned for the dark theme. */
export const HEALTH_COLOR: Record<
  string,
  { bar: string; text: string; chip: string; hex: string }
> = {
  LN: {
    bar: "bg-emerald-500",
    text: "text-emerald-600 dark:text-emerald-400",
    chip: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 ring-1 ring-inset ring-emerald-500/30",
    hex: "#34d399",
  },
  LO: {
    bar: "bg-lime-500",
    text: "text-lime-600 dark:text-lime-400",
    chip: "bg-lime-500/15 text-lime-700 dark:text-lime-300 ring-1 ring-inset ring-lime-500/30",
    hex: "#a3e635",
  },
  MED: {
    bar: "bg-amber-500",
    text: "text-amber-600 dark:text-amber-400",
    chip: "bg-amber-500/15 text-amber-700 dark:text-amber-300 ring-1 ring-inset ring-amber-500/30",
    hex: "#fbbf24",
  },
  HI: {
    bar: "bg-red-500",
    text: "text-red-600 dark:text-red-400",
    chip: "bg-red-500/15 text-red-700 dark:text-red-300 ring-1 ring-inset ring-red-500/30",
    hex: "#f87171",
  },
};

export const RISK_ZH: Record<string, string> = {
  Low: "低",
  Medium: "中",
  High: "高",
};

/** Dark-theme risk chip classes. */
export const RISK_COLOR: Record<string, string> = {
  Low: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 ring-1 ring-inset ring-emerald-500/30",
  Medium: "bg-amber-500/15 text-amber-700 dark:text-amber-300 ring-1 ring-inset ring-amber-500/30",
  High: "bg-red-500/15 text-red-700 dark:text-red-300 ring-1 ring-inset ring-red-500/30",
};

export type RiskLevel = "Low" | "Medium" | "High";

/** Map a 0..100 health score to a discrete state + risk (used by mock fleet). */
export function scoreToState(score: number): {
  state: (typeof HEALTH_ORDER)[number];
  risk: RiskLevel;
} {
  if (score >= 80) return { state: "LN", risk: "Low" };
  if (score >= 60) return { state: "LO", risk: "Low" };
  if (score >= 40) return { state: "MED", risk: "Medium" };
  return { state: "HI", risk: "High" };
}
