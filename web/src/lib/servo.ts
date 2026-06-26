/** Shared label / colour maps for the Servo health states and risk levels. */

export const HEALTH_ORDER = ["LN", "LO", "MED", "HI"] as const;

export const HEALTH_ZH: Record<string, string> = {
  LN: "健康",
  LO: "輕度退化",
  MED: "中度退化",
  HI: "高度退化",
};

/** Literal Tailwind classes per health state (JIT-safe). */
export const HEALTH_COLOR: Record<
  string,
  { bar: string; text: string; chip: string }
> = {
  LN: { bar: "bg-emerald-500", text: "text-emerald-600", chip: "bg-emerald-100 text-emerald-700" },
  LO: { bar: "bg-lime-500", text: "text-lime-600", chip: "bg-lime-100 text-lime-700" },
  MED: { bar: "bg-amber-500", text: "text-amber-600", chip: "bg-amber-100 text-amber-700" },
  HI: { bar: "bg-red-500", text: "text-red-600", chip: "bg-red-100 text-red-700" },
};

export const RISK_ZH: Record<string, string> = {
  Low: "低",
  Medium: "中",
  High: "高",
};

export const RISK_COLOR: Record<string, string> = {
  Low: "bg-emerald-100 text-emerald-700",
  Medium: "bg-amber-100 text-amber-700",
  High: "bg-red-100 text-red-700",
};
