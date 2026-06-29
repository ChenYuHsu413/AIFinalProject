"use client";

import { useEffect, useState } from "react";

import { HEALTH_COLOR, scoreToState } from "@/lib/servo";
import { cn } from "@/lib/utils";

/** Health-score zones (matches scoreToState thresholds) → arc colours. */
const ZONES: [number, number, string][] = [
  [0, 40, HEALTH_COLOR.HI.hex],
  [40, 60, HEALTH_COLOR.MED.hex],
  [60, 80, HEALTH_COLOR.LO.hex],
  [80, 100, HEALTH_COLOR.LN.hex],
];

// Round trig-derived coordinates so the SSR (Node) and client (browser) emit
// byte-identical path strings — Math.cos/sin can differ in the last ULP across
// engines, which would otherwise trigger a hydration mismatch.
const rnd = (n: number) => Math.round(n * 100) / 100;

function polar(cx: number, cy: number, r: number, deg: number) {
  const a = (deg * Math.PI) / 180;
  return [rnd(cx + r * Math.cos(a)), rnd(cy + r * Math.sin(a))] as const;
}

function arcPath(cx: number, cy: number, r: number, a0: number, a1: number) {
  const [x0, y0] = polar(cx, cy, r, a0);
  const [x1, y1] = polar(cx, cy, r, a1);
  const large = a1 - a0 <= 180 ? 0 : 1;
  return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
}

/** Triangle-needle polygon points from the hub toward `deg`. */
function needlePoints(
  cx: number,
  cy: number,
  deg: number,
  len: number,
  half: number,
  tail: number,
) {
  const a = (deg * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const tip = [rnd(cx + c * len), rnd(cy + s * len)];
  const bl = [rnd(cx - s * half), rnd(cy + c * half)];
  const br = [rnd(cx + s * half), rnd(cy - c * half)];
  const tl = [rnd(cx - c * tail), rnd(cy - s * tail)];
  return `${bl[0]},${bl[1]} ${tip[0]},${tip[1]} ${br[0]},${br[1]} ${tl[0]},${tl[1]}`;
}

/**
 * Segmented health-score gauge with a triangle needle (style ④). The 270° dial
 * is split into red / amber / lime / emerald zones so the needle's zone reads at
 * a glance; the needle, arc fill and number animate from 0 → score on mount
 * (easeOutCubic). Pure SVG; respects prefers-reduced-motion.
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
  const target = Math.max(0, Math.min(100, score));
  const [val, setVal] = useState(0);

  useEffect(() => {
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const duration = reduce ? 1 : 900;
    let raf = 0;
    let startTs = 0;
    const tick = (ts: number) => {
      if (!startTs) startTs = ts;
      const p = Math.min(1, (ts - startTs) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(target * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);

  const color = HEALTH_COLOR[scoreToState(val).state].hex;
  const stroke = 12;
  const r = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const start = 135;
  const sweep = 270;
  const gap = 2; // degrees between segments
  const angle = start + (val / 100) * sweep;

  const tickOuter = r - stroke / 2 - 2;
  const labelR = r - stroke / 2 - 20;
  // Major (labelled) ticks at the colour-zone thresholds so the numbers annotate
  // where each zone starts; faint uniform minor ticks every 10 for texture.
  const ticks = [
    ...[10, 20, 30, 50, 70, 90].map((t) => ({ t, major: false })),
    ...[0, 40, 60, 80, 100].map((t) => ({ t, major: true })),
  ].map(({ t, major }) => {
    const a = ((start + (t / 100) * sweep) * Math.PI) / 180;
    const innerR = tickOuter - (major ? 8 : 4);
    const c = Math.cos(a);
    const s = Math.sin(a);
    return {
      t,
      major,
      x1: rnd(cx + tickOuter * c),
      y1: rnd(cy + tickOuter * s),
      x2: rnd(cx + innerR * c),
      y2: rnd(cy + innerR * s),
      lx: rnd(cx + labelR * c),
      ly: rnd(cy + labelR * s),
    };
  });

  return (
    <div
      className={cn("relative inline-flex", className)}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size}>
        {/* colour-zone segments */}
        {ZONES.map(([z0, z1, hex]) => (
          <path
            key={z0}
            d={arcPath(
              cx,
              cy,
              r,
              start + (z0 / 100) * sweep + (z0 > 0 ? gap : 0),
              start + (z1 / 100) * sweep - (z1 < 100 ? gap : 0),
            )}
            fill="none"
            stroke={hex}
            strokeWidth={stroke}
            strokeLinecap="butt"
            opacity={0.9}
          />
        ))}
        {/* tick marks + number labels */}
        {ticks.map((m) => (
          <g key={m.t}>
            <line
              x1={m.x1}
              y1={m.y1}
              x2={m.x2}
              y2={m.y2}
              stroke="currentColor"
              className={m.major ? "text-muted-foreground/70" : "text-muted-foreground/40"}
              strokeWidth={m.major ? 1.8 : 1}
            />
            {m.major && (
              <text
                x={m.lx}
                y={m.ly}
                textAnchor="middle"
                dominantBaseline="central"
                className="fill-muted-foreground"
                style={{
                  fontSize: Math.max(7, Math.round(size * 0.054)),
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {m.t}
              </text>
            )}
          </g>
        ))}
        {/* triangle needle (neutral, for contrast against the zones) */}
        <polygon
          points={needlePoints(cx, cy, angle, r - stroke - 6, 5, 12)}
          className="fill-foreground"
          style={{ filter: "drop-shadow(0 1px 1.5px rgba(0,0,0,.35))" }}
        />
        <circle cx={cx} cy={cy} r={7} className="fill-foreground" />
        <circle cx={cx} cy={cy} r={3} className="fill-card" />
      </svg>
      <div
        className="absolute inset-0 flex flex-col items-center justify-end"
        style={{ paddingBottom: Math.round(size * 0.1) }}
      >
        <span
          className="font-bold leading-none tabular-nums"
          style={{ color, fontSize: Math.round(size * 0.2) }}
        >
          {Math.round(val)}
        </span>
        <span
          className="uppercase leading-none tracking-wide text-muted-foreground"
          style={{ fontSize: Math.max(8, Math.round(size * 0.058)), marginTop: 2 }}
        >
          Health Score
        </span>
      </div>
    </div>
  );
}
