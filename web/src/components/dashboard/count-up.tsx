"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Animates a number from its previous value to `value` (easeOutCubic) whenever
 * `value` changes — so it rolls up 0 → value on mount, and smoothly retargets on
 * a later update (e.g. cache → model). Renders the rounded integer; wrap units
 * outside. Respects prefers-reduced-motion. SSR-safe (starts at 0 on both sides).
 */
export function CountUp({
  value,
  duration = 700,
  className,
}: {
  value: number;
  duration?: number;
  className?: string;
}) {
  const [display, setDisplay] = useState(0);
  // Live displayed value, updated only inside the animation tick (not eagerly),
  // so a discarded StrictMode effect run can't clobber the start point.
  const displayRef = useRef(0);

  useEffect(() => {
    const from = displayRef.current;
    const to = value;
    if (from === to) return;

    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const dur = reduce ? 1 : duration;
    let raf = 0;
    let startTs = 0;
    const tick = (ts: number) => {
      if (!startTs) startTs = ts;
      const p = Math.min(1, (ts - startTs) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      const v = from + (to - from) * eased;
      displayRef.current = v;
      setDisplay(v);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);

  return (
    <span className={className} style={{ fontVariantNumeric: "tabular-nums" }}>
      {Math.round(display)}
    </span>
  );
}
