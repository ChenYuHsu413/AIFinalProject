"use client";

import { useEffect, useRef } from "react";

export interface TrendSeries {
  key: string; // row field to read (e.g. "pred_prob" or a subsystem key)
  label: string;
  color: string;
  thick?: boolean;
}

type Row = { global_t: number; [k: string]: number | string | boolean };

/**
 * Lightweight canvas strip chart for the live feed. Draws directly from a ref
 * getter on requestAnimationFrame (no React re-render), so it stays smooth at
 * high update rates where the SVG (recharts) chart janks.
 */
export function CanvasTrend({
  frames,
  series,
  running,
  height = 360,
}: {
  frames: () => ReadonlyArray<Row>;
  series: TrendSeries[];
  running: boolean;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const padL = 30;
    const padR = 10;
    const padT = 10;
    const padB = 18;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = wrap.clientWidth;
      const h = height;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // axis text colour follows the theme (inherited text colour)
      const textColor = getComputedStyle(canvas).color || "#888";
      const grid = "rgba(128,128,128,0.18)";
      const plotL = padL;
      const plotR = w - padR;
      const plotT = padT;
      const plotB = h - padB;
      const yOf = (v: number) =>
        plotT + (1 - Math.max(0, Math.min(1, v))) * (plotB - plotT);

      // y gridlines + labels (0 .. 1)
      ctx.font = "10px system-ui, sans-serif";
      ctx.fillStyle = textColor;
      ctx.strokeStyle = grid;
      ctx.lineWidth = 1;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const gv of [0, 0.25, 0.5, 0.75, 1]) {
        const y = yOf(gv);
        ctx.beginPath();
        ctx.moveTo(plotL, y);
        ctx.lineTo(plotR, y);
        ctx.stroke();
        ctx.globalAlpha = 0.7;
        ctx.fillText(gv.toFixed(2), plotL - 4, y);
        ctx.globalAlpha = 1;
      }

      const rows = frames();
      if (rows.length >= 2) {
        const tMin = rows[0].global_t;
        const tMax = rows[rows.length - 1].global_t;
        const span = tMax - tMin || 1;
        const xOf = (t: number) => plotL + ((t - tMin) / span) * (plotR - plotL);

        // 0.5 alarm threshold
        ctx.strokeStyle = "rgba(239,68,68,0.45)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(plotL, yOf(0.5));
        ctx.lineTo(plotR, yOf(0.5));
        ctx.stroke();
        ctx.setLineDash([]);

        // series lines
        for (const s of series) {
          ctx.strokeStyle = s.color;
          ctx.lineWidth = s.thick ? 2.5 : 1.4;
          ctx.beginPath();
          for (let i = 0; i < rows.length; i++) {
            const v = Number(rows[i][s.key] ?? 0);
            const x = xOf(rows[i].global_t);
            const y = yOf(v);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }

        // x end labels
        ctx.fillStyle = textColor;
        ctx.globalAlpha = 0.7;
        ctx.textBaseline = "bottom";
        ctx.textAlign = "left";
        ctx.fillText(`${tMin.toFixed(0)}s`, plotL + 1, h - 4);
        ctx.textAlign = "right";
        ctx.fillText(`${tMax.toFixed(0)}s`, plotR - 1, h - 4);
        ctx.globalAlpha = 1;
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // redraw loop is self-sustaining; `running` toggles re-mount via key upstream
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, series]);

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {series.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 rounded-sm"
              style={{ width: s.thick ? 14 : 10, backgroundColor: s.color }}
            />
            {s.label}
          </span>
        ))}
      </div>
      <div ref={wrapRef} style={{ height }}>
        <canvas ref={canvasRef} />
        {!running && (
          <p className="mt-2 text-xs text-muted-foreground">（已暫停，畫面凍結）</p>
        )}
      </div>
    </div>
  );
}
