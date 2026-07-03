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
  // Read latest props from refs so the draw loop sets up ONCE (props like
  // `series`/`frames` get fresh identities each render and must not thrash it).
  const framesRef = useRef(frames);
  const seriesRef = useRef(series);
  useEffect(() => {
    framesRef.current = frames;
    seriesRef.current = series;
  });

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

      const rows = framesRef.current();
      const series = seriesRef.current;
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

        // event markers: model-alert (pred crosses 0.5 up) vs ground-truth alarm,
        // first occurrence per scenario, with the lead-time gap shaded + labelled.
        const alertBySid = new Map<number, { x: number; t: number }>();
        const alarmBySid = new Map<number, { x: number; t: number }>();
        for (let i = 1; i < rows.length; i++) {
          const sid = Number(rows[i].sid ?? 0);
          const pPrev = Number(rows[i - 1].pred_prob ?? 0);
          const pCur = Number(rows[i].pred_prob ?? 0);
          const aPrev = Number(rows[i - 1].alarm ?? 0);
          const aCur = Number(rows[i].alarm ?? 0);
          const t = Number(rows[i].global_t);
          if (pPrev < 0.5 && pCur >= 0.5 && !alertBySid.has(sid))
            alertBySid.set(sid, { x: xOf(t), t });
          if (aPrev === 0 && aCur === 1 && !alarmBySid.has(sid))
            alarmBySid.set(sid, { x: xOf(t), t });
        }
        // gap band + lead label (alert -> alarm of the same scenario)
        for (const [sid, am] of alarmBySid) {
          const al = alertBySid.get(sid);
          if (al && am.x > al.x) {
            ctx.fillStyle = "rgba(245,158,11,0.13)";
            ctx.fillRect(al.x, plotT, am.x - al.x, plotB - plotT);
            ctx.fillStyle = "#d97706";
            ctx.font = "bold 11px system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            ctx.fillText(`提前 ${(am.t - al.t).toFixed(1)}s`, (al.x + am.x) / 2, plotT + 2);
          }
        }
        // vertical lines
        for (const [, al] of alertBySid) {
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          ctx.moveTo(al.x, plotT);
          ctx.lineTo(al.x, plotB);
          ctx.stroke();
        }
        ctx.setLineDash([]);
        for (const [, am] of alarmBySid) {
          ctx.strokeStyle = "#ef4444";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(am.x, plotT);
          ctx.lineTo(am.x, plotB);
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

    };

    // setInterval (not requestAnimationFrame) so the chart keeps drawing even
    // when the tab is backgrounded — a monitor view should not freeze.
    draw();
    const id = setInterval(draw, 33); // ~30 fps
    return () => clearInterval(id);
  }, [height]);

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
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-0 border-l-2 border-dashed" style={{ borderColor: "#f59e0b" }} />
          AI 示警
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-0 border-l-2" style={{ borderColor: "#ef4444" }} />
          真實告警
        </span>
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
