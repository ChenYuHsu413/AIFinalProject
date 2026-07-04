"use client";

import { useEffect, useRef, useState } from "react";
import { Activity, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";
import { CNN_FEATMAP as D, type CnnLevel } from "@/lib/cnnFeatmap";

const CH = ["#f97316", "#14b8a6", "#a855f7"]; // channel line colours
const SPEEDS = [0.5, 1, 2];
const LEVEL_TONE: Record<string, string> = { LN: "#22c55e", LO: "#eab308", MED: "#f97316", HI: "#ef4444" };

/** Animated canvas: a conv kernel slides over the real waveform, lighting up the
 *  feature map; numbered markers show where each detector fires. Real activations. */
function SliderCanvas({ levelIdx }: { levelIdx: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const jRef = useRef(0);
  const lvRef = useRef(levelIdx);
  const playRef = useRef(true);
  const spdRef = useRef(1);
  const [playing, setPlaying] = useState(true);
  const [spd, setSpd] = useState(1);

  useEffect(() => { lvRef.current = levelIdx; jRef.current = 0; }, [levelIdx]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = 680, H = 460, dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = W * dpr;
    canvas.height = H * dpr;

    const { n_in: N, n_feat: M, kernel: K, fmax, detectors } = D;
    const ix0 = 48, ix1 = 664, iyT = 52, iyB = 184, fx0 = 48, fx1 = 664, fyT = 262;
    const rowH = (412 - 262) / 6;
    const xIn = (t: number) => ix0 + (t / (N - 1)) * (ix1 - ix0);
    const xF = (c: number) => fx0 + (c / M) * (fx1 - fx0);
    const cw = (fx1 - fx0) / M;

    const draw = () => {
      const L: CnnLevel = D.levels[lvRef.current];
      const IN = L.input, F = L.filters, peaks = L.peaks;
      let vmin = 1e9, vmax = -1e9;
      IN.forEach((r) => r.forEach((v) => { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }));
      const yIn = (v: number) => iyB - ((v - vmin) / (vmax - vmin)) * (iyB - iyT - 6) - 3;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const mut = getComputedStyle(canvas).color || "#94a3b8";
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.fillStyle = mut;
      ctx.fillText("① 真實波形（尖峰＝能量爆發）· 健康度：", ix0, 24);
      ctx.fillStyle = LEVEL_TONE[L.code]; ctx.font = "bold 12px system-ui";
      ctx.fillText(L.code + " " + L.zh, ix0 + 232, 24);
      ctx.font = "12px system-ui"; ctx.fillStyle = mut;
      let lx = ix1 - 250;
      D.in_channels_zh.forEach((n, k) => {
        ctx.fillStyle = CH[k]; ctx.fillRect(lx, 16, 10, 3);
        ctx.fillStyle = mut; ctx.fillText(n, lx + 14, 22);
        lx += n.length * 12 + 30;
      });
      ctx.strokeStyle = "rgba(128,128,128,0.18)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ix0, iyB); ctx.lineTo(ix1, iyB); ctx.stroke();
      for (let k = 0; k < IN.length; k++) {
        ctx.strokeStyle = CH[k]; ctx.lineWidth = 1.2; ctx.beginPath();
        for (let t = 0; t < N; t++) { const x = xIn(t), yv = yIn(IN[k][t]); t ? ctx.lineTo(x, yv) : ctx.moveTo(x, yv); }
        ctx.stroke();
      }
      // detector markers: numbered dots at each detector's peak position on the waveform
      for (let dted = 0; dted < peaks.length; dted++) {
        const px = xIn(Math.min(N - 1, 2 * peaks[dted]));
        ctx.strokeStyle = "rgba(20,184,166,0.35)"; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, iyT); ctx.lineTo(px, iyB); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = "#0d9488"; ctx.beginPath(); ctx.arc(px, iyT + 2, 8, 0, 7); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.font = "bold 10px system-ui"; ctx.textAlign = "center";
        ctx.fillText(String(dted + 1), px, iyT + 6); ctx.textAlign = "left"; ctx.font = "12px system-ui";
      }
      // sliding receptive field
      const j = jRef.current, c = Math.min(N - 1, 2 * j);
      const bx0 = xIn(Math.max(0, c - (K - 1) / 2)), bx1 = xIn(Math.min(N - 1, c + (K - 1) / 2));
      ctx.fillStyle = "rgba(245,158,11,0.18)"; ctx.fillRect(bx0, iyT - 2, bx1 - bx0, iyB - iyT + 2);
      ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 1.5; ctx.strokeRect(bx0, iyT - 2, bx1 - bx0, iyB - iyT + 2);
      ctx.fillStyle = "#f59e0b"; ctx.fillText("卷積核滑動 →", bx0, iyT - 6);

      // feature map
      ctx.fillStyle = mut;
      ctx.fillText("② 6 個偵測器的特徵圖（越亮＝越像它認得的樣式；圈起＝該偵測器最活躍處）", fx0, fyT - 8);
      for (let r = 0; r < F.length; r++) {
        const ry = fyT + r * rowH;
        for (let col = 0; col <= Math.min(j, M - 1); col++) {
          const a = Math.min(1, F[r][col] / fmax);
          ctx.fillStyle = "rgba(20,184,166," + (0.06 + 0.94 * a) + ")";
          ctx.fillRect(xF(col), ry + 2, cw + 0.6, rowH - 3);
        }
        ctx.strokeStyle = "rgba(128,128,128,0.15)"; ctx.strokeRect(fx0, ry + 2, fx1 - fx0, rowH - 3);
        // peak cell highlight (once revealed)
        if (j >= peaks[r]) {
          ctx.strokeStyle = "#0d9488"; ctx.lineWidth = 1.6;
          ctx.strokeRect(xF(peaks[r]), ry + 2, cw + 0.6, rowH - 3);
        }
        ctx.fillStyle = mut; ctx.textAlign = "right";
        ctx.fillText((r + 1) + " · " + detectors[r].desc, fx0 - 6, ry + rowH / 2 + 4);
        ctx.textAlign = "left";
      }
      // current column + connector
      const cx = xF(Math.min(j, M - 1));
      ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 2; ctx.strokeRect(cx, fyT + 2, cw + 0.6, rowH * 6 - 3);
      ctx.strokeStyle = "rgba(245,158,11,0.7)"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo((bx0 + bx1) / 2, iyB + 2); ctx.lineTo(cx + cw / 2, fyT + 2); ctx.stroke(); ctx.setLineDash([]);
    };

    const step = () => {
      if (playRef.current) jRef.current = jRef.current > M + 6 ? 0 : jRef.current + 1;
      draw();
    };
    draw();
    let timer = setInterval(step, Math.round(90 / spdRef.current));
    const onSpd = () => { clearInterval(timer); timer = setInterval(step, Math.round(90 / spdRef.current)); };
    (canvas as HTMLCanvasElement & { _onSpd?: () => void })._onSpd = onSpd;
    return () => clearInterval(timer);
  }, []);

  const retime = () => (canvasRef.current as (HTMLCanvasElement & { _onSpd?: () => void }) | null)?._onSpd?.();

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
        <button onClick={() => { playRef.current = !playRef.current; setPlaying(playRef.current); }}
          className="rounded-md border border-border/70 px-3 py-1 font-medium">
          {playing ? "⏸ 暫停" : "▶ 播放"}
        </button>
        <button onClick={() => { jRef.current = 0; playRef.current = true; setPlaying(true); }}
          className="rounded-md border border-border/70 px-2.5 py-1 text-muted-foreground">↻ 重播</button>
        <span className="ml-1 text-muted-foreground">速度</span>
        {SPEEDS.map((s) => (
          <button key={s} onClick={() => { spdRef.current = s; setSpd(s); retime(); }}
            className={cn("rounded px-2 py-1 text-xs", spd === s ? "bg-teal-600 text-white" : "bg-muted text-muted-foreground")}>
            {s}×
          </button>
        ))}
      </div>
      <canvas ref={canvasRef} className="w-full text-muted-foreground" style={{ height: "auto" }} />
    </div>
  );
}

export function CnnFeatureSlider() {
  const [open, setOpen] = useState(false);
  const [levelIdx, setLevelIdx] = useState(D.levels.findIndex((l) => l.code === "HI"));
  return (
    <div className="rounded-xl border border-border/70 bg-card/70 shadow-sm backdrop-blur-sm">
      <button type="button" onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-5 py-3 text-sm font-medium">
        <Activity className="h-4 w-4 text-teal-400" />
        1D-CNN 怎麼「看」波形？（動畫教學）
        <ChevronRight className={cn("ml-auto h-4 w-4 transition-transform", open && "rotate-90")} />
      </button>
      {open && (
        <div className="border-t border-border/70 px-5 py-4">
          <div className="mb-3 rounded-lg bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
            <p className="mb-1 font-medium text-foreground">白話：1D-CNN = 一組「圖章」在波形上滑過去蓋章</p>
            每個<b>濾波器（偵測器）就是一枚圖章</b>，刻著一種波形樣式。圖章從左滑到右，比對「框住的這段像不像我」——
            <b>像就蓋亮章</b>。切換<b>健康度</b>看差別：<b>健康(LN)時偵測器多半安靜</b>、
            <b>退化越重、亮章越多</b>。波形上的 <b>①~⑥ 圓點</b>標出每個偵測器<b>最活躍的位置</b>，對應右邊同號的特徵列。全部為真實卷積激活。
          </div>
          {/* level switch */}
          <div className="mb-3 inline-flex flex-wrap gap-1 rounded-lg border border-border/70 bg-card/70 p-1 text-xs">
            {D.levels.map((l, i) => (
              <button key={l.code} onClick={() => setLevelIdx(i)}
                className={cn("rounded-md px-2.5 py-1 font-medium",
                  i === levelIdx ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                {l.code} {l.zh}
              </button>
            ))}
          </div>
          <SliderCanvas levelIdx={levelIdx} />
          <p className="mt-2 text-xs text-muted-foreground">
            資料：真實 PHM 波形窗 → 第一層 1D 卷積（核 7 · 步幅 2）→ 6 濾波器真實激活。
            由 <code>python -m src.models.servo_cnn_featmap</code> 產生。
          </p>
        </div>
      )}
    </div>
  );
}
