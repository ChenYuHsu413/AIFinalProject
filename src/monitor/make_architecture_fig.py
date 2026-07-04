"""Render the Live Monitor system-architecture diagram to PNG + SVG (report asset).

Self-contained (no external framework CSS): draws boxes/arrows with matplotlib so
the figure opens standalone and embeds in Word/PDF. Chinese labels use a CJK font
if available (Windows: Microsoft JhengHei), else falls back gracefully.

Run: python -m src.monitor.make_architecture_fig
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.utils.paths import resolve  # noqa: E402

# Prefer a CJK-capable font so the Chinese labels render (not tofu).
for _f in ["Microsoft JhengHei", "Microsoft YaHei", "PMingLiU", "SimHei", "Noto Sans CJK TC"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# muted palette: (edge, fill)
C = {
    "amber": ("#B8860B", "#FBEFD8"),
    "gray": ("#8A9099", "#EEF0F2"),
    "blue": ("#3C6FD0", "#E4ECFB"),
    "teal": ("#1E8F84", "#DDF1EE"),
    "red": ("#C64A46", "#FBE6E5"),
    "purple": ("#6E4CB0", "#ECE6F7"),
    "green": ("#2E8B57", "#E1F1E7"),
}
TEXT = "#1f2937"


def box(ax, x, y, w, h, title, sub, color, dashed=False):
    edge, fill = C[color]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                       linewidth=1.4, edgecolor=edge, facecolor=fill,
                       linestyle=(0, (5, 4)) if dashed else "solid")
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + 0.9, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color=TEXT)
        ax.text(cx, cy - 1.0, sub, ha="center", va="center", fontsize=8.5, color="#4b5563")
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color=TEXT)


def arrow(ax, x1, y1, x2, y2, dashed=False, color="#6b7280"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 72)
    ax.axis("off")

    # data sources
    box(ax, 8, 58, 34, 10, "現在：合成產生器", "假 sensor 訊號", "amber")
    ax.annotate("", xy=(52.5, 62.5), xytext=(42.5, 62.5),
                arrowprops=dict(arrowstyle="<|-|>", color="#6b7280", lw=1.6))
    ax.text(47.5, 65, "可替換", ha="center", va="center", fontsize=8.5, color="#6b7280")
    box(ax, 53, 58, 46, 10, "未來：ESP32／PLC", "真實感測器（換上即接）", "gray", dashed=True)

    # pluggable adapter bar
    box(ax, 8, 47, 91, 6.5, "可插拔資料源介面（adapter）", "", "gray")
    arrow(ax, 25, 58, 25, 53.5)
    arrow(ax, 76, 58, 76, 53.5, dashed=True)

    # pipeline
    arrow(ax, 19, 47, 19, 41)
    box(ax, 8, 29, 24, 12, "即時串流", "SSE /monitor/stream", "blue")
    box(ax, 38, 29, 24, 12, "模型推論", "早期預警／判斷", "teal")
    box(ax, 68, 29, 22, 12, "告警引擎", "門檻 → 通知", "red")
    box(ax, 96, 29, 20, 12, "監控介面", "雷達／趨勢／燈", "purple")
    arrow(ax, 32, 35, 38, 35)
    arrow(ax, 62, 35, 68, 35)
    arrow(ax, 90, 35, 96, 35)

    # PHM model insertion
    box(ax, 38, 12, 24, 11, "真實 PHM 106GB", "訓練好的模型", "green")
    arrow(ax, 50, 23, 50, 29, dashed=True, color="#2E8B57")
    ax.text(63.5, 26, "← 真實模型插入點", ha="left", va="center", fontsize=9, color="#2E8B57")

    ax.text(8, 6, "虛線＝未來／尚未接上", ha="left", va="center", fontsize=8.5, color="#6b7280")
    ax.set_title("Live Monitor 系統架構：可插拔資料源 → 即時串流 → 告警；真實 PHM 模型可插入",
                 fontsize=12.5, fontweight="bold", color=TEXT, pad=12)

    out_dir = resolve("outputs/figures/monitor_v4")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"system_architecture.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[arch] -> {out_dir}/system_architecture.png (+ .svg)")


if __name__ == "__main__":
    main()
