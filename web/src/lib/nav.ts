import {
  BarChart3,
  BellRing,
  BookOpen,
  Bot,
  Dna,
  FileBarChart,
  FlaskConical,
  Gauge,
  HeartPulse,
  Info,
  LayoutDashboard,
  Library,
  Lightbulb,
  Rocket,
  Search,
  Target,
  TrendingDown,
  Upload,
  Zap,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  label: string;
  href: string;
  icon: LucideIcon;
}

export type AccentKey = "violet" | "blue" | "emerald" | "amber" | "rose" | "slate";

export interface NavGroup {
  /** Group heading; null for ungrouped top/bottom items. */
  title: string | null;
  items: NavItem[];
  accent: AccentKey;
  /** Supplementary module (A/B/B+/C) — grouped under one collapsible expander. */
  supplementary?: boolean;
}

/** Literal Tailwind class strings per accent (kept literal so the JIT keeps them). */
export const ACCENTS: Record<
  AccentKey,
  {
    chip: string;
    dot: string;
    hover: string;
    text: string;
    active: string;
    tint: string;
  }
> = {
  violet: {
    chip: "bg-violet-500/15 text-violet-300 ring-1 ring-inset ring-violet-500/25",
    dot: "bg-violet-400",
    hover: "hover:border-violet-400/40 hover:bg-violet-500/10",
    text: "text-violet-300",
    active: "bg-violet-500/15 text-violet-100 ring-1 ring-inset ring-violet-400/40",
    tint: "hover:bg-violet-500/10 hover:text-violet-200",
  },
  blue: {
    chip: "bg-sky-500/15 text-sky-300 ring-1 ring-inset ring-sky-500/25",
    dot: "bg-sky-400",
    hover: "hover:border-sky-400/40 hover:bg-sky-500/10",
    text: "text-sky-300",
    active: "bg-sky-500/15 text-sky-100 ring-1 ring-inset ring-sky-400/40",
    tint: "hover:bg-sky-500/10 hover:text-sky-200",
  },
  emerald: {
    chip: "bg-emerald-500/15 text-emerald-300 ring-1 ring-inset ring-emerald-500/25",
    dot: "bg-emerald-400",
    hover: "hover:border-emerald-400/40 hover:bg-emerald-500/10",
    text: "text-emerald-300",
    active: "bg-emerald-500/15 text-emerald-100 ring-1 ring-inset ring-emerald-400/40",
    tint: "hover:bg-emerald-500/10 hover:text-emerald-200",
  },
  amber: {
    chip: "bg-amber-500/15 text-amber-300 ring-1 ring-inset ring-amber-500/25",
    dot: "bg-amber-400",
    hover: "hover:border-amber-400/40 hover:bg-amber-500/10",
    text: "text-amber-300",
    active: "bg-amber-500/15 text-amber-100 ring-1 ring-inset ring-amber-400/40",
    tint: "hover:bg-amber-500/10 hover:text-amber-200",
  },
  rose: {
    chip: "bg-rose-500/15 text-rose-300 ring-1 ring-inset ring-rose-500/25",
    dot: "bg-rose-400",
    hover: "hover:border-rose-400/40 hover:bg-rose-500/10",
    text: "text-rose-300",
    active: "bg-rose-500/15 text-rose-100 ring-1 ring-inset ring-rose-400/40",
    tint: "hover:bg-rose-500/10 hover:text-rose-200",
  },
  slate: {
    chip: "bg-slate-500/15 text-slate-300 ring-1 ring-inset ring-slate-500/25",
    dot: "bg-slate-400",
    hover: "hover:border-slate-400/40 hover:bg-slate-500/10",
    text: "text-slate-300",
    active: "bg-primary/15 text-primary ring-1 ring-inset ring-primary/40",
    tint: "hover:bg-slate-500/10 hover:text-slate-200",
  },
};

/** Mirrors the Streamlit NAV_GROUPS so the two UIs stay in sync. */
export const NAV_GROUPS: NavGroup[] = [
  {
    title: null,
    accent: "slate",
    items: [{ label: "總覽 Overview", href: "/", icon: LayoutDashboard }],
  },
  {
    title: "伺服馬達健康（主線）",
    accent: "violet",
    items: [
      { label: "Servo 健康儀表板", href: "/servo/dashboard", icon: Gauge },
      { label: "AI 訓練模擬器", href: "/servo/simulator", icon: FlaskConical },
      { label: "馬達欄位解釋", href: "/servo/glossary", icon: BookOpen },
      { label: "維修知識庫", href: "/servo/knowledge", icon: Library },
      { label: "LLM 維護助理", href: "/servo/assistant", icon: Bot },
    ],
  },
  {
    title: "運維中心",
    accent: "blue",
    items: [
      { label: "告警 / 工單", href: "/alerts", icon: BellRing },
      { label: "報表中心", href: "/reports", icon: FileBarChart },
    ],
  },
  {
    title: "模組 A · 靜態風險 (AI4I)",
    accent: "blue",
    supplementary: true,
    items: [
      { label: "手動單筆預測", href: "/module-a/predict", icon: Target },
      { label: "What-if 敏感度分析", href: "/module-a/what-if", icon: Lightbulb },
      { label: "批次 CSV 上傳", href: "/module-a/batch", icon: Upload },
      { label: "模型評估結果", href: "/module-a/evaluation", icon: BarChart3 },
    ],
  },
  {
    title: "模組 B · 動態健康度 (IMS)",
    accent: "emerald",
    supplementary: true,
    items: [
      { label: "健康度總覽", href: "/module-b/overview", icon: HeartPulse },
      { label: "RUL 預測", href: "/module-b/rul", icon: TrendingDown },
      { label: "互動探索", href: "/module-b/explore", icon: Search },
    ],
  },
  {
    title: "模組 B+ · 多軌跡泛化 (XJTU)",
    accent: "amber",
    supplementary: true,
    items: [
      { label: "多軌跡泛化", href: "/module-b-plus/generalization", icon: Dna },
      { label: "B+ 延伸應用", href: "/module-b-plus/applications", icon: Rocket },
    ],
  },
  {
    title: "模組 C · 馬達電流診斷 (Paderborn)",
    accent: "rose",
    supplementary: true,
    items: [{ label: "馬達電流故障診斷", href: "/module-c", icon: Zap }],
  },
  {
    title: null,
    accent: "slate",
    items: [{ label: "關於本專案", href: "/about", icon: Info }],
  },
];
