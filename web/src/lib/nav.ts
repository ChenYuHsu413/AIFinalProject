import {
  BarChart3,
  BookOpen,
  Bot,
  Dna,
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
  /** Render the group header as a collapsible toggle. */
  collapsible?: boolean;
  /** Initial open state when collapsible (overridden open if a child is active). */
  defaultOpen?: boolean;
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
    chip: "bg-violet-100 text-violet-600",
    dot: "bg-violet-500",
    hover: "hover:border-violet-300 hover:bg-violet-50/60",
    text: "text-violet-600",
    active: "bg-gradient-to-r from-violet-500 to-violet-600 text-white shadow-sm",
    tint: "hover:bg-violet-50 hover:text-violet-700",
  },
  blue: {
    chip: "bg-blue-100 text-blue-600",
    dot: "bg-blue-500",
    hover: "hover:border-blue-300 hover:bg-blue-50/60",
    text: "text-blue-600",
    active: "bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-sm",
    tint: "hover:bg-blue-50 hover:text-blue-700",
  },
  emerald: {
    chip: "bg-emerald-100 text-emerald-600",
    dot: "bg-emerald-500",
    hover: "hover:border-emerald-300 hover:bg-emerald-50/60",
    text: "text-emerald-600",
    active: "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white shadow-sm",
    tint: "hover:bg-emerald-50 hover:text-emerald-700",
  },
  amber: {
    chip: "bg-amber-100 text-amber-600",
    dot: "bg-amber-500",
    hover: "hover:border-amber-300 hover:bg-amber-50/60",
    text: "text-amber-600",
    active: "bg-gradient-to-r from-amber-500 to-amber-600 text-white shadow-sm",
    tint: "hover:bg-amber-50 hover:text-amber-700",
  },
  rose: {
    chip: "bg-rose-100 text-rose-600",
    dot: "bg-rose-500",
    hover: "hover:border-rose-300 hover:bg-rose-50/60",
    text: "text-rose-600",
    active: "bg-gradient-to-r from-rose-500 to-rose-600 text-white shadow-sm",
    tint: "hover:bg-rose-50 hover:text-rose-700",
  },
  slate: {
    chip: "bg-slate-100 text-slate-600",
    dot: "bg-slate-400",
    hover: "hover:border-slate-300 hover:bg-slate-50",
    text: "text-slate-600",
    active: "bg-gradient-to-r from-slate-600 to-slate-700 text-white shadow-sm",
    tint: "hover:bg-slate-100 hover:text-slate-700",
  },
};

/** Mirrors the Streamlit NAV_GROUPS so the two UIs stay in sync. */
export const NAV_GROUPS: NavGroup[] = [
  {
    title: null,
    accent: "slate",
    items: [{ label: "首頁總覽", href: "/", icon: LayoutDashboard }],
  },
  {
    title: "模組 Servo · 伺服馬達健康（主線）",
    accent: "violet",
    collapsible: true,
    defaultOpen: true,
    items: [
      { label: "Servo 健康儀表板", href: "/servo/dashboard", icon: Gauge },
      { label: "AI 訓練模擬器", href: "/servo/simulator", icon: FlaskConical },
      { label: "馬達欄位解釋", href: "/servo/glossary", icon: BookOpen },
      { label: "LLM 維護助理", href: "/servo/assistant", icon: Bot },
      { label: "維修知識庫", href: "/servo/knowledge", icon: Library },
    ],
  },
  {
    title: "模組 A · 靜態風險 (AI4I)",
    accent: "blue",
    collapsible: true,
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
    collapsible: true,
    items: [
      { label: "健康度總覽", href: "/module-b/overview", icon: HeartPulse },
      { label: "RUL 預測", href: "/module-b/rul", icon: TrendingDown },
      { label: "互動探索", href: "/module-b/explore", icon: Search },
    ],
  },
  {
    title: "模組 B+ · 多軌跡泛化 (XJTU)",
    accent: "amber",
    collapsible: true,
    items: [
      { label: "多軌跡泛化", href: "/module-b-plus/generalization", icon: Dna },
      { label: "B+ 延伸應用", href: "/module-b-plus/applications", icon: Rocket },
    ],
  },
  {
    title: "模組 C · 馬達電流診斷 (Paderborn)",
    accent: "rose",
    collapsible: true,
    items: [{ label: "馬達電流故障診斷", href: "/module-c", icon: Zap }],
  },
  {
    title: null,
    accent: "slate",
    items: [{ label: "關於本專案", href: "/about", icon: Info }],
  },
];
