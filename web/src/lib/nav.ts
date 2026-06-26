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

export interface NavGroup {
  /** Group heading; null for ungrouped top/bottom items. */
  title: string | null;
  items: NavItem[];
}

/** Mirrors the Streamlit NAV_GROUPS so the two UIs stay in sync. */
export const NAV_GROUPS: NavGroup[] = [
  {
    title: null,
    items: [{ label: "首頁總覽", href: "/", icon: LayoutDashboard }],
  },
  {
    title: "模組 Servo · 伺服馬達健康（主線）",
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
    items: [
      { label: "手動單筆預測", href: "/module-a/predict", icon: Target },
      { label: "What-if 敏感度分析", href: "/module-a/what-if", icon: Lightbulb },
      { label: "批次 CSV 上傳", href: "/module-a/batch", icon: Upload },
      { label: "模型評估結果", href: "/module-a/evaluation", icon: BarChart3 },
    ],
  },
  {
    title: "模組 B · 動態健康度 (IMS)",
    items: [
      { label: "健康度總覽", href: "/module-b/overview", icon: HeartPulse },
      { label: "RUL 預測", href: "/module-b/rul", icon: TrendingDown },
      { label: "互動探索", href: "/module-b/explore", icon: Search },
    ],
  },
  {
    title: "模組 B+ · 多軌跡泛化 (XJTU)",
    items: [
      { label: "多軌跡泛化", href: "/module-b-plus/generalization", icon: Dna },
      { label: "B+ 延伸應用", href: "/module-b-plus/applications", icon: Rocket },
    ],
  },
  {
    title: "模組 C · 馬達電流診斷 (Paderborn)",
    items: [{ label: "馬達電流故障診斷", href: "/module-c", icon: Zap }],
  },
  {
    title: null,
    items: [{ label: "關於本專案", href: "/about", icon: Info }],
  },
];
