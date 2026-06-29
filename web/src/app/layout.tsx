import type { Metadata } from "next";
import { Geist, Geist_Mono, Noto_Sans_TC } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";
import { Header } from "@/components/header";
import { SidebarProvider } from "@/components/sidebar-context";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// 思源黑體（Source Han Sans / Noto Sans CJK 繁中版）— SIL OFL 1.1，免費可商用。
// CJK 字檔較大，關閉 preload，由瀏覽器依 unicode-range 按需載入。
const notoSansTC = Noto_Sans_TC({
  variable: "--font-noto-sans-tc",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  display: "swap",
  preload: false,
});

export const metadata: Metadata = {
  title: "AI Servo Motor Health Command Center",
  description:
    "AI 伺服馬達健康監控與智慧維護指揮中心 — FastAPI + Next.js 前端",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${notoSansTC.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        {/* Apply the persisted / system theme before first paint (no FOUC).
            Defaults to the OS preference; an explicit choice is stored in
            localStorage by ThemeToggle. See Next.js "preventing flash" guide. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var s=localStorage.getItem("theme");var d=s?s==="dark":window.matchMedia("(prefers-color-scheme: dark)").matches;var e=document.documentElement;e.classList.toggle("dark",d);e.style.colorScheme=d?"dark":"light";}catch(e){document.documentElement.classList.add("dark");}})();`,
          }}
        />
      </head>
      <body className="min-h-full bg-background text-foreground" suppressHydrationWarning>
        <SidebarProvider>
          <div className="flex h-screen">
            <Sidebar />
            <div className="flex min-w-0 flex-1 flex-col">
              <Header />
              <main className="command-surface flex-1 overflow-y-auto">
                {children}
              </main>
            </div>
          </div>
        </SidebarProvider>
      </body>
    </html>
  );
}
