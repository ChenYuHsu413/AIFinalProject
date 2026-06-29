"use client";

/**
 * Pre-paint theme initialiser. Applies the persisted (localStorage) theme — or
 * the system preference when none is stored — before the first paint, so there
 * is no flash of the wrong theme.
 *
 * Rendered as `type="text/javascript"` on the server (so it executes during HTML
 * parsing, before paint) and `type="text/plain"` on the client (inert — it
 * already ran on initial load). This keeps the no-FOUC behaviour while avoiding
 * React's dev-only "Encountered a script tag while rendering" warning. See the
 * Next.js "preventing flash before hydration" guide.
 */
const THEME_INIT = `(function(){try{var s=localStorage.getItem("theme");var d=s?s==="dark":window.matchMedia("(prefers-color-scheme: dark)").matches;var e=document.documentElement;e.classList.toggle("dark",d);e.style.colorScheme=d?"dark":"light";}catch(e){document.documentElement.classList.add("dark");}})();`;

export function ThemeScript() {
  return (
    <script
      type={typeof window === "undefined" ? "text/javascript" : "text/plain"}
      suppressHydrationWarning
      dangerouslySetInnerHTML={{ __html: THEME_INIT }}
    />
  );
}
