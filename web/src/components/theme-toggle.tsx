"use client";

import { useSyncExternalStore } from "react";
import { Moon, Sun } from "lucide-react";

/**
 * Dark / light theme toggle. The initial theme is applied pre-paint by the
 * inline script in the root layout (system preference unless a choice is
 * stored). This button flips the `dark` class on <html> and persists the choice;
 * the icon reads the live DOM state via useSyncExternalStore (no flash, no
 * setState-in-effect).
 */
function subscribe(callback: () => void) {
  const observer = new MutationObserver(callback);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  return () => observer.disconnect();
}

export function ThemeToggle() {
  const dark = useSyncExternalStore(
    subscribe,
    () => document.documentElement.classList.contains("dark"),
    () => true, // server snapshot: matches the dark-by-default identity
  );

  function toggle() {
    const next = !dark;
    const el = document.documentElement;
    el.classList.toggle("dark", next);
    el.style.colorScheme = next ? "dark" : "light";
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      /* localStorage unavailable — keep in-memory only */
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="切換深淺色主題"
      title={dark ? "切換為淺色模式" : "切換為深色模式"}
      className="flex h-7 w-7 items-center justify-center rounded-md border border-border/70 bg-card/50 text-muted-foreground transition-colors hover:text-foreground"
    >
      {dark ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
    </button>
  );
}
