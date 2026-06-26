"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

interface SidebarState {
  /** Desktop: collapsed to an icon-only rail. */
  collapsed: boolean;
  toggleCollapsed: () => void;
  /** Mobile: slide-in drawer open. */
  mobileOpen: boolean;
  setMobileOpen: (v: boolean) => void;
}

const SidebarCtx = createContext<SidebarState | null>(null);

const STORAGE_KEY = "command-sidebar-collapsed";

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  // restore desktop collapse preference (localStorage is unavailable during SSR,
  // so this must run after mount rather than in a lazy initializer)
  useEffect(() => {
    if (typeof window === "undefined") return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setCollapsed(window.localStorage.getItem(STORAGE_KEY) === "1");
  }, []);

  const toggleCollapsed = () =>
    setCollapsed((c) => {
      const next = !c;
      if (typeof window !== "undefined") {
        window.localStorage.setItem(STORAGE_KEY, next ? "1" : "0");
      }
      return next;
    });

  return (
    <SidebarCtx.Provider
      value={{ collapsed, toggleCollapsed, mobileOpen, setMobileOpen }}
    >
      {children}
    </SidebarCtx.Provider>
  );
}

export function useSidebar(): SidebarState {
  const ctx = useContext(SidebarCtx);
  if (!ctx) throw new Error("useSidebar must be used within <SidebarProvider>");
  return ctx;
}
