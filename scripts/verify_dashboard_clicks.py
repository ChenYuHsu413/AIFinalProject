"""Drive the Streamlit dashboard via Playwright and verify navigation:
the homepage tiles jump to the right page, the sidebar button nav syncs
its highlight (active = primary button), and single-item groups (模組 B+)
stay reachable after navigating away and back.

Outputs screenshots to outputs/screenshots/dashboard_verify/.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright, Page, TimeoutError as PWTimeout

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

URL = "http://127.0.0.1:8501"
OUT = Path("outputs/screenshots/dashboard_verify")
OUT.mkdir(parents=True, exist_ok=True)

# (tile-button-text, main-content-marker-after-click, sidebar-nav-label)
TILES = [
    ("模組 A · 單筆風險預測", "輸入運轉條件", "手動單筆預測"),
    ("模組 B · 健康度總覽", "健康度總覽", "健康度總覽"),
    ("模組 B+ · 多軌跡泛化", "多軌跡泛化", "多軌跡泛化"),
]


def wait_idle(page: Page, ms: int = 1500) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except PWTimeout:
        pass
    page.wait_for_timeout(ms)


def screenshot(page: Page, name: str) -> Path:
    p = OUT / f"{name}.png"
    page.screenshot(path=str(p), full_page=True)
    print(f"  saved {p}  ({p.stat().st_size // 1024} KB)")
    return p


SIDEBAR = 'section[data-testid="stSidebar"]'


def sidebar_active_text(page: Page) -> str:
    """Text of the highlighted (primary) sidebar nav button, '' if none."""
    try:
        return page.locator(
            f'{SIDEBAR} button[kind="primary"]'
        ).first.inner_text(timeout=2_000).strip()
    except Exception:
        return ""


def click_sidebar(page: Page, label: str) -> bool:
    """Click the sidebar nav button whose text contains ``label``."""
    try:
        page.locator(f"{SIDEBAR} button").filter(
            has_text=label
        ).first.click(timeout=3_000)
        return True
    except Exception as e:
        print(f"  ERR sidebar click {label!r}: {e}")
        return False


def main() -> int:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        print(f"[1] open {URL}")
        page.goto(URL, wait_until="domcontentloaded", timeout=30_000)
        wait_idle(page, 3000)

        try:
            page.wait_for_selector("text=快速入口", timeout=20_000)
        except PWTimeout:
            print("  ERR: dashboard '快速入口' section never appeared")
            screenshot(page, "00_load_fail")
            return 2

        screenshot(page, "00_home_initial")

        active = sidebar_active_text(page)
        print(f"  sidebar active: {active!r}  (expected to contain '首頁總覽')")
        if "首頁總覽" not in active:
            print("  WARN: sidebar doesn't show 首頁總覽 active on initial load")

        all_ok = True
        slugs = ["a", "b", "bplus"]
        for i, (btn_text, marker, menu_label) in enumerate(TILES, start=1):
            print(f"[{i+1}] click tile: {btn_text!r}")
            btn_locator = page.get_by_role("button").filter(has_text=btn_text)
            count = btn_locator.count()
            print(f"  found {count} matching button(s)")
            if count == 0:
                print(f"  ERR no button matches {btn_text!r}")
                screenshot(page, f"{i:02d}_no_button")
                all_ok = False
                continue
            try:
                btn_locator.first.click(timeout=10_000)
            except Exception as e:
                print(f"  ERR clicking {btn_text!r}: {e}")
                screenshot(page, f"{i:02d}_click_fail")
                all_ok = False
                continue

            wait_idle(page, 2000)

            # Verify the page content changed.
            try:
                body_text = page.locator('div[data-testid="stMain"]').first.inner_text(timeout=3_000)
            except Exception:
                body_text = page.locator("body").inner_text()
            page_ok = marker in body_text

            active = sidebar_active_text(page)
            menu_ok = menu_label in active

            screenshot(page, f"{i:02d}_after_click_{slugs[i-1]}")
            print(f"  marker '{marker}' in main: {page_ok}")
            print(f"  sidebar active: {active!r}  (expected {menu_label!r}) → {menu_ok}")

            if not (page_ok and menu_ok):
                all_ok = False

            print("  -> back to dashboard via sidebar 首頁總覽")
            if not click_sidebar(page, "首頁總覽"):
                all_ok = False
                break
            wait_idle(page, 2000)

        # Regression for the reported bug: a single-item group (模組 B+) must
        # stay reachable after navigating away and back via the sidebar.
        print("[5] round-trip: 多軌跡泛化 → 健康度總覽 → 多軌跡泛化 (sidebar only)")
        rt_ok = True
        for label, marker in [("多軌跡泛化", "多軌跡泛化"),
                              ("健康度總覽", "健康度總覽"),
                              ("多軌跡泛化", "多軌跡泛化")]:
            if not click_sidebar(page, label):
                rt_ok = False
                break
            wait_idle(page, 1500)
            try:
                body_text = page.locator('div[data-testid="stMain"]').first.inner_text(timeout=3_000)
            except Exception:
                body_text = page.locator("body").inner_text()
            active = sidebar_active_text(page)
            step_ok = (marker in body_text) and (label in active)
            print(f"  click {label!r} → marker {marker in body_text}, "
                  f"active {active!r} → {step_ok}")
            rt_ok = rt_ok and step_ok
        if not rt_ok:
            print("  FAIL: single-item group not reachable on round-trip")
            all_ok = False
        screenshot(page, "98_bplus_round_trip")

        screenshot(page, "99_final_back_home")
        browser.close()

        print(f"\n{'PASS' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
