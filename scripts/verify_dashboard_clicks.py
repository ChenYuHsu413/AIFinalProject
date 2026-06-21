"""Drive the Streamlit dashboard via Playwright and verify tile clicks
navigate to the right page AND that the sidebar option_menu selection
syncs after each click.

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

# (tile-button-text, main-content-marker-after-click, sidebar-option-menu-label)
TILES = [
    ("手動單筆預測", "手動單筆預測", "手動單筆預測"),
    ("What-if 敏感度", "What-if", "What-if 敏感度分析"),
    ("批次 CSV 上傳", "批次", "批次 CSV 上傳"),
    ("模型評估", "模型評估", "模型評估結果"),
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


def get_option_menu_active(page: Page) -> str | None:
    """The option_menu is rendered inside a custom-component iframe.
    Find the iframe whose body contains nav labels, return active item text."""
    for frame in page.frames:
        try:
            txt = frame.locator("body").inner_text(timeout=500)
        except Exception:
            continue
        if "首頁總覽" in txt and "手動單筆預測" in txt:
            # active item gets 'active' class in streamlit-option-menu
            active = frame.locator(".nav-link.active").first
            try:
                return active.inner_text(timeout=500).strip()
            except Exception:
                return None
    return None


def click_sidebar_home(page: Page) -> bool:
    for frame in page.frames:
        try:
            txt = frame.locator("body").inner_text(timeout=500)
        except Exception:
            continue
        if "首頁總覽" in txt and "手動單筆預測" in txt:
            try:
                frame.get_by_text("首頁總覽", exact=False).first.click(timeout=3_000)
                return True
            except Exception as e:
                print(f"  ERR home click in frame: {e}")
                return False
    print("  ERR could not find option_menu frame")
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

        active = get_option_menu_active(page)
        print(f"  option_menu active item: {active!r}  (expected '首頁總覽')")
        if active != "首頁總覽":
            print("  WARN: option_menu doesn't show 首頁總覽 active on initial load")

        all_ok = True
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

            active = get_option_menu_active(page)
            menu_ok = active == menu_label

            screenshot(page, f"{i:02d}_after_click_{['manual','whatif','batch','eval'][i-1]}")
            print(f"  marker '{marker}' in main: {page_ok}")
            print(f"  option_menu active: {active!r}  (expected {menu_label!r}) → {menu_ok}")

            if not (page_ok and menu_ok):
                all_ok = False

            print(f"  -> back to dashboard via sidebar option_menu")
            if not click_sidebar_home(page):
                all_ok = False
                break
            wait_idle(page, 2000)

        screenshot(page, "99_final_back_home")
        browser.close()

        print(f"\n{'PASS' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
