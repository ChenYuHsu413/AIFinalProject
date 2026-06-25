"""Optional, whitelist-only crawler for the knowledge base.

Disabled by default (``sources.yaml: enabled: false``) so the demo runs fully
offline.  When enabled it:
  * fetches ONLY the URLs listed in sources.yaml,
  * checks robots.txt for each URL and skips disallowed ones,
  * waits ``request_delay_seconds`` between requests,
  * writes cleaned text to ``knowledge_dir`` as markdown for the RAG index.

``requests`` is imported lazily so the package is not required unless crawling.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import yaml

from src.knowledge.cleaner import html_to_text
from src.utils.paths import load_config, resolve


def _robots_ok(url: str, user_agent: str) -> bool:
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False  # fail closed: if robots can't be read, don't crawl


def crawl() -> List[Dict[str, str]]:
    """Crawl the whitelisted sources. Returns a list of saved-doc records."""
    cfg = load_config()["knowledge"]
    sources_path = resolve(cfg["sources_yaml"])
    if not sources_path.exists():
        print("找不到 sources.yaml；略過爬取。")
        return []
    spec = yaml.safe_load(sources_path.read_text(encoding="utf-8")) or {}
    if not spec.get("enabled", False):
        print("sources.yaml enabled=false；離線模式，略過爬取。")
        return []

    try:
        import requests  # type: ignore
    except Exception:
        print("未安裝 requests；無法爬取（離線 demo 不需要）。")
        return []

    ua = spec.get("user_agent", "FinalProject-KB-Crawler/0.1")
    delay = float(spec.get("request_delay_seconds", 2))
    kb_dir = resolve(cfg["knowledge_dir"])
    kb_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, str]] = []
    for src in spec.get("sources", []) or []:
        url, topic = src.get("url"), src.get("topic", "crawled")
        if not url:
            continue
        if not _robots_ok(url, ua):
            print(f"robots.txt 不允許或無法讀取，略過：{url}")
            continue
        try:
            resp = requests.get(url, headers={"User-Agent": ua}, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print(f"抓取失敗（{type(e).__name__}），略過：{url}")
            continue
        text = html_to_text(resp.text)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic).strip("_") or "crawled"
        out = kb_dir / f"crawled_{slug}.md"
        out.write_text(f"# {topic}\n\n來源：{url}\n\n{text}", encoding="utf-8")
        saved.append({"url": url, "topic": topic, "file": out.name})
        print(f"已存：{out.name}（{len(text)} 字）")
        time.sleep(delay)
    return saved


if __name__ == "__main__":
    crawl()
