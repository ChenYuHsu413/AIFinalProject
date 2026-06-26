"""Text cleaning + chunking for the knowledge base.

Kept dependency-light: HTML is stripped with a small regex fallback if
BeautifulSoup is unavailable, so the offline markdown path needs nothing extra.
"""
from __future__ import annotations

import re
from typing import List


def html_to_text(html: str) -> str:
    """Strip HTML to plain text (BeautifulSoup if present, else regex fallback)."""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                      flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    return clean_text(text)


def clean_text(text: str) -> str:
    """Collapse whitespace and drop empty lines."""
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def chunk_text(text: str, chunk_chars: int = 600, overlap: int = 100) -> List[str]:
    """Split text into overlapping character windows on paragraph boundaries."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= chunk_chars:
            buf = f"{buf}\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            # carry an overlap tail for context continuity
            tail = buf[-overlap:] if overlap and buf else ""
            buf = f"{tail}\n{p}" if tail else p
    if buf:
        chunks.append(buf)
    return chunks
