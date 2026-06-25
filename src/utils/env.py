"""Tiny ``.env`` loader (standard library only — no python-dotenv dependency).

Reads ``<project-root>/.env`` and copies its ``KEY=value`` pairs into
``os.environ`` for keys that are not already set (empty values are skipped, so a
blank line in ``.env`` never shadows a real environment variable).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.utils.paths import project_root

_LOADED = False


def load_dotenv(path: Optional[str | Path] = None, override: bool = False) -> None:
    global _LOADED
    if _LOADED and path is None and not override:
        return
    p = Path(path) if path else project_root() / ".env"
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and (override or key not in os.environ):
                os.environ[key] = val
    if path is None:
        _LOADED = True
