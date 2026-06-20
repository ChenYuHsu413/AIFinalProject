"""Project path & configuration helpers.

Centralises the location of the repository root and the YAML config so the rest
of the code never relies on hard-coded absolute paths.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    """Return the repository root (the directory that owns config.yaml)."""
    return Path(__file__).resolve().parents[2]


def resolve(relative: str | Path) -> Path:
    """Resolve a path that is relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (project_root() / p)


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load ``config.yaml`` (cached). Pass an explicit path for testing."""
    path = Path(config_path) if config_path else project_root() / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs() -> None:
    """Create the standard output directories if they do not already exist."""
    cfg = load_config()
    for key in ("outputs_figures", "outputs_metrics", "outputs_models", "outputs_reports"):
        resolve(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
