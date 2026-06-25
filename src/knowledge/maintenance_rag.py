"""Maintenance RAG: build a knowledge index from local docs and retrieve
chunks relevant to a query or to a prediction's abnormal features.

Offline-first — reads markdown/text under ``knowledge.knowledge_dir``.  The
index is cached in-process; ``rebuild_index`` re-reads the docs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.knowledge.cleaner import chunk_text, clean_text
from src.knowledge.retriever import TfidfIndex
from src.utils.paths import load_config, resolve

_INDEX: Optional[TfidfIndex] = None


def list_documents() -> List[Dict[str, str]]:
    """List local KB documents with a short preview."""
    cfg = load_config()["knowledge"]
    kb_dir = resolve(cfg["knowledge_dir"])
    docs = []
    if not kb_dir.exists():
        return docs
    for p in sorted(kb_dir.glob("*.md")) + sorted(kb_dir.glob("*.txt")):
        text = clean_text(p.read_text(encoding="utf-8"))
        title = text.splitlines()[0].lstrip("# ").strip() if text else p.stem
        docs.append({"source": p.name, "title": title,
                     "preview": text[:160], "chars": str(len(text))})
    return docs


def build_index(force: bool = False) -> TfidfIndex:
    global _INDEX
    if _INDEX is not None and not force:
        return _INDEX
    cfg = load_config()["knowledge"]
    kb_dir = resolve(cfg["knowledge_dir"])
    chunk_chars = int(cfg.get("chunk_chars", 600))
    overlap = int(cfg.get("chunk_overlap", 100))

    texts: List[str] = []
    metas: List[Dict[str, str]] = []
    if kb_dir.exists():
        for p in sorted(kb_dir.glob("*.md")) + sorted(kb_dir.glob("*.txt")):
            raw = clean_text(p.read_text(encoding="utf-8"))
            title = raw.splitlines()[0].lstrip("# ").strip() if raw else p.stem
            for ch in chunk_text(raw, chunk_chars, overlap):
                texts.append(ch)
                metas.append({"source": p.name, "title": title, "topic": p.stem})
    _INDEX = TfidfIndex(texts=texts, metas=metas).build() if texts else TfidfIndex([], [])
    return _INDEX


def rebuild_index() -> TfidfIndex:
    return build_index(force=True)


def search(query: str, top_k: Optional[int] = None) -> List[Dict[str, object]]:
    cfg = load_config()["knowledge"]
    k = top_k or int(cfg.get("top_k", 4))
    return build_index().search(query, top_k=k)


def retrieve_for_prediction(prediction: Dict[str, Any],
                            top_k: Optional[int] = None) -> List[Dict[str, object]]:
    """Build a query from a prediction's health state + top abnormal features."""
    state = prediction.get("health_state_zh", prediction.get("predicted_health_state", ""))
    hints = [t.get("hint", "") for t in prediction.get("top_features", [])]
    feats = [t.get("feature", "") for t in prediction.get("top_features", [])]
    query = f"伺服馬達 滾珠螺桿 {state} " + " ".join(feats + hints)
    return search(query, top_k=top_k)
