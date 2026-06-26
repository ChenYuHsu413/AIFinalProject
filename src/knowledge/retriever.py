"""TF-IDF retriever (sklearn-only, language-agnostic).

Uses a character n-gram analyzer so it works for Chinese text without a word
segmenter and without adding a BM25 dependency.  Cosine similarity over TF-IDF
vectors approximates lexical relevance well enough for a small KB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TfidfIndex:
    texts: List[str]
    metas: List[Dict[str, str]]
    vectorizer: TfidfVectorizer = field(default=None)  # type: ignore[assignment]
    matrix: object = field(default=None)

    def build(self) -> "TfidfIndex":
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                          min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.texts)
        return self

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, object]]:
        if not self.texts or self.matrix is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        order = sims.argsort()[::-1][:top_k]
        out = []
        for i in order:
            if sims[i] <= 0:
                continue
            out.append({
                "text": self.texts[i],
                "score": round(float(sims[i]), 4),
                **self.metas[i],
            })
        return out
