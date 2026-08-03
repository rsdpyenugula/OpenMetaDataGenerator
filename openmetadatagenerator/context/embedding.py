"""Semantic retrieval over a text corpus.

Wraps a sentence-transformer model to (a) embed a corpus of chunks once and (b)
retrieve the top-k chunks most similar to a query (a table's name + columns). This
powers both the code and document context providers and the accuracy scorer.

The dependency is optional; if ``sentence-transformers`` is not installed the index
degrades gracefully to lexical (token-overlap) scoring so the pipeline still runs.
"""
from __future__ import annotations

import math
import re


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class EmbeddingIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._chunks: list[str] = []
        self._emb = None  # numpy array (n, d) or None when using lexical fallback

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = False  # lexical fallback

    def build(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self._load()
        if self._model:
            import numpy as np
            self._emb = np.asarray(self._model.encode(chunks, normalize_embeddings=True)) \
                if chunks else np.zeros((0, 384))

    def query(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        if not self._chunks:
            return []
        if self._model:
            import numpy as np
            q = np.asarray(self._model.encode([text], normalize_embeddings=True))[0]
            sims = self._emb @ q
            idx = sims.argsort()[::-1][:k]
            return [(self._chunks[i], float(sims[i])) for i in idx]
        # lexical fallback: cosine over token sets
        qt = _tokens(text)
        scored = []
        for ch in self._chunks:
            ct = _tokens(ch)
            inter = len(qt & ct)
            denom = math.sqrt(len(qt) * len(ct)) or 1.0
            scored.append((ch, inter / denom))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def cosine(index: EmbeddingIndex, a: str, b: str) -> float:
    """Similarity of two strings using the index's model (or lexical fallback)."""
    index._load()
    if index._model:
        import numpy as np
        e = index._model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(e[0], e[1]))
    at, bt = _tokens(a), _tokens(b)
    if not at or not bt:
        return 0.0
    return len(at & bt) / math.sqrt(len(at) * len(bt))
