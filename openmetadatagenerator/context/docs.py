"""Document context provider.

Given a path to a documentation corpus (Markdown, text, CSV/TSV requirement sheets,
data dictionaries, ...), it indexes the content and attaches the passages most
relevant to each table. Tabular files (``.csv``/``.tsv``) are flattened row-wise so a
data-dictionary row like ``orders, one row per customer order`` becomes a retrievable
chunk keyed by the entity name.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..model import Table
from .base import ContextProvider
from .embedding import EmbeddingIndex

_DOC_EXT = (".md", ".txt", ".rst", ".csv", ".tsv")


class DocContext(ContextProvider):
    name = "docs"

    def __init__(self, doc_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_chars: int = 3000, top_k: int = 3):
        self.root = Path(doc_path)
        self.max_chars = max_chars
        self.top_k = top_k
        self.index = EmbeddingIndex(embed_model)
        self._chunks: list[str] = []

    def _load_corpus(self) -> None:
        if self._chunks or not self.root.exists():
            return
        for f in self.root.rglob("*"):
            if not (f.is_file() and f.suffix.lower() in _DOC_EXT):
                continue
            try:
                if f.suffix.lower() in (".csv", ".tsv"):
                    delim = "\t" if f.suffix.lower() == ".tsv" else ","
                    with open(f, newline="", errors="ignore") as fh:
                        for row in csv.reader(fh, delimiter=delim):
                            line = " | ".join(c for c in row if c).strip()
                            if line:
                                self._chunks.append(f"[{f.name}] {line}")
                else:
                    text = f.read_text(errors="ignore")
                    for para in text.split("\n\n"):
                        para = para.strip()
                        if len(para) > 20:
                            self._chunks.append(f"[{f.name}] {para[: self.max_chars]}")
            except Exception:
                continue
        self.index.build(self._chunks)

    def attach(self, tables: list[Table]) -> None:
        self._load_corpus()
        if not self._chunks:
            return
        for t in tables:
            query = f"{t.name} {t.schema} " + " ".join(c.name for c in t.columns[:30])
            hits = [ch for ch, score in self.index.query(query, k=self.top_k) if score > 0.2]
            # Also surface any chunk that explicitly names the table.
            for ch in self._chunks:
                if t.name.lower() in ch.lower() and ch not in hits:
                    hits.insert(0, ch)
            if hits:
                t.doc_context = "\n".join(hits[: self.top_k + 2])[: self.max_chars * 2]
