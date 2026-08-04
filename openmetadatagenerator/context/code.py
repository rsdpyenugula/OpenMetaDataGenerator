"""Code context provider.

Given a path to a code repository (SQL, Python, dbt models, ...), it indexes the
files and attaches, to each table, the snippets most relevant to that table — both by
exact filename/identifier match (a file that mentions the table name) and by semantic
retrieval. This surfaces the transformation logic that produced a table, which is the
single strongest signal for an accurate description.
"""
from __future__ import annotations

from pathlib import Path

from ..model import Table
from .base import ContextProvider
from .embedding import EmbeddingIndex

_CODE_EXT = (".sql", ".py", ".scala", ".java", ".yml", ".yaml", ".sh", ".r")


class CodeContext(ContextProvider):
    name = "code"

    def __init__(self, code_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_chars: int = 4000, top_k: int = 3):
        self.root = Path(code_path)
        self.max_chars = max_chars
        self.top_k = top_k
        self.index = EmbeddingIndex(embed_model)
        self._chunks: list[tuple[str, str]] = []  # (path, text)

    def _load_corpus(self) -> None:
        if self._chunks or not self.root.exists():
            return
        for f in self.root.rglob("*"):
            if f.is_file() and f.suffix.lower() in _CODE_EXT:
                try:
                    text = f.read_text(errors="ignore")[: self.max_chars]
                except Exception:
                    continue
                if text.strip():
                    self._chunks.append((str(f.relative_to(self.root)), text))
        self.index.build([f"{p}\n{t}" for p, t in self._chunks])

    def attach(self, tables: list[Table]) -> None:
        self._load_corpus()
        if not self._chunks:
            return
        paths = [p for p, _ in self._chunks]
        for t in tables:
            hits: list[str] = []
            # 1) exact: a file whose name references the table.
            for p, text in self._chunks:
                stem = Path(p).stem.lower()
                if t.name.lower() in stem or stem in t.name.lower():
                    hits.append(f"[{p}]\n{text}")
            # 2) semantic: retrieve by table name + column names.
            query = f"{t.schema}.{t.name}: " + ", ".join(c.name for c in t.columns[:30])
            for chunk, score in self.index.query(query, k=self.top_k):
                if score > 0.2 and chunk not in hits:
                    hits.append(chunk)
            if hits:
                t.code_context = "\n\n".join(hits)[: self.max_chars * self.top_k]
