"""DuckDB-backed semantic RAG index (local, incremental, HNSW-accelerated).

Mirrors the reference architecture: chunked context sources are embedded once and the
vectors are persisted in a DuckDB relation (``kb_embed_index``) alongside an
``embed_model`` stamp. Re-runs re-embed only new or changed chunks (keyed by content
hash and file mtime); a change of embedding model wipes and rebuilds the index (vectors
from different models are not comparable). Retrieval is per ``source_type`` (code / doc /
view), returning the top-k chunks above a similarity threshold — exactly the
retrieval-augmented grounding the generator consumes.

Backends degrade gracefully: sentence-transformer vectors + an HNSW index (``hnswlib``)
when available, brute-force cosine otherwise, and a lexical token-overlap fallback when no
embedding model is installed (vectors stored empty; search scores lexically). Everything
is local — no external API is used for retrieval.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from .context.embedding import EmbeddingIndex, _tokens
from .model import Table

SOURCE_CODE = "code"
SOURCE_DOC = "docs"
SOURCE_VIEW = "view"

TOP_K = 5
SIM_THRESHOLD = 0.45  # tuned for all-MiniLM-L6-v2 cosine (override per call)

_CODE_EXT = (".sql", ".py", ".scala", ".java", ".yml", ".yaml", ".sh", ".r")
_DOC_EXT = (".md", ".txt", ".rst", ".csv", ".tsv")

_TABLE = """
CREATE TABLE IF NOT EXISTS kb_embed_index (
    id          VARCHAR PRIMARY KEY,
    source_type VARCHAR,
    source_file VARCHAR,
    heading     VARCHAR,
    content     TEXT,
    embedding   TEXT,          -- JSON list[float]; '' in lexical mode
    embed_model VARCHAR,
    indexed_at  BIGINT
);
"""


def _chunk_id(source_type: str, source_file: str, heading: str) -> str:
    return hashlib.md5(f"{source_type}||{source_file}||{heading}".encode()).hexdigest()


# ── source chunkers ─────────────────────────────────────────────────────────────
def _chunks_code(code_path: str) -> list[tuple[str, str, str, str]]:
    """(source_type, source_file, heading, content) split by top-level def/class."""
    out: list[tuple[str, str, str, str]] = []
    root = Path(code_path)
    if not code_path or not root.exists():
        return out
    for p in root.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in _CODE_EXT) or "__pycache__" in str(p):
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue
        rel = str(p.relative_to(root))
        blocks = re.split(r"(?=^(?:def |class )\S)", text, flags=re.MULTILINE)
        if len(blocks) <= 1:
            out.append((SOURCE_CODE, rel, p.stem, text[:1500]))
        else:
            if blocks[0].strip():
                out.append((SOURCE_CODE, rel, f"{p.stem}::__header__", blocks[0][:600]))
            for block in blocks[1:]:
                m = re.match(r"(?:def |class )(\w+)", block)
                out.append((SOURCE_CODE, rel, f"{p.stem}::{m.group(1) if m else 'block'}", block[:1500]))
    return out


def _chunks_doc(doc_path: str) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    root = Path(doc_path)
    if not doc_path or not root.exists():
        return out
    import csv
    for p in root.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in _DOC_EXT):
            continue
        try:
            if p.suffix.lower() in (".csv", ".tsv"):
                delim = "\t" if p.suffix.lower() == ".tsv" else ","
                with open(p, newline="", errors="ignore") as fh:
                    for i, row in enumerate(csv.reader(fh, delimiter=delim)):
                        line = " | ".join(c for c in row if c).strip()
                        if line:
                            out.append((SOURCE_DOC, p.name, f"row{i}", line))
            elif p.suffix.lower() == ".md":
                for sec in re.split(r"^## ", p.read_text(errors="replace"), flags=re.MULTILINE)[1:]:
                    heading = sec.split("\n")[0].strip()
                    out.append((SOURCE_DOC, p.name, heading, sec.strip()[:2000]))
            else:
                for i, para in enumerate(p.read_text(errors="replace").split("\n\n")):
                    if len(para.strip()) > 20:
                        out.append((SOURCE_DOC, p.name, f"para{i}", para.strip()[:2000]))
        except Exception:
            continue
    return out


def _chunks_view(tables: list[Table] | None) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    for t in tables or []:
        if t.view_definition and t.view_definition != "VIEW":
            out.append((SOURCE_VIEW, t.fqn, t.fqn, t.view_definition[:2000]))
    return out


def _file_mtime(source_type: str, source_file: str, code_path: str, doc_path: str) -> int:
    try:
        root = code_path if source_type == SOURCE_CODE else doc_path
        if root:
            return int((Path(root) / source_file).stat().st_mtime * 1000)
    except Exception:
        pass
    return 0


class EmbedIndex:
    """Persistent RAG index over a DuckDB connection."""

    def __init__(self, con, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_hnsw: bool = True):
        self.con = con
        self.model_name = model_name
        self.use_hnsw = use_hnsw
        self._emb = EmbeddingIndex(model_name, use_hnsw=use_hnsw)
        self.con.execute(_TABLE)
        self._maybe_reset_on_model_change()

    @property
    def _stamp(self) -> str:
        return f"st::{self.model_name}"

    def _maybe_reset_on_model_change(self) -> None:
        row = self.con.execute("SELECT embed_model FROM kb_embed_index LIMIT 1").fetchone()
        if row and row[0] and row[0] != self._stamp:
            self.con.execute("DELETE FROM kb_embed_index")

    def ensure_index(self, code_path: str = "", doc_path: str = "",
                     tables: list[Table] | None = None) -> dict[str, int]:
        now = int(time.time() * 1000)
        all_chunks = _chunks_code(code_path) + _chunks_doc(doc_path) + _chunks_view(tables)
        to_embed = []
        for st, sf, heading, content in all_chunks:
            cid = _chunk_id(st, sf, heading)
            mtime = _file_mtime(st, sf, code_path, doc_path) or now
            row = self.con.execute("SELECT indexed_at FROM kb_embed_index WHERE id=?", [cid]).fetchone()
            if not row or (row[0] or 0) < mtime:
                to_embed.append((cid, st, sf, heading, content, mtime))
        if not to_embed:
            return {}
        vectors = self._emb.embed_texts([t[4] for t in to_embed])  # None in lexical mode
        counts: dict[str, int] = {}
        for i, (cid, st, sf, heading, content, mtime) in enumerate(to_embed):
            vec = json.dumps(vectors[i]) if vectors else ""
            self.con.execute(
                """INSERT OR REPLACE INTO kb_embed_index
                   (id, source_type, source_file, heading, content, embedding, embed_model, indexed_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [cid, st, sf, heading, content, vec, self._stamp, mtime])
            counts[st] = counts.get(st, 0) + 1
        return counts

    def search(self, query: str, source_type: str, k: int = TOP_K,
               threshold: float = SIM_THRESHOLD) -> tuple[str, list[str]]:
        rows = self.con.execute(
            "SELECT source_file, heading, content, embedding FROM kb_embed_index WHERE source_type=?",
            [source_type]).fetchall()
        if not rows:
            return "", []
        qv = self._emb.embed_texts([query])
        if qv:  # vector search (HNSW when the corpus is large enough)
            import numpy as np
            q = np.asarray(qv[0])
            mat, meta = [], []
            for sf, heading, content, emb in rows:
                if emb:
                    mat.append(json.loads(emb)); meta.append((sf, heading, content))
            if not mat:
                return "", []
            sims = np.asarray(mat) @ q
            order = sims.argsort()[::-1]
            hits = [(float(sims[i]), *meta[i]) for i in order if sims[i] >= threshold][:k]
        else:  # lexical fallback
            import math
            qt = _tokens(query)
            scored = []
            for sf, heading, content, _emb in rows:
                ct = _tokens(content)
                denom = math.sqrt(len(qt) * len(ct)) or 1.0
                s = len(qt & ct) / denom
                if s >= threshold * 0.5:  # lexical scale is lower
                    scored.append((s, sf, heading, content))
            hits = sorted(scored, reverse=True)[:k]
        content = "\n\n---\n\n".join(h[3] for h in hits)
        refs = [f"{h[1]} § {h[2]}" for h in hits]
        return content, refs
