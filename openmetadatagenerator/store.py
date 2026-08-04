"""Local knowledge store (DuckDB) — mirrors the production architecture.

Three relations, matching the reference system:

* ``kb_tables``  — one row per table, keyed by fully-qualified name. Carries existing +
  generated descriptions, the attached context (code / doc), the exact ``prompt_context``
  the model saw, table-level ``upstreams`` (lineage) as JSON, a per-item
  ``similarity_score`` (cosine of description vs context), ``rework_iters``,
  ``grounding_notes``, and pull/generate timestamps.
* ``kb_columns`` — one row per column, with existing + generated description, fine-grained
  ``upstreams`` as JSON (column lineage), ``prompt_context``, ``similarity_score`` and
  ``rework_iters``.
* ``kb_embed_index`` — the persistent RAG index (see :mod:`embed_index`): chunk content +
  its embedding vector + an ``embed_model`` stamp, so retrieval is incremental and
  survives across runs.

Lineage is first-class and rehydrated into :class:`~openmetadatagenerator.model.Table`
objects on load, so the wave scheduler and inherit strategy behave identically whether
tables come from a source or from the store. DuckDB is an optional dependency
(``pip install openmetadatagenerator[store]``), imported lazily.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .model import Column, Table

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_tables (
    fqn                   VARCHAR PRIMARY KEY,
    catalog               VARCHAR,
    schema_name           VARCHAR,
    name                  VARCHAR,
    existing_desc         TEXT,
    view_definition       TEXT,
    code_context          TEXT,
    doc_context           TEXT,
    upstreams             TEXT,       -- JSON list of upstream fqns/urns
    generated_desc        TEXT,
    prompt_context        TEXT,
    grounding_notes       TEXT,
    similarity_score      REAL,
    rework_iters          INTEGER DEFAULT 0,
    pulled_at             BIGINT,
    generated_at          BIGINT
);
CREATE TABLE IF NOT EXISTS kb_columns (
    id                    VARCHAR PRIMARY KEY,   -- fqn#name
    table_fqn             VARCHAR,
    name                  VARCHAR,
    data_type             VARCHAR,
    existing_desc         TEXT,
    upstreams             TEXT,       -- JSON list of [upstream_fqn, upstream_col]
    generated_desc        TEXT,
    prompt_context        TEXT,
    similarity_score      REAL,
    rework_iters          INTEGER DEFAULT 0
);
"""


class KnowledgeStore:
    def __init__(self, path: str = ".cache/knowledge.duckdb"):
        import duckdb
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(path)
        for stmt in _SCHEMA.strip().split(";"):
            if stmt.strip():
                self.con.execute(stmt)

    # ------------------------------------------------------------------ write
    def save_tables(self, tables: list[Table]) -> int:
        now = int(time.time() * 1000)
        for t in tables:
            gen_at = now if t.generated_description else None
            self.con.execute(
                "INSERT OR REPLACE INTO kb_tables VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [t.fqn, t.catalog, t.schema, t.name, t.description, t.view_definition,
                 t.code_context, t.doc_context, json.dumps(t.upstreams),
                 t.generated_description, t.prompt_context, t.grounding_notes,
                 None, t.rework_iters, now, gen_at])
            for c in t.columns:
                self.con.execute(
                    "INSERT OR REPLACE INTO kb_columns VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [f"{t.fqn}#{c.name}", t.fqn, c.name, c.data_type, c.description,
                     json.dumps([list(u) for u in c.upstreams]),
                     c.generated_description, c.prompt_context, None, 0])
        return len(tables)

    # ------------------------------------------------------------------- read
    def load_tables(self, keyword: str = "") -> list[Table]:
        rows = self.con.execute(
            """SELECT fqn, catalog, schema_name, name, existing_desc, view_definition,
                      code_context, doc_context, upstreams, generated_desc,
                      prompt_context, grounding_notes, rework_iters
               FROM kb_tables
               WHERE ? = '' OR LOWER(fqn) LIKE ?
               ORDER BY fqn""",
            [keyword, f"%{keyword.lower()}%"]).fetchall()
        tables: list[Table] = []
        for (fqn, cat, sch, name, desc, vdef, code, doc, ups, gen,
             pctx, notes, rw) in rows:
            t = Table(catalog=cat, schema=sch, name=name, description=desc or "",
                      view_definition=vdef or "", code_context=code or "",
                      doc_context=doc or "", upstreams=json.loads(ups or "[]"),
                      generated_description=gen or "", prompt_context=pctx or "",
                      grounding_notes=notes or "", rework_iters=rw or 0)
            for (cn, dt, cdesc, cups, cgen, cctx) in self.con.execute(
                    """SELECT name, data_type, existing_desc, upstreams,
                              generated_desc, prompt_context
                       FROM kb_columns WHERE table_fqn = ? ORDER BY name""", [fqn]).fetchall():
                t.columns.append(Column(
                    name=cn, data_type=dt or "", description=cdesc or "",
                    upstreams=[tuple(u) for u in json.loads(cups or "[]")],
                    generated_description=cgen or "", prompt_context=cctx or ""))
            tables.append(t)
        return tables

    def coverage(self, keyword: str = "") -> dict:
        t = self.con.execute(
            """SELECT COUNT(*), SUM(CASE WHEN generated_desc <> '' THEN 1 ELSE 0 END)
               FROM kb_tables WHERE ? = '' OR LOWER(fqn) LIKE ?""",
            [keyword, f"%{keyword.lower()}%"]).fetchone()
        c = self.con.execute(
            """SELECT COUNT(*), SUM(CASE WHEN generated_desc <> '' THEN 1 ELSE 0 END)
               FROM kb_columns WHERE ? = '' OR LOWER(table_fqn) LIKE ?""",
            [keyword, f"%{keyword.lower()}%"]).fetchone()
        return {"tables": t[0], "tables_described": t[1] or 0,
                "columns": c[0], "columns_described": c[1] or 0}

    def close(self) -> None:
        self.con.close()
