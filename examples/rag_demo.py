"""Demo of the DuckDB-backed, HNSW-accelerated local RAG — no API keys.

Run:
    python examples/rag_demo.py

It creates a tiny sample corpus (a SQL transformation file + a Markdown data
dictionary), builds the persistent DuckDB embedding index (kb_embed_index), and shows
retrieval pulling the *right* snippet into a table's generation prompt — so the
retrieval-augmented grounding path is visible end-to-end:

  1. index a code + docs corpus into DuckDB (incremental; re-runs re-embed only changes),
  2. retrieve the top code/doc chunk for a query table,
  3. attach the retrieved context and generate — the description is grounded in it
     and tagged [AIG | High].

With ``pip install -e ".[embed]"`` retrieval uses sentence-transformer vectors + an HNSW
index; without it, it falls back to lexical scoring (still returns the right chunk here).
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from openmetadatagenerator.config import GenerationConfig
from openmetadatagenerator.context.embedding import EmbeddingIndex
from openmetadatagenerator.embed_index import SOURCE_CODE, SOURCE_DOC, EmbedIndex
from openmetadatagenerator.generation import Generator
from openmetadatagenerator.model import Column, Table
from benchmark.mock_llm import MockBackend


def _write_corpus(root: Path) -> tuple[str, str]:
    code, docs = root / "code", root / "docs"
    code.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    (code / "orders_daily.sql").write_text(
        "-- Daily order rollup. One row per calendar day.\n"
        "CREATE TABLE mart.orders_daily AS\n"
        "SELECT DATE(order_ts) AS day, COUNT(*) AS n, SUM(amount) AS revenue\n"
        "FROM raw.orders GROUP BY DATE(order_ts);")
    (code / "customers.sql").write_text(
        "CREATE TABLE dim.customers AS SELECT customer_id, email, signup_ts FROM raw.customer_events;")
    (docs / "dictionary.md").write_text(
        "## orders_daily\n"
        "Daily aggregate of orders: one row per calendar day, with the number of orders "
        "and total revenue for that day.\n\n"
        "## customers\n"
        "Customer dimension: one row per customer with contact and signup information.\n")
    return str(code), str(docs)


def main() -> None:
    root = Path("outputs/rag_demo")
    code_path, doc_path = _write_corpus(root)

    print("=" * 68)
    print("OpenMetaDataGenerator — DuckDB + HNSW local RAG demo")
    print("=" * 68)

    con = duckdb.connect(str(root / "kb.duckdb"))
    idx = EmbedIndex(con, use_hnsw=True)
    counts = idx.ensure_index(code_path=code_path, doc_path=doc_path)
    total = con.execute("SELECT COUNT(*) FROM kb_embed_index").fetchone()[0]
    print(f"\n[1] indexed corpus into DuckDB kb_embed_index: {counts}  (rows now: {total})")
    print("    re-running would re-embed only changed chunks (incremental).")

    # A table we want documented; retrieve grounding for it.
    t = Table("shop", "mart", "orders_daily", columns=[
        Column("day", "date"), Column("n", "int"), Column("revenue", "decimal")])
    query = f"{t.schema}.{t.name}: " + ", ".join(c.name for c in t.columns)

    code_hit, code_refs = idx.search(query, SOURCE_CODE, threshold=0.0)
    doc_hit, doc_refs = idx.search(query, SOURCE_DOC, threshold=0.0)
    print(f"\n[2] retrieved for '{t.fqn}':")
    print(f"    code  <- {code_refs[:1]}\n          {code_hit.splitlines()[1] if code_hit else '(none)'}")
    print(f"    docs  <- {doc_refs[:1]}\n          {doc_hit.strip()[:80] if doc_hit else '(none)'}")

    # Attach retrieved context and generate.
    t.code_context, t.doc_context = code_hit, doc_hit
    results = Generator(MockBackend(), EmbeddingIndex(), GenerationConfig(workers=1)).run([t])
    con.close()

    print("\n[3] generated (grounded in the retrieved context):")
    print(f"      {t.generated_description}")
    print(f"      [{t.grounding_notes}]")
    print(f"\n    -> {len(results)} rows; the description is tagged High because retrieval")
    print("       supplied real code + doc grounding for this table.")
    print("=" * 68)


if __name__ == "__main__":
    main()
