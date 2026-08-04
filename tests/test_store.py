"""DuckDB knowledge store + DuckDB-backed RAG index (lineage round-trip)."""
from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")  # store/RAG tests require the [store] extra

from openmetadatagenerator.config import GenerationConfig
from openmetadatagenerator.context.embedding import EmbeddingIndex
from openmetadatagenerator.embed_index import SOURCE_CODE, EmbedIndex
from openmetadatagenerator.generation import Generator
from openmetadatagenerator.store import KnowledgeStore
from benchmark.mock_llm import MockBackend
from benchmark.public_sakila import build_sakila


def test_store_roundtrip_preserves_lineage_and_descriptions(tmp_path):
    tables, _ = build_sakila()
    Generator(MockBackend(), EmbeddingIndex(), GenerationConfig(workers=2)).run(tables)

    store = KnowledgeStore(str(tmp_path / "kb.duckdb"))
    store.save_tables(tables)
    loaded = store.load_tables()
    store.close()

    assert len(loaded) == len(tables)
    by_fqn = {t.fqn: t for t in loaded}

    # table-level lineage preserved (payment references customer/staff/rental)
    pay = by_fqn["sakila.public.payment"]
    assert any("customer" in u for u in pay.upstreams)
    assert pay.generated_description  # descriptions persisted

    # fine-grained column lineage preserved
    fk = next(c for c in pay.columns if c.name == "customer_id")
    assert fk.upstreams and fk.upstreams[0][0].endswith("customer")


def test_store_coverage(tmp_path):
    tables, _ = build_sakila()
    Generator(MockBackend(), EmbeddingIndex(), GenerationConfig(workers=2)).run(tables)
    store = KnowledgeStore(str(tmp_path / "kb.duckdb"))
    store.save_tables(tables)
    cov = store.coverage()
    store.close()
    assert cov["tables"] == len(tables)
    assert cov["tables_described"] == len(tables)  # mock describes everything
    assert cov["columns"] > 0


def test_embed_index_retrieves_relevant_code(tmp_path):
    con = duckdb.connect(str(tmp_path / "kb.duckdb"))
    idx = EmbedIndex(con, use_hnsw=True)
    # write a tiny code corpus
    code = tmp_path / "code"
    code.mkdir()
    (code / "orders.sql").write_text(
        "-- build the orders fact\nCREATE TABLE orders AS SELECT order_id, amount FROM src;")
    (code / "unrelated.py").write_text("def helper():\n    return 42\n")
    n = idx.ensure_index(code_path=str(code))
    assert n.get(SOURCE_CODE, 0) >= 2

    content, refs = idx.search("orders fact table with amount", SOURCE_CODE, k=1, threshold=0.0)
    con.close()
    assert content  # returns the most relevant chunk
    assert "orders" in content.lower()
