"""End-to-end orchestration: source -> context -> generate -> CSV."""
from __future__ import annotations

from .config import Config
from .context import CodeContext, DocContext
from .context.embedding import EmbeddingIndex
from .generation import Generator
from .llm import get_backend
from .model import GenerationResult, Table
from .output import write_csv
from .sources import get_source


def run_pipeline(source: str, cfg: Config | None = None, keyword: str = "",
                 limit: int | None = None, **source_kwargs) -> list[GenerationResult]:
    """Pull metadata, attach code/doc context, generate descriptions, write CSV.

    When ``cfg.persist`` is set, the run is backed by the DuckDB knowledge store: pulled
    metadata + lineage are saved, context is retrieved from the persistent (HNSW) RAG
    index, and generated descriptions are written back — so subsequent runs are
    incremental. Otherwise an in-memory retrieval path is used.
    """
    cfg = cfg or Config()

    tables: list[Table] = get_source(source, **source_kwargs).fetch_tables(keyword, limit)

    store = None
    if cfg.persist:
        from .embed_index import SOURCE_CODE, SOURCE_DOC, EmbedIndex
        from .store import KnowledgeStore
        store = KnowledgeStore(cfg.db_path)
        store.save_tables(tables)  # persist pulled metadata + lineage
        idx = EmbedIndex(store.con, cfg.embed_model, use_hnsw=cfg.use_hnsw)
        idx.ensure_index(code_path=cfg.code_path, doc_path=cfg.doc_path, tables=tables)
        for t in tables:  # retrieval-augmented context from the persistent index
            q = f"{t.schema}.{t.name}: " + ", ".join(c.name for c in t.columns[:30])
            if cfg.code_path:
                t.code_context = idx.search(q, SOURCE_CODE)[0] or t.code_context
            if cfg.doc_path:
                t.doc_context = idx.search(q, SOURCE_DOC)[0] or t.doc_context
    else:
        if cfg.code_path:
            CodeContext(cfg.code_path, cfg.embed_model).attach(tables)
        if cfg.doc_path:
            DocContext(cfg.doc_path, cfg.embed_model).attach(tables)

    index = EmbeddingIndex(cfg.embed_model, use_hnsw=cfg.use_hnsw)
    llm = get_backend(cfg.llm_provider, cfg.llm_model, temperature=cfg.generation.temperature_start)
    results = Generator(llm, index, cfg.generation).run(tables)

    write_csv(results, cfg.output_csv)
    if store is not None:
        store.save_tables(tables)  # persist generated descriptions + provenance
        store.close()
    return results


def generate_for_tables(tables: list[Table], cfg: Config | None = None) -> list[GenerationResult]:
    """Generate for an already-materialized table list (used by the benchmark)."""
    cfg = cfg or Config()
    index = EmbeddingIndex(cfg.embed_model)
    llm = get_backend(cfg.llm_provider, cfg.llm_model, temperature=cfg.generation.temperature_start)
    return Generator(llm, index, cfg.generation).run(tables)
