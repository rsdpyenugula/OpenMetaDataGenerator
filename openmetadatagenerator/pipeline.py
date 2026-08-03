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
    """Pull metadata, attach code/doc context, generate descriptions, write CSV."""
    cfg = cfg or Config()

    tables: list[Table] = get_source(source, **source_kwargs).fetch_tables(keyword, limit)

    if cfg.code_path:
        CodeContext(cfg.code_path, cfg.embed_model).attach(tables)
    if cfg.doc_path:
        DocContext(cfg.doc_path, cfg.embed_model).attach(tables)

    index = EmbeddingIndex(cfg.embed_model)
    llm = get_backend(cfg.llm_provider, cfg.llm_model, temperature=cfg.generation.temperature_start)
    results = Generator(llm, index, cfg.generation).run(tables)

    write_csv(results, cfg.output_csv)
    return results


def generate_for_tables(tables: list[Table], cfg: Config | None = None) -> list[GenerationResult]:
    """Generate for an already-materialized table list (used by the benchmark)."""
    cfg = cfg or Config()
    index = EmbeddingIndex(cfg.embed_model)
    llm = get_backend(cfg.llm_provider, cfg.llm_model, temperature=cfg.generation.temperature_start)
    return Generator(llm, index, cfg.generation).run(tables)
