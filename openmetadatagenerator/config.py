"""Runtime configuration.

All settings are plain environment variables (optionally loaded from a local
``.env``). Nothing here is provider- or deployment-specific: the same config drives
any metadata source and any LLM backend. Secrets (API keys, tokens) are read from
the environment and never written to disk by this library.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_dotenv(Path.cwd() / ".env")


@dataclass
class GenerationConfig:
    """Hyper-parameters for the generation loop (see ``generation.py``)."""
    target_coverage: float = 0.90       # fraction of objects that must get a description
    target_accuracy: float = 0.75       # mean cosine(output, context) stop criterion
    rework_sim_threshold: float = 0.60  # per-item cosine below which we regenerate
    max_coverage_iters: int = 4         # table-level coverage retries
    max_rework_iters: int = 3           # rework passes per rework strategy invocation
    max_agent_iters: int = 6            # iterations of the controlled strategy-selection loop
    judge: bool = True                  # LLM-judge gates whether a rework replaces the old desc
    temperature_start: float = 0.0
    temperature_step: float = 0.1       # ramp per rework pass to escape local minima
    workers: int = 8
    model_ctx_chars: int = 30_000       # soft cap on prompt context length
    # Canonicalization (see canonicalize.py).
    canonicalize: bool = True           # canonicalize-first + sibling propagation
    canonical_min_freq: int = 3         # concept must recur in >= this many tables
    inherit_lineage: bool = True        # columns inherit upstream column descriptions


@dataclass
class Config:
    llm_provider: str = field(default_factory=lambda: os.environ.get("OMDG_LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.environ.get("OMDG_LLM_MODEL", ""))
    embed_model: str = field(default_factory=lambda: os.environ.get(
        "OMDG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    code_path: str = field(default_factory=lambda: os.environ.get("OMDG_CODE_PATH", ""))
    doc_path: str = field(default_factory=lambda: os.environ.get("OMDG_DOC_PATH", ""))
    output_csv: str = field(default_factory=lambda: os.environ.get("OMDG_OUTPUT_CSV", "descriptions.csv"))
    generation: GenerationConfig = field(default_factory=GenerationConfig)
