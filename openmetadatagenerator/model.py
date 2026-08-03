"""Core data model shared across sources, context providers, generation, and output.

The model is deliberately catalog-agnostic: a :class:`Table` is identified by a
fully-qualified name (``catalog.schema.table``) and carries the minimal signals the
generator needs — columns, an optional existing description, an optional view/SQL
definition, and coarse (table-level) plus fine-grained (column-level) upstream
lineage. Every metadata source (DataHub, Databricks, Snowflake, ...) normalizes its
native representation into these dataclasses so the rest of the pipeline is uniform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Column:
    name: str
    data_type: str = ""
    description: str = ""
    # Fine-grained upstream lineage: list of (upstream_table_fqn, upstream_column).
    upstreams: list[tuple[str, str]] = field(default_factory=list)
    # Populated by generation; kept separate from the ingested ``description``.
    generated_description: str = ""
    # The exact context string shown to the model (for auditing / CSV export).
    prompt_context: str = ""


@dataclass
class Table:
    catalog: str
    schema: str
    name: str
    columns: list[Column] = field(default_factory=list)
    description: str = ""                # existing/ingested description, if any
    view_definition: str = ""           # SQL for views, else ""
    # Coarse (table-level) upstream lineage as fully-qualified names.
    upstreams: list[str] = field(default_factory=list)
    # Free-form context attached by :mod:`context` providers (code, docs, ...).
    code_context: str = ""
    doc_context: str = ""
    # Populated by generation.
    generated_description: str = ""
    prompt_context: str = ""
    grounding_notes: str = ""
    rework_iters: int = 0

    @property
    def fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.name}"

    @property
    def is_view(self) -> bool:
        return bool(self.view_definition) or self.schema.lower().endswith("view") \
            or self.name.lower().startswith(("vw_", "v_"))

    @property
    def layer(self) -> str:
        """Coarse medallion-style layer inferred from the schema name.

        Used both as a light generation prior and for layer-conditioned quality
        rules. Falls back to ``"unknown"`` when no convention matches.
        """
        s = self.schema.lower()
        for tag in ("raw", "bronze", "staging", "stg", "intermediate", "silver",
                    "mart", "gold", "managed", "reference", "dim", "fact"):
            if tag in s:
                return tag
        return "unknown"


@dataclass
class GenerationResult:
    """One generated description plus the provenance needed for evaluation."""
    object_type: str                    # "table" | "column"
    object_name: str                    # fqn (table) or fqn.column
    parent_table: str                   # "" for tables; parent fqn for columns
    context: str                        # exact prompt context the model saw
    output: str                         # generated description
    grounding_notes: str                # short provenance/lineage trail
    similarity: Optional[float] = None  # cosine(output, context), if scored
