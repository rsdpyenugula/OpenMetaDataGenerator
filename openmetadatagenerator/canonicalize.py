"""Canonicalization of column concepts across a catalog.

Enterprise schemas are massively redundant: the same logical column
(``dw_insert_ts``, ``customer_id``, ``amount``, ...) recurs across hundreds or
thousands of tables. Describing each occurrence independently is wasteful (repeated
LLM calls) and inconsistent (the same concept gets subtly different wording in
different tables). Canonicalization exploits this redundancy:

* **Canonical key.** Column names are normalized to a canonical key so naming variants
  (``Order_ID``, ``order_id``, ``orderid``) collapse to one concept.
* **Canonicalize-first.** For high-frequency canonical concepts we generate a single
  description once and seed it onto every occurrence, so table-level generation only
  has to describe the long tail of table-specific columns.
* **Sibling propagation.** After generation, any still-undescribed occurrence of a
  canonical concept inherits the best-grounded description among its siblings.

This is a cost \emph{and} consistency mechanism: fewer model calls, identical wording
for identical concepts. Columns that carry table-specific lineage are exempt from
sharing so genuinely different semantics are preserved.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .model import Column, Table

# Common data-warehouse audit prefixes/suffixes that don't change a column's concept.
_NOISE = re.compile(r"[^a-z0-9]+")


def canonical_key(name: str) -> str:
    """Normalize a column name to a canonical concept key.

    Lowercases and strips non-alphanumerics so ``Order_ID`` / ``order_id`` / ``orderid``
    map together. Intentionally conservative: it does not merge semantically distinct
    columns that happen to be related.
    """
    return _NOISE.sub("", name.lower())


def group_columns(tables: list[Table]) -> dict[str, list[tuple[Table, Column]]]:
    """Group every (table, column) by canonical key."""
    groups: dict[str, list[tuple[Table, Column]]] = defaultdict(list)
    for t in tables:
        for c in t.columns:
            groups[canonical_key(c.name)].append((t, c))
    return groups


def high_frequency_concepts(tables: list[Table], min_freq: int = 3
                            ) -> dict[str, list[tuple[Table, Column]]]:
    """Canonical concepts that occur in at least ``min_freq`` distinct tables and carry
    no fine-grained lineage (so a shared description is appropriate)."""
    out: dict[str, list[tuple[Table, Column]]] = {}
    for key, members in group_columns(tables).items():
        tbls = {t.fqn for t, _ in members}
        no_lineage = all(not c.upstreams for _, c in members)
        if len(tbls) >= min_freq and no_lineage:
            out[key] = members
    return out
