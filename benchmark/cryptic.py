"""Cryptic-name benchmark: Sakila with obfuscated identifiers.

This benchmark targets the regime that motivates OMDG: enterprise catalogs whose
column names carry little or no meaning (opaque codes, warehouse naming conventions,
auto-generated part slots). We take the Sakila schema — keeping its gold descriptions,
foreign-key lineage, and DAG structure — and deterministically obfuscate every table
and column name (``customer`` -> ``t05``, ``customer_id`` -> ``c05_00``), decoupling
name from meaning.

Grounding context is *realistic, non-leaking* evidence: the transformation SQL that
built each warehouse table from a readable legacy source, e.g.::

    CREATE TABLE warehouse.t05 AS
    SELECT customer_id AS c05_00, store_id AS c05_01, ... FROM legacy.customer;

The context contains the *name mapping* (cryptic column <- readable source column),
never the gold description text, so scoring against gold is not trivial retrieval —
the model still has to know what a ``customer_id`` means. Without context, the names
are meaningless and a model can only guess; with context, the mapping makes the
semantics recoverable. Lineage edges (and fine-grained FK -> PK column lineage) are
preserved under the renaming.
"""
from __future__ import annotations

from openmetadatagenerator.model import Column, Table

from .generate import Gold
from .public_sakila import _SAKILA, _SHARED


def _mapping() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Deterministic obfuscation maps: table -> tNN, (table, column) -> cNN_MM."""
    tmap: dict[str, str] = {}
    cmap: dict[str, dict[str, str]] = {}
    for ti, (tname, (cols, _fks, _td, _cd)) in enumerate(_SAKILA.items()):
        tmap[tname] = f"t{ti:02d}"
        cmap[tname] = {cn: f"c{ti:02d}_{ci:02d}" for ci, (cn, _ct) in enumerate(cols)}
    return tmap, cmap


def _rename_sql(tname: str, cols: list[tuple[str, str]], cmap_t: dict[str, str]) -> str:
    selects = ",\n    ".join(f"{cn} AS {cmap_t[cn]}" for cn, _ in cols)
    return (f"-- ETL: build warehouse table from the legacy '{tname}' table\n"
            f"CREATE TABLE warehouse.{{new}} AS\nSELECT\n    {selects}\nFROM legacy.{tname};")


def build_cryptic(with_context: bool = True) -> tuple[list[Table], Gold]:
    """Sakila with obfuscated names. ``with_context`` attaches the rename-SQL code
    context and a one-line catalog note; without it, only the cryptic schema remains."""
    tmap, cmap = _mapping()
    tables: list[Table] = []
    gold = Gold()

    for tname, (cols, fks, tdesc, cdesc) in _SAKILA.items():
        new_t = tmap[tname]
        fqn = f"warehouse.public.{new_t}"
        columns: list[Column] = []
        for cn, ct in cols:
            up = fks.get(cn)
            ups = []
            if up:
                up_pk = f"{up}_id"
                up_cryptic_pk = cmap[up].get(up_pk, "")
                if up_cryptic_pk:
                    ups = [(f"warehouse.public.{tmap[up]}", up_cryptic_pk)]
            columns.append(Column(cmap[tname][cn], ct, upstreams=ups))

        t = Table("warehouse", "public", new_t, columns=columns,
                  upstreams=[f"warehouse.public.{tmap[u]}" for u in dict.fromkeys(fks.values())])
        if with_context:
            t.code_context = _rename_sql(tname, cols, cmap[tname]).replace("{new}", new_t)
            t.doc_context = (f"[catalog_note] warehouse.{new_t}: migrated from the legacy "
                             f"'{tname}' table of the DVD-rental system.")
        tables.append(t)

        # Gold: cryptic keys, original (semantic) descriptions.
        gold.table_desc[fqn] = tdesc
        for cn, cd in cdesc.items():
            gold.column_desc[f"{fqn}.{cmap[tname][cn]}"] = cd
        for cn, cd in _SHARED.items():
            if cn in cmap[tname] and f"{fqn}.{cmap[tname][cn]}" not in gold.column_desc:
                gold.column_desc[f"{fqn}.{cmap[tname][cn]}"] = cd

    return tables, gold
