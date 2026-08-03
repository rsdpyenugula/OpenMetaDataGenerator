"""Synthetic metadata benchmark generator.

We generate a controllable, fully-labelled catalog so that description quality can be
measured against ground truth — something real catalogs cannot provide (their existing
descriptions are sparse and inconsistent, which is the very problem we solve).

Design. Each table is instantiated from a *concept* (a domain entity with a known
grain and a set of typed attributes, e.g. ``orders`` = "one row per customer order").
Tables are arranged in a medallion DAG:

    raw.<entity>            (source events; grain = one row per event)
      -> intermediate.<entity>_enriched   (joined/cleaned; grain preserved)
        -> mart.<entity>_daily            (aggregated; grain = one row per entity per day)

For every table we emit:
  * a schema (typed columns, some carrying fine-grained lineage to upstreams),
  * a *gold* table description and gold column descriptions (from the concept),
  * optional **code context** (a SQL stub implementing the transformation) and
    **doc context** (a data-dictionary line), toggled per table so we can ablate
    context availability.

The generator is seeded for reproducibility. ``build_benchmark`` returns the list of
:class:`~openmetadatagenerator.model.Table` objects plus a gold-standard mapping used
by :mod:`benchmark.evaluate`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from openmetadatagenerator.model import Column, Table

# A small library of domain concepts. Each concept defines a grain and typed
# attributes with human descriptions — the source of ground truth.
CONCEPTS = {
    "orders": {
        "grain": "one row per customer order",
        "cols": [
            ("order_id", "string", "Unique identifier for the order."),
            ("customer_id", "string", "Identifier of the customer who placed the order."),
            ("order_ts", "timestamp", "Timestamp when the order was placed."),
            ("amount", "decimal", "Total monetary amount of the order."),
            ("status", "string", "Current fulfilment status of the order."),
        ],
    },
    "payments": {
        "grain": "one row per payment transaction",
        "cols": [
            ("payment_id", "string", "Unique identifier for the payment transaction."),
            ("order_id", "string", "Order this payment settles."),
            ("method", "string", "Payment method used (e.g. card, wallet)."),
            ("amount", "decimal", "Amount captured by the payment."),
            ("paid_ts", "timestamp", "Timestamp when the payment was captured."),
        ],
    },
    "sessions": {
        "grain": "one row per user session",
        "cols": [
            ("session_id", "string", "Unique identifier for the session."),
            ("user_id", "string", "User the session belongs to."),
            ("start_ts", "timestamp", "Timestamp when the session started."),
            ("duration_s", "int", "Session duration in seconds."),
            ("device", "string", "Device class used for the session."),
        ],
    },
    "shipments": {
        "grain": "one row per shipment",
        "cols": [
            ("shipment_id", "string", "Unique identifier for the shipment."),
            ("order_id", "string", "Order being shipped."),
            ("carrier", "string", "Carrier handling the shipment."),
            ("shipped_ts", "timestamp", "Timestamp when the shipment left the warehouse."),
            ("delivered_ts", "timestamp", "Timestamp when the shipment was delivered."),
        ],
    },
    "reviews": {
        "grain": "one row per product review",
        "cols": [
            ("review_id", "string", "Unique identifier for the review."),
            ("product_id", "string", "Product being reviewed."),
            ("user_id", "string", "User who wrote the review."),
            ("rating", "int", "Star rating from 1 to 5."),
            ("review_ts", "timestamp", "Timestamp when the review was submitted."),
        ],
    },
}


@dataclass
class Gold:
    """Ground-truth descriptions for one benchmark instance."""
    table_desc: dict[str, str] = field(default_factory=dict)          # fqn -> desc
    column_desc: dict[str, str] = field(default_factory=dict)         # fqn.col -> desc


def _sql_stub(entity: str, layer: str, upstream_fqn: str, cols: list[str]) -> str:
    sel = ", ".join(cols)
    if layer == "raw":
        return f"-- ingestion of {entity} events\nCREATE TABLE raw.{entity} AS\nSELECT {sel} FROM source.{entity}_stream;"
    if layer == "intermediate":
        return (f"-- clean & enrich {entity}\nCREATE VIEW intermediate.{entity}_enriched AS\n"
                f"SELECT {sel}\nFROM {upstream_fqn}\nWHERE order_ts IS NOT NULL;")
    return (f"-- daily aggregate of {entity}\nCREATE TABLE mart.{entity}_daily AS\n"
            f"SELECT DATE(order_ts) AS day, COUNT(*) AS n, SUM(amount) AS total\n"
            f"FROM {upstream_fqn}\nGROUP BY DATE(order_ts);")


def build_benchmark(n_entities: int = 5, seed: int = 7,
                    context_prob: float = 0.6) -> tuple[list[Table], Gold]:
    """Build a benchmark catalog.

    ``context_prob`` is the probability that a given table is granted code/doc
    context — the knob used to ablate context availability. Returns ``(tables, gold)``.
    """
    rng = random.Random(seed)
    entities = list(CONCEPTS)[:n_entities]
    tables: list[Table] = []
    gold = Gold()

    for entity in entities:
        concept = CONCEPTS[entity]
        raw_cols = [Column(name=n, data_type=dt) for n, dt, _ in concept["cols"]]
        raw = Table("demo", "raw", entity, columns=raw_cols,
                    view_definition="")
        # intermediate (view) preserves grain, adds lineage
        inter_cols = [Column(name=n, data_type=dt, upstreams=[(raw.fqn, n)])
                      for n, dt, _ in concept["cols"]]
        inter = Table("demo", "intermediate", f"{entity}_enriched", columns=inter_cols,
                      upstreams=[raw.fqn], view_definition="VIEW")
        # mart aggregates -> new grain
        mart_cols = [Column("day", "date", upstreams=[(inter.fqn, "order_ts")]),
                     Column("n", "int"), Column("total", "decimal", upstreams=[(inter.fqn, "amount")])]
        mart = Table("demo", "mart", f"{entity}_daily", columns=mart_cols,
                     upstreams=[inter.fqn])

        # --- gold labels ---
        gold.table_desc[raw.fqn] = f"Raw event table capturing {entity}. Grain: {concept['grain']}."
        gold.table_desc[inter.fqn] = (f"Cleaned and enriched {entity}, one row preserved from "
                                      f"raw.{entity}. Grain: {concept['grain']}.")
        gold.table_desc[mart.fqn] = (f"Daily aggregate of {entity}: counts and totals. "
                                     f"Grain: one row per day.")
        for (n, _, d) in concept["cols"]:
            gold.column_desc[f"{raw.fqn}.{n}"] = d
            gold.column_desc[f"{inter.fqn}.{n}"] = d
        gold.column_desc[f"{mart.fqn}.day"] = "Calendar day of aggregation."
        gold.column_desc[f"{mart.fqn}.n"] = f"Number of {entity} on the day."
        gold.column_desc[f"{mart.fqn}.total"] = "Sum of amount over the day."

        # --- context (ablatable) ---
        colnames = [n for n, _, _ in concept["cols"]]
        for t, layer, up in ((raw, "raw", ""), (inter, "intermediate", raw.fqn),
                             (mart, "mart", inter.fqn)):
            if rng.random() < context_prob:
                t.code_context = _sql_stub(entity, layer, up, colnames)
            if rng.random() < context_prob:
                t.doc_context = f"[data_dictionary] {t.schema}.{t.name}: {concept['grain']}."
        tables += [raw, inter, mart]

    rng.shuffle(tables)
    return tables, gold
