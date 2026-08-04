"""TPC-H public-schema benchmark.

Unlike the synthetic benchmark, this uses the widely-known TPC-H schema (8 tables,
public specification) as a *realistic* catalog. Foreign-key references are treated as
lineage edges (a referenced dimension is an upstream of the referencing table), giving
a real dependency DAG: ``region -> nation -> {supplier, customer} -> {orders, partsupp}
-> lineitem``. Gold table/column descriptions are paraphrased from the public TPC-H
specification so description quality can be scored against ground truth.

This provides an external-validity complement to the controllable synthetic benchmark:
the schema, names, and lineage are not authored to flatter the method.
"""
from __future__ import annotations

from openmetadatagenerator.model import Column, Table

from .generate import Gold

# (schema.table): (columns[(name,type)], fk_upstream_tables, gold_table_desc, {col: gold})
_TPCH = {
    "region": (
        [("r_regionkey", "int"), ("r_name", "string"), ("r_comment", "string")], [],
        "Reference table of geographic regions. One row per region.",
        {"r_regionkey": "Unique identifier of the region.",
         "r_name": "Name of the region.",
         "r_comment": "Free-text comment about the region."}),
    "nation": (
        [("n_nationkey", "int"), ("n_name", "string"), ("n_regionkey", "int"),
         ("n_comment", "string")], ["region"],
        "Reference table of nations, each belonging to a region. One row per nation.",
        {"n_nationkey": "Unique identifier of the nation.",
         "n_name": "Name of the nation.",
         "n_regionkey": "Region the nation belongs to.",
         "n_comment": "Free-text comment about the nation."}),
    "supplier": (
        [("s_suppkey", "int"), ("s_name", "string"), ("s_address", "string"),
         ("s_nationkey", "int"), ("s_phone", "string"), ("s_acctbal", "decimal"),
         ("s_comment", "string")], ["nation"],
        "Master table of suppliers. One row per supplier.",
        {"s_suppkey": "Unique identifier of the supplier.",
         "s_name": "Name of the supplier.",
         "s_nationkey": "Nation in which the supplier is located.",
         "s_acctbal": "Account balance of the supplier."}),
    "customer": (
        [("c_custkey", "int"), ("c_name", "string"), ("c_address", "string"),
         ("c_nationkey", "int"), ("c_phone", "string"), ("c_acctbal", "decimal"),
         ("c_mktsegment", "string"), ("c_comment", "string")], ["nation"],
        "Master table of customers. One row per customer.",
        {"c_custkey": "Unique identifier of the customer.",
         "c_name": "Name of the customer.",
         "c_nationkey": "Nation in which the customer is located.",
         "c_acctbal": "Account balance of the customer.",
         "c_mktsegment": "Market segment the customer belongs to."}),
    "part": (
        [("p_partkey", "int"), ("p_name", "string"), ("p_mfgr", "string"),
         ("p_brand", "string"), ("p_type", "string"), ("p_size", "int"),
         ("p_container", "string"), ("p_retailprice", "decimal"),
         ("p_comment", "string")], [],
        "Master table of parts offered for sale. One row per part.",
        {"p_partkey": "Unique identifier of the part.",
         "p_name": "Name of the part.",
         "p_mfgr": "Manufacturer of the part.",
         "p_retailprice": "Retail price of the part."}),
    "partsupp": (
        [("ps_partkey", "int"), ("ps_suppkey", "int"), ("ps_availqty", "int"),
         ("ps_supplycost", "decimal"), ("ps_comment", "string")], ["part", "supplier"],
        "Association of parts to the suppliers that provide them. One row per part-supplier pair.",
        {"ps_partkey": "Part being supplied.",
         "ps_suppkey": "Supplier providing the part.",
         "ps_availqty": "Available quantity of the part from this supplier.",
         "ps_supplycost": "Cost at which the supplier provides the part."}),
    "orders": (
        [("o_orderkey", "int"), ("o_custkey", "int"), ("o_orderstatus", "string"),
         ("o_totalprice", "decimal"), ("o_orderdate", "date"),
         ("o_orderpriority", "string"), ("o_clerk", "string"),
         ("o_shippriority", "int"), ("o_comment", "string")], ["customer"],
        "Fact table of customer orders. One row per order.",
        {"o_orderkey": "Unique identifier of the order.",
         "o_custkey": "Customer who placed the order.",
         "o_orderstatus": "Current status of the order.",
         "o_totalprice": "Total price of the order.",
         "o_orderdate": "Date the order was placed."}),
    "lineitem": (
        [("l_orderkey", "int"), ("l_partkey", "int"), ("l_suppkey", "int"),
         ("l_linenumber", "int"), ("l_quantity", "decimal"),
         ("l_extendedprice", "decimal"), ("l_discount", "decimal"),
         ("l_tax", "decimal"), ("l_returnflag", "string"),
         ("l_linestatus", "string"), ("l_shipdate", "date"),
         ("l_shipmode", "string"), ("l_comment", "string")],
        ["orders", "partsupp"],
        "Fact table of individual line items within orders. One row per order line item.",
        {"l_orderkey": "Order this line item belongs to.",
         "l_partkey": "Part sold on this line item.",
         "l_suppkey": "Supplier of the part on this line item.",
         "l_quantity": "Quantity of the part ordered on this line item.",
         "l_extendedprice": "Extended price for this line item.",
         "l_discount": "Discount applied to this line item.",
         "l_shipdate": "Date this line item was shipped."}),
}


def build_tpch(with_doc_context: bool = False) -> tuple[list[Table], Gold]:
    """Return TPC-H tables (with FK-derived lineage) and gold labels.

    ``with_doc_context`` optionally attaches a one-line data-dictionary note per table
    (a light form of documentation grounding) to study context sensitivity on a real
    schema.
    """
    tables: list[Table] = []
    gold = Gold()
    for name, (cols, fks, tdesc, cdesc) in _TPCH.items():
        fqn = f"tpch.public.{name}"
        columns = [Column(nm, dt) for nm, dt in cols]
        # attach column lineage on the FK columns pointing at referenced tables
        for c in columns:
            for up in fks:
                if c.name.endswith("key") and up[0] == c.name.split("_")[1][0]:
                    c.upstreams = [(f"tpch.public.{up}", c.name)]
        t = Table("tpch", "public", name, columns=columns,
                  upstreams=[f"tpch.public.{u}" for u in fks])
        if with_doc_context:
            # Partial hint only (the grain), not the full gold description, to avoid
            # trivially leaking the answer into the context.
            import re as _re
            grain = _re.search(r"One row per [^.]+\.", tdesc)
            t.doc_context = f"[tpch_spec] {name}: {grain.group(0) if grain else 'reference entity.'}"
        tables.append(t)
        gold.table_desc[fqn] = tdesc
        for cn, cd in cdesc.items():
            gold.column_desc[f"{fqn}.{cn}"] = cd
    return tables, gold
