"""End-to-end demo of the OpenMetaDataGenerator agentic pipeline on a REAL,
open-source schema (Sakila) — no API keys required.

Run:
    python examples/agentic_demo.py

Sakila is the classic open-source sample database (a DVD-rental store, 15 tables here).
It exercises every mechanism: ``last_update`` recurs in every table (canonicalize-first),
foreign keys form a deep lineage DAG with a store<->staff cycle (waves + inherit), sparse
root tables leave gaps (sibling), and grounded vs. ungrounded objects get High vs. Low
confidence tags. We run the full pipeline with a deterministic, context-following demo
backend and print:

  1. canonicalize-first — recurring column concepts described once, seeded everywhere,
  2. lineage-aware waves — the topological generation order derived from foreign keys,
  3. the planned, controlled agentic decision trace (inherit / sibling / rework / stop),
  4. sample descriptions with their [AIG | High] / [AIG | Low] confidence tags + provenance,
  5. a written CSV of every generated table/column description.

To reproduce with a real cloud LLM instead of the demo backend, swap ``DemoBackend()``
for ``get_backend("anthropic")`` (etc.) and set that provider's API key.
"""
from __future__ import annotations

import json
import re

from benchmark.mock_llm import MockBackend
from benchmark.public_sakila import build_sakila
from openmetadatagenerator.canonicalize import high_frequency_concepts
from openmetadatagenerator.config import GenerationConfig
from openmetadatagenerator.context.embedding import EmbeddingIndex
from openmetadatagenerator.generation import Generator, _topo_waves
from openmetadatagenerator.output import write_csv

# Obscure columns a name-only model won't attempt -> left for the sibling strategy.
_OBSCURE = {"description", "rating", "district", "postal_code", "phone", "username",
            "replacement_cost", "rental_duration", "release_year", "length"}


class DemoBackend(MockBackend):
    """Context-following demo model. During bulk (wave) generation it declines two
    kinds of columns, creating real coverage gaps for the controller to fill:
      * foreign-key columns (``*_id`` that are not the table's own PK) — recovered by
        the **inherit** strategy from their upstream primary key, and
      * a set of obscure columns — recovered by the **sibling** strategy from a table's
        already-described columns.
    It answers the controller's sibling-fill and canonical-concept calls normally."""

    def generate(self, prompt: str, system: str = "", temperature=None) -> str:
        m = re.search(r"Columns to describe:\s*([^\n]+)", prompt)
        if m:  # sibling-fill call
            cols = [c.strip() for c in m.group(1).split(",") if c.strip()]
            return json.dumps({"columns": {c: f"Inferred from sibling columns: the {c} value."
                                           for c in cols}})
        if prompt.startswith("Concept:"):  # canonical-concept call
            nm = re.search(r"named '([^']+)'", prompt)
            return f"Canonical description of {nm.group(1) if nm else 'value'} across all tables."
        base = super().generate(prompt, system, temperature)
        try:
            obj = json.loads(base)
        except Exception:
            return base
        tm = re.search(r"Table:\s*\S+\.(\w+)", prompt)
        pk = f"{tm.group(1)}_id" if tm else ""
        kept = {}
        for c, d in obj.get("columns", {}).items():
            cl = c.lower()
            is_fk = cl.endswith("_id") and cl != pk
            if is_fk or cl in _OBSCURE:
                continue  # decline -> becomes a coverage gap (inherit / sibling fills it)
            kept[c] = d
        obj["columns"] = kept
        return json.dumps(obj)


def main() -> None:
    tables, _gold = build_sakila(with_doc_context=False)  # sparse: force the loop to work
    n_tables = len(tables)
    n_cols = sum(len(t.columns) for t in tables)

    print("=" * 70)
    print(f"OpenMetaDataGenerator — agentic demo on Sakila  ({n_tables} tables, {n_cols} columns)")
    print("=" * 70)

    concepts = high_frequency_concepts(tables, min_freq=3)
    print(f"\n[1] canonicalize-first: {len(concepts)} recurring concept(s) described once:")
    for _key, members in concepts.items():
        names = sorted({c.name for _, c in members})
        print(f"      '{names[0]}' -> {len(members)} occurrences across "
              f"{len({t.fqn for t, _ in members})} tables")
    print("\n[2] lineage-aware waves (from foreign keys, upstream-first):")
    for i, wave in enumerate(_topo_waves(tables), 1):
        print(f"      wave {i}: {', '.join(t.name for t in wave)}")

    cfg = GenerationConfig(workers=2, target_accuracy=0.9)
    gen = Generator(DemoBackend(), EmbeddingIndex(), cfg)
    results = gen.run(tables)

    print("\n[3] controlled agentic decision trace (measured vs. targets each iteration):")
    print(f"      {'strategy':9s} {'coverage':>9} {'accuracy':>9}  cand(inh/sib/rew)")
    for strat, g in gen.trace:
        print(f"      {strat:9s} {g['col_coverage']:>8.0%} {g['accuracy']:>9.2f}  "
              f"{g['inherit_candidates']}/{g['sibling_candidates']}/{g['rework_candidates']}")

    print("\n[4] sample descriptions with confidence tags + grounding provenance:")
    for t in [x for x in tables if x.name in ("payment", "film", "country")]:
        print(f"      • {t.fqn}\n          {t.generated_description}\n          [{t.grounding_notes}]")
        # show a couple of columns to surface High/Low tags at column level too
        for c in t.columns[:2]:
            if c.generated_description:
                print(f"          - {c.name}: {c.generated_description}")

    n = write_csv(results, "outputs/agentic_demo.csv")
    covered = sum(1 for t in tables for c in t.columns if c.generated_description)
    print(f"\n[5] wrote {n} rows -> outputs/agentic_demo.csv   "
          f"(column coverage {covered}/{n_cols} = {covered / n_cols:.0%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
