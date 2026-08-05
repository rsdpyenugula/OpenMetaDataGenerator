# OpenMetaDataGenerator

[![ci](https://github.com/rsdpyenugula/OpenMetaDataGenerator/actions/workflows/ci.yml/badge.svg)](https://github.com/rsdpyenugula/OpenMetaDataGenerator/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Context-grounded, lineage-aware automatic description generation for data-catalog metadata.**

Modern data platforms hold tens of thousands of tables whose technical metadata
(schemas, lineage) is captured automatically but whose *semantic* metadata (what a
table means, what one row represents, what each column is) is chronically missing.
OpenMetaDataGenerator generates that semantic layer at scale by grounding a large
language model in three signals — **schema**, **external context** (transformation
code and documentation), and **data lineage** — and emitting auditable descriptions
to a catalog or to CSV.

The system is source- and model-agnostic:

| Layer | Pluggable implementations |
|-------|---------------------------|
| **Metadata source** | DataHub Core (GraphQL), Databricks (Unity Catalog `information_schema`), Snowflake (`INFORMATION_SCHEMA`) |
| **Local store** | DuckDB (`kb_tables` / `kb_columns` / `kb_embed_index`) — persists metadata, lineage, descriptions, and embeddings for incremental runs |
| **RAG retrieval** | local embeddings (sentence-transformers) + **HNSW** index (`hnswlib`), persisted in DuckDB; lexical fallback with no model |
| **Context** | path to code repository, path to documentation corpus |
| **LLM backend** | OpenAI, Anthropic, AWS Bedrock, Google Vertex, Azure OpenAI, Gemini |
| **Output** | tidy CSV (one row per described object) |

The data flow mirrors a production catalog-documentation loop:

```
source (pull) → DuckDB (store + HNSW RAG index) → LLM (agentic gen) → CSV/catalog (write-back)
```

## Method in one paragraph

**Canonicalize-first:** the same logical column recurs across thousands of tables, so we
canonicalize column names, describe each frequent concept once, and seed it everywhere
(cheaper, and consistent). **Lineage-aware waves:** remaining tables are generated in
topological order so a downstream table's prompt inherits the *already-generated*
descriptions of its upstreams — grounding propagates through the lineage DAG. **Planned,
controlled agentic loop:** a strategy controller then measures coverage and per-item
accuracy against explicit targets each iteration and selects the next action —
*inherit* (free copy from same-named/upstream columns), *sibling* (infer a table's empty
columns from its described ones), *rework* (regenerate the low-similarity tail), or
*stop* — with an **LLM-judge** gating whether a reworked description replaces the old one.
The loop is bounded and auditable (the decision trace is logged). See
[`generation.py`](openmetadatagenerator/generation.py) and
[`canonicalize.py`](openmetadatagenerator/canonicalize.py).

## Install

```bash
pip install -e ".[all]"          # or pick extras: .[datahub,anthropic,embed]
```

## Use

```bash
# Generate descriptions for a Databricks catalog with code + doc grounding
export OMDG_LLM_PROVIDER=anthropic OMDG_LLM_MODEL=claude-3-5-sonnet-latest
omdg databricks --keyword my_catalog \
    --code-path ./transformations --doc-path ./docs --out descriptions.csv
```

Or from Python:

```python
from openmetadatagenerator import Config
from openmetadatagenerator.pipeline import run_pipeline

cfg = Config(); cfg.llm_provider = "openai"; cfg.code_path = "./sql"
results = run_pipeline("snowflake", cfg, keyword="analytics")
```

## Run the demo (no API keys)

See the whole agentic pipeline work on a **real open-source schema (Sakila**, 15 tables)
in one command:

```bash
make demo          # or: python examples/agentic_demo.py
```

It exercises every mechanism — canonicalize-first, lineage waves (incl. a store↔staff
cycle), the controlled agentic loop, and `[AIG | High]`/`[AIG | Low]` confidence tags —
then writes a CSV. Abridged output:

```
[1] canonicalize-first: 3 recurring concept(s) described once:
      'last_update' -> 15 occurrences across 15 tables

[2] lineage-aware waves (from foreign keys, upstream-first):
      wave 1: actor, category, country, language
      ...
      wave 4: customer, inventory, payment, rental, staff, store

[3] controlled agentic decision trace (measured vs. targets each iteration):
      strategy   coverage  accuracy  cand(inh/sib/rew)
      inherit        62%      ...     20/11/65     <- fills FK columns from upstream PKs
      sibling        88%      ...     0/3/86       <- fills obscure cols from siblings
      rework        100%      ...     0/0/96
      stop          100%      ...     0/0/96

[4] sample descriptions with confidence tags + grounding provenance:
      • sakila.public.country
          [AIG | Low]  ...            [NO CONTEXT: schema + column names only]
      • sakila.public.film
          [AIG | High] ...            [INHERITED-ONLY: 1 upstream(s)]

[5] wrote 96 rows -> outputs/agentic_demo.csv   (column coverage 81/81 = 100%)
```

Coverage climbs 62% → 88% → 100% as the controller picks `inherit` then `sibling`.
Swap the demo backend for `get_backend("anthropic")` (etc.) to run it with a real LLM.

A second demo shows the **DuckDB + HNSW local RAG** retrieving a snippet into a table's prompt
(needs the `store` extra for DuckDB):

```bash
pip install -e ".[store]"
make rag-demo      # or: python examples/rag_demo.py
```
It indexes a sample code + docs corpus into `kb_embed_index`, retrieves the right chunk
for a query table, and generates a description grounded in it (tagged `[AIG | High]`).

## Reproduce the benchmark

A synthetic, fully-labelled benchmark measures description quality against ground
truth and ablates the two contributions (context, lineage):

```bash
python -m benchmark.run                                  # synthetic, deterministic (no API keys)
python -m benchmark.run --benchmark tpch                 # real public schema (TPC-H)
python -m benchmark.run --benchmark sakila               # real public schema (Sakila)
python -m benchmark.run --benchmark cryptic              # Sakila with obfuscated names
python -m benchmark.run --benchmark cryptic --provider gemini --model gemini-3.1-pro-preview
```

**Headline result** (Gemini 3.1 Pro, mean over 3 runs): on descriptive-name schemas a strong
LLM is already saturated and context adds a few points at most — but on the **cryptic-name**
variant of the same schema (identifiers obfuscated, gold and lineage unchanged), grounding is
**decisive: accuracy 0.27 → 0.71 (+43 points, sd ≤ 0.006)**. The value of grounding is
governed by name informativeness — it concentrates exactly where enterprise catalogs hurt.

Four benchmarks are included: a **synthetic** medallion catalog with controllable context
and lineage; the **TPC-H** and **Sakila** public schemas (foreign keys read as lineage) —
external-validity checks on schemas we did not author; and the **cryptic** variant of Sakila
(obfuscated identifiers, unchanged gold + lineage) targeting the regime where names carry no
meaning. Outputs `benchmark/results*.json` and `paper/results_table*.tex`; the three raw
runs behind the headline mean±sd are committed as `benchmark/results_cryptic_run*.json`.
Ablation *mechanics* reproduce without API keys; use `--provider` for a real cloud LLM.

## Paper

The accompanying paper is in [`paper/`](paper/): `make paper` builds the arXiv PDF
(`main.tex`, tectonic) and `make paper-tmlr` builds the anonymized double-blind TMLR
submission (`main_tmlr.tex`, official `tmlr.sty`).

## License

Apache-2.0. See [LICENSE](LICENSE).
