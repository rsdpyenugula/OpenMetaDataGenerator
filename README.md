# OpenMetaDataGenerator

[![ci](https://github.com/<your-account>/OpenMetaDataGenerator/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-account>/OpenMetaDataGenerator/actions/workflows/ci.yml)
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
| **Context** | path to code repository, path to documentation corpus |
| **LLM backend** | OpenAI, Anthropic, AWS Bedrock, Google Vertex, Azure OpenAI |
| **Output** | tidy CSV (one row per described object) |

## Method in one paragraph

Tables are generated in **topological (lineage) order** so a downstream table's
prompt can inherit the *already-generated* descriptions of its upstreams — grounding
propagates through the lineage DAG instead of documenting each table in isolation.
Each prompt fuses schema signals, retrieved code/doc context, and column-level
lineage. Two closed control loops then run: a **coverage** loop retries empty objects,
and a **rework** loop scores each description's semantic similarity to the exact
context it was grounded in and regenerates the low-similarity tail with a rising
temperature until a target accuracy is met. See [`generation.py`](openmetadatagenerator/generation.py).

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

## Reproduce the benchmark

A synthetic, fully-labelled benchmark measures description quality against ground
truth and ablates the two contributions (context, lineage):

```bash
python -m benchmark.run                                  # synthetic, deterministic (no API keys)
python -m benchmark.run --benchmark tpch                 # real public schema (TPC-H)
python -m benchmark.run --benchmark tpch --provider anthropic --model claude-3-5-sonnet-latest
```

Two benchmarks are included: a **synthetic** medallion catalog with controllable context
and lineage, and the **TPC-H** public schema (8 tables) whose foreign keys are read as
lineage — an external-validity check on a schema we did not author. Outputs
`benchmark/results*.json` and `paper/results_table*.tex`. Ablation *trends* reproduce
without API keys; run with `--provider` to reproduce with a real cloud LLM.

## Paper

The accompanying paper (arXiv / TMLR) is in [`paper/`](paper/). Build with
`cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## License

Apache-2.0. See [LICENSE](LICENSE).
