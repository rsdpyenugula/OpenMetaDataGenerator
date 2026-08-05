# arXiv submission metadata

**Title:** OpenMetaDataGenerator: Context-Grounded, Lineage-Aware Automatic Description Generation for Data-Catalog Metadata

**Authors:** Ravi Satya Durga Prasad Yenugula

**Primary category:** cs.DB (Databases)
**Cross-list:** cs.LG (Machine Learning), cs.AI (Artificial Intelligence), cs.CL (Computation and Language)

**Comments:** 9 pages, 2 tables, 1 algorithm. Code, synthetic + TPC-H benchmarks, and evaluation harness: https://github.com/rsdpyenugula/OpenMetaDataGenerator (Apache-2.0).

**License:** arXiv non-exclusive license to distribute (compatible with later TMLR publication).

---

## Abstract (arXiv)

Enterprise data platforms routinely track tens of thousands of tables whose technical
metadata—schemas and lineage—is captured automatically, yet whose semantic metadata—what
a table means, what one row represents, what each column measures—is chronically absent,
slowing discovery, governance, and analytics. We present OpenMetaDataGenerator (OMDG), an
open-source system that generates this missing semantic layer at scale by grounding a
large language model in table schema, external context (transformation code and
documentation), and data lineage. OMDG contributes: a topological wave scheduler that
generates descriptions in lineage order so downstream tables inherit their upstreams'
grounded descriptions; canonicalize-first generation that describes each recurring column
concept once and seeds it catalog-wide; a planned, controlled agentic loop that measures
coverage and per-item accuracy against explicit targets and selects inherit / sibling /
rework actions with an LLM-judge gating replacements; and an auditable substrate (DuckDB
knowledge store, persistent incremental HNSW retrieval, confidence tags). The system is
source-agnostic (DataHub, Databricks, Snowflake) and model-agnostic (six providers). We
evaluate on a labelled synthetic benchmark, two public schemas (TPC-H, Sakila), and a
cryptic-name variant of Sakila with obfuscated identifiers. With a strong production model
the experiments expose a two-regime picture: when names are self-describing the model is
already saturated and context adds at most a few points, but when names carry no meaning
grounding is decisive—accuracy rises from 0.27 to 0.71 (+43 points, stable across three
runs). The value of grounding is governed by name informativeness, concentrating exactly
where real enterprise catalogs hurt most. We release the system, benchmarks, and
evaluation harness.

---

## Submission checklist

- [x] GitHub handle set to `rsdpyenugula` in `main.tex`, `README.md`, `CITATION.cff`.
- [x] PDF builds locally: `make paper` (tectonic).
- [x] Real-LLM numbers in all tables (Gemini 3.1 Pro; cryptic headline = mean±sd over 3 runs).
- [x] `main.tex` (arXiv) and `main_tmlr.tex` (anonymized, tmlr.sty) both compile.
- [ ] Tag a release (e.g. v0.1.0) so the arXiv "Code" link is reproducible.
