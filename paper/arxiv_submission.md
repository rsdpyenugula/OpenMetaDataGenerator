# arXiv submission metadata

**Title:** OpenMetaDataGenerator: Context-Grounded, Lineage-Aware Automatic Description Generation for Data-Catalog Metadata

**Authors:** Ravi Satya Durga Prasad Yenugula

**Primary category:** cs.DB (Databases)
**Cross-list:** cs.LG (Machine Learning), cs.AI (Artificial Intelligence), cs.CL (Computation and Language)

**Comments:** 9 pages, 2 tables, 1 algorithm. Code, synthetic + TPC-H benchmarks, and evaluation harness: https://github.com/<your-account>/OpenMetaDataGenerator (Apache-2.0).

**License:** arXiv non-exclusive license to distribute (compatible with later TMLR publication).

---

## Abstract (arXiv)

Enterprise data platforms routinely track tens of thousands of tables whose technical
metadata—schemas and lineage—is captured automatically, yet whose semantic metadata—what
a table means, what one row represents, what each column measures—is chronically absent,
slowing discovery, governance, and analytics. We present OpenMetaDataGenerator (OMDG), an
open-source system that generates this missing semantic layer at scale by grounding a
large language model in three complementary signals: table schema, external context
(transformation code and documentation), and data lineage. OMDG contributes (i) a
topological *wave scheduler* that generates descriptions in lineage order so downstream
tables inherit the grounded descriptions of their upstreams, propagating context through
the lineage DAG; (ii) retrieval-grounded, provenance-tagged prompting that fuses schema,
code, documentation, and column-level lineage; and (iii) two closed control loops—a
coverage loop and an accuracy-driven rework loop that scores each description's similarity
to the exact context it was grounded in and regenerates a low-similarity tail. The system
is source-agnostic (DataHub, Databricks, Snowflake) and model-agnostic (OpenAI, Anthropic,
Bedrock, Vertex, Azure). Because production catalogs lack ground-truth semantics, we
introduce a synthetic, fully-labelled benchmark and complement it with the public TPC-H
schema (foreign keys read as lineage); across both, external context substantially
improves accuracy and grain recovery over a schema-only baseline. We release the system,
benchmarks, and evaluation harness.

---

## Submission checklist

- [ ] Replace `<your-account>` with the real GitHub org/user in `main.tex`, `README.md`, `CITATION.cff`.
- [ ] Build camera PDF: Overleaf or `make paper` (needs a TeX distribution).
- [ ] Run real-LLM numbers and paste into `paper/results_table*.tex` (kept auto-generated):
      `make benchmark-llm PROVIDER=<p> MODEL=<m>` and `... --benchmark tpch`.
- [ ] Verify `main.tex` compiles standalone (article class) OR swap in the TMLR class for TMLR.
- [ ] Tag a release (e.g. v0.1.0) so the arXiv "Code" link is reproducible.
