# TMLR submission statement

*(TMLR evaluates on technical soundness and clarity of contribution rather than novelty or
expected impact. This statement summarizes the submission against TMLR's criteria; adapt
into the OpenReview submission form. TMLR does not use a traditional cover letter—this is a
summary for the Action Editor.)*

## Summary of contributions
We introduce **OpenMetaDataGenerator (OMDG)**, a system and method for automatically
generating semantic descriptions of data-catalog objects (tables and columns). The central
idea is to treat **data lineage as a scheduling structure**: descriptions are generated in
topological order so a downstream table's prompt inherits the freshly generated
descriptions of its upstreams, propagating grounding through the lineage DAG. Around this we
provide retrieval-grounded, provenance-tagged prompting and two closed control loops
(coverage and an accuracy-driven rework loop). The system is source-agnostic (DataHub,
Databricks, Snowflake) and model-agnostic (five LLM providers).

## Why TMLR (claims are supported, and the scope is clearly delimited)
1. **The claims are precise and tested.** We claim that (a) external context improves
   description accuracy and grain recovery over a schema-only baseline, and (b) lineage
   provides an ordering that lets grounding propagate. Claim (a) is supported by ablations
   on two benchmarks (synthetic and TPC-H). For claim (b) we are explicit that the
   deterministic reference backend is lineage-neutral by construction and that lineage's
   effect on accuracy is measured with real LLMs via the released harness—we do not
   overclaim from the reproducible reference numbers.
2. **Reproducibility.** All ablation trends run without API keys or GPUs via a
   deterministic backend; the same harness reproduces with any of five cloud providers.
   Code, both benchmarks, and the evaluation harness are released under Apache-2.0.
3. **Honest evaluation design.** Because production catalogs lack ground truth, we build a
   labelled synthetic benchmark and a real public schema (TPC-H), keep documentation
   context to a partial hint to avoid target leakage, and additionally specify a
   human-evaluation protocol (correctness / completeness / groundedness with $\kappa$).

## Limitations we state plainly
The synthetic benchmark isolates mechanisms but is not a substitute for human evaluation on
production catalogs; the similarity-based rework signal can under-credit terse but correct
descriptions; and generated text should be surfaced with provenance and confidence for
human-in-the-loop governance. These are discussed in Section (Discussion and Limitations).

## Reproducibility statement
`pip install -e ".[all]"` then `make test` (unit tests) and `make benchmark` (reproducible
ablations). Real-LLM numbers: `make benchmark-llm PROVIDER=<p> MODEL=<m>`.

## Suggested reviewer expertise
Data management / metadata systems; retrieval-augmented generation; LLM evaluation.

## Prior dissemination
An identical preprint is (to be) posted on arXiv (cs.DB); this is permitted under TMLR's
policy on preprints.

## Anonymized build (double-blind)
TMLR review is double-blind. Submit the PDF built from `paper/main_tmlr.tex`
(`make paper-tmlr`): it uses the official `tmlr.sty`, replaces the author block with
"Anonymous authors", and swaps the GitHub URL for an anonymized-supplementary pointer via
the `\REPO` macro. For the camera-ready, switch to `\usepackage[accepted]{tmlr}` and restore
the real author block and repository URL.
