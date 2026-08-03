# TMLR reviewer-response template

*Reusable structure for the revision cycle. TMLR is iterative: respond point-by-point, make
the change in the paper, and reference where. Be concrete and non-defensive.*

---

**We thank the reviewers for their careful reading. We have revised the paper and respond
to each point below; changes are marked in blue in the updated PDF and summarized here.**

## Reviewer <#>

> **R<#>.1 —** *(paste the reviewer's point verbatim)*

**Response.** *(Agree/clarify. State the concrete change and where it lives.)* We have
`[added/revised]` `[Section X / Table Y / Alg. Z]` to `[…]`. `[If experiment requested:]`
We ran `[setup]` and report `[result]` in `[Table]`.

> **R<#>.2 —** …

**Response.** …

---

## Common points and prepared answers

**"The synthetic benchmark may not reflect real schemas."**
> We agree and for this reason (a) added the public TPC-H schema (foreign keys as lineage),
> which we did not author, and (b) specify a human-evaluation protocol (Section on human
> eval). The synthetic benchmark is presented as a *controlled* complement, not a
> replacement, and this framing is stated explicitly.

**"The similarity metric can be gamed / rewards lexical overlap."**
> Correct; we report it as a faithfulness *proxy* and pair it with grain-recovery (a
> semantic check) and the human protocol. The rework loop's use of this signal is bounded
> (threshold + iteration cap) and its limitation is discussed in Section (Limitations).

**"How much does lineage scheduling actually help vs. context?"**
> We disentangle the two in the ablation (no-lineage vs. no-context vs. schema-only). We are
> explicit that the deterministic reference backend is lineage-neutral by construction and
> that the lineage effect on accuracy is measured with real LLMs via the released harness;
> we report those numbers in `[Table]`.

**"Reproducibility of the LLM results."**
> All trends reproduce without API keys via the deterministic backend; real-LLM numbers are
> reproducible with `make benchmark-llm PROVIDER=<p> MODEL=<m>` and pinned model ids listed
> in `[Appendix]`.

**"Novelty relative to RAG / LLM-for-data-management."**
> Our contribution is not RAG per se but treating the *lineage DAG as a scheduling
> structure* for grounded generation, plus the closed-loop coverage/rework control and an
> auditable provenance export. We clarified this in Related Work and the Introduction.

---

## Changelog for this revision
- `[Section/Table]` — `[what changed]`
- …
