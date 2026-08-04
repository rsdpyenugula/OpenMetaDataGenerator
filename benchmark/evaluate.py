"""Benchmark metrics.

Given generated results and the gold labels, we report:

* **coverage** — fraction of objects (tables + columns) that received a non-empty
  description.
* **accuracy** — mean cosine similarity between the generated description and the
  *gold* description (semantic correctness against ground truth).
* **faithfulness** — mean cosine between the generated description and the *context*
  it was grounded in (a proxy for hallucination: low faithfulness = ungrounded).
* **exact-grain** — fraction of table descriptions that recover the correct grain
  phrase ("one row per ...").

All similarities use the shared embedding index (with a lexical fallback), so results
are comparable across ablations run in the same environment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from openmetadatagenerator.context.embedding import EmbeddingIndex, cosine
from openmetadatagenerator.model import GenerationResult

from .generate import Gold

_TAG = re.compile(r"^\s*\[(?:AIG\s*\|\s*(?:High|Low)|Reviewed)\]\s*", re.IGNORECASE)


def _untag(s: str) -> str:
    """Strip a leading confidence tag so scoring compares semantics, not the tag."""
    return _TAG.sub("", s or "").strip()


@dataclass
class Metrics:
    coverage: float
    accuracy: float
    faithfulness: float
    exact_grain: float
    n: int

    def row(self) -> str:
        return (f"coverage={self.coverage:.3f}  accuracy={self.accuracy:.3f}  "
                f"faithfulness={self.faithfulness:.3f}  exact_grain={self.exact_grain:.3f}  n={self.n}")


def evaluate(results: list[GenerationResult], gold: Gold, n_expected: int,
             index: EmbeddingIndex | None = None) -> Metrics:
    index = index or EmbeddingIndex()
    gold_map = {**gold.table_desc, **gold.column_desc}

    acc, faith, grain_hits, grain_total = [], [], 0, 0
    for r in results:
        out = _untag(r.output)
        g = gold_map.get(r.object_name)
        if g:
            acc.append(cosine(index, out, g))
        if r.context:
            faith.append(cosine(index, out, r.context))
        if r.object_type == "table":
            grain_total += 1
            gg = gold.table_desc.get(r.object_name, "")
            want = _grain(gg)
            if want and want in out.lower():
                grain_hits += 1

    coverage = len(results) / n_expected if n_expected else 0.0
    return Metrics(
        coverage=round(coverage, 4),
        accuracy=round(sum(acc) / len(acc), 4) if acc else 0.0,
        faithfulness=round(sum(faith) / len(faith), 4) if faith else 0.0,
        exact_grain=round(grain_hits / grain_total, 4) if grain_total else 0.0,
        n=len(results),
    )


def _grain(desc: str) -> str:
    import re
    m = re.search(r"one row per [^\.\n]+", desc.lower())
    return m.group(0) if m else ""
