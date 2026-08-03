"""CSV export.

The canonical output is a single tidy CSV with one row per described object (table or
column). The schema is designed to feed directly into an external LLM-judge / RAGAS
faithfulness pipeline: each row pairs the *exact context the model saw* with its
*output*, plus a provenance note explaining how the description was grounded.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .model import GenerationResult

HEADER = ["object_type", "object_name", "parent_table",
          "context", "output", "grounding_notes", "similarity"]


def write_csv(results: list[GenerationResult], path: str) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(HEADER)
        for r in results:
            w.writerow([r.object_type, r.object_name, r.parent_table,
                        r.context, r.output, r.grounding_notes,
                        "" if r.similarity is None else f"{r.similarity:.4f}"])
    return len(results)
