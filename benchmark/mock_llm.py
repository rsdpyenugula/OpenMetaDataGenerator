"""Deterministic reference backend for reproducible benchmarking.

Real experiments use cloud LLM backends. To let anyone reproduce the *relative*
trends (context and lineage ablations) without API keys or cost, we provide a
deterministic generator that is sensitive to exactly the signals the method supplies:
code/doc context, upstream descriptions, and column lineage present in the prompt.

It is intentionally simple — it extracts the entity, grain, and grounded column
facts from the prompt — so that when context/lineage are ablated away, its output
degrades, mirroring the behaviour of a context-grounded LLM. This makes the ablation
tables runnable in CI while the paper additionally reports real-LLM numbers.
"""
from __future__ import annotations

import json
import re

from openmetadatagenerator.llm.base import LLMBackend


class MockBackend(LLMBackend):
    name = "mock"

    def __init__(self, model: str = "mock", **kw):
        super().__init__(model, **kw)

    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        fqn = _find(r"Table:\s*([^\n]+)", prompt)
        layer = _find(r"Layer:\s*([^\n]+)", prompt) or "unknown"
        entity = fqn.split(".")[-1].replace("_enriched", "").replace("_daily", "") if fqn else "data"
        cols = _columns(prompt)

        grain = ""
        m = re.search(r"one row per [^.\n]+", prompt, re.I)
        if m:
            grain = m.group(0).lower()

        has_code = "Transformation code:" in prompt
        has_doc = "Documentation:" in prompt
        has_up = "Upstream tables" in prompt

        # Table description degrades when no grounding is available.
        if has_code or has_doc or has_up:
            lead = _layer_phrase(layer) if layer in ("raw", "intermediate", "mart") \
                else f"Table describing {entity}:"
            desc = f"{lead} {entity}."
            if grain:
                desc += f" {grain[0].upper()}{grain[1:]}."
            # NOTE: the deterministic mock does not semantically exploit inherited
            # upstream descriptions (only a real LLM does), so it is lineage-neutral by
            # construction. Lineage's effect on accuracy is measured in the real-LLM runs.
        else:
            desc = f"Table {entity} in the {layer} layer."

        col_out = {}
        for cn in cols:
            gd = _find(rf"{re.escape(cn)}[^:\n]*:\s*([^\n]+)", prompt)
            if gd and (has_doc or has_code):
                col_out[cn] = gd.strip()
            elif cn.endswith("_id") or cn == "id":
                col_out[cn] = f"Identifier for {cn[:-3] or entity}."
            elif cn.endswith("_ts") or cn.endswith("_ts"):
                col_out[cn] = f"Timestamp for {cn.replace('_ts','')}."
            else:
                col_out[cn] = f"The {cn.replace('_',' ')} value."
        return json.dumps({"table": desc, "columns": col_out})


def _find(pat: str, s: str) -> str:
    m = re.search(pat, s)
    return m.group(1).strip() if m else ""


def _columns(prompt: str) -> list[str]:
    line = _find(r"Columns:\s*([^\n]+)", prompt)
    return [re.sub(r"\s*\(.*?\)", "", c).strip() for c in line.split(",") if c.strip()] if line else []


def _layer_phrase(layer: str) -> str:
    return {"raw": "Raw event table capturing",
            "intermediate": "Cleaned and enriched",
            "mart": "Daily aggregate of"}.get(layer, "Table for")
