"""Lineage-aware description generation.

The generator is the core contribution of OpenMetaDataGenerator. It combines three
ideas:

1. **Topological wave scheduling.** Tables are generated in dependency order (upstream
   first) so that a downstream table's prompt can include the *already-generated*
   descriptions of its upstreams. This propagates grounding through the lineage DAG
   instead of documenting every table in isolation.

2. **Context-grounded prompting.** Each prompt is assembled from schema signals
   (layer, columns, view SQL), external evidence (code + document context), and
   inherited upstream descriptions + fine-grained column lineage.

3. **Two closed control loops.** A *coverage* loop retries objects that came back
   empty; a *rework* loop scores each description's semantic similarity to the exact
   context it was grounded in and regenerates the low-similarity tail with a rising
   temperature, until mean accuracy clears a target or an iteration cap is hit.

The design is provider-agnostic (any :class:`~openmetadatagenerator.llm.base.LLMBackend`)
and produces auditable provenance for every object (the exact context + a grounding
note), which the CSV export and the benchmark's faithfulness metric consume.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .config import GenerationConfig
from .context.embedding import EmbeddingIndex, cosine
from .llm.base import LLMBackend
from .model import GenerationResult, Table

_SYSTEM = (
    "You are a senior data engineer writing precise, factual data-catalog "
    "descriptions. Ground every statement in the provided context. Do not invent "
    "columns, metrics, or semantics that are not supported. Prefer one or two "
    "sentences. State the grain (what one row represents) when it is inferable."
)


def _topo_waves(tables: list[Table]) -> list[list[Table]]:
    """Group tables into dependency waves (Kahn's algorithm over in-scope lineage)."""
    by_fqn = {t.fqn: t for t in tables}
    # Restrict edges to the in-scope set; external upstreams are ignored for ordering.
    deps: dict[str, set[str]] = {t.fqn: set() for t in tables}
    for t in tables:
        for up in t.upstreams:
            # upstreams may be URNs or fqns; match by suffix on the fqn.
            for fqn in by_fqn:
                if up.endswith(fqn) or fqn in up:
                    if fqn != t.fqn:
                        deps[t.fqn].add(fqn)
    waves: list[list[Table]] = []
    done: set[str] = set()
    remaining = set(deps)
    while remaining:
        ready = [f for f in remaining if deps[f] <= done]
        if not ready:  # cycle: break it by taking everything left
            ready = list(remaining)
        waves.append([by_fqn[f] for f in sorted(ready)])
        done |= set(ready)
        remaining -= set(ready)
    return waves


def _prompt(t: Table, upstream_desc: dict[str, str], max_chars: int) -> str:
    lines = [f"Table: {t.fqn}", f"Layer: {t.layer}"]
    if t.columns:
        cols = ", ".join(f"{c.name} ({c.data_type})" if c.data_type else c.name
                         for c in t.columns[:60])
        lines.append(f"Columns: {cols}")
    if t.view_definition and t.view_definition != "VIEW":
        lines.append(f"View SQL:\n{t.view_definition[:1500]}")
    if t.code_context:
        lines.append(f"Transformation code:\n{t.code_context}")
    if t.doc_context:
        lines.append(f"Documentation:\n{t.doc_context}")
    # Inherited upstream grounding.
    ups = [f"- {fqn}: {upstream_desc[fqn]}" for fqn in t.upstreams
           if upstream_desc.get(fqn)]
    if not ups:
        ups = [f"- {fqn}: {upstream_desc[fqn]}" for fqn in upstream_desc
               if any(fqn in u or u.endswith(fqn) for u in t.upstreams)]
    if ups:
        lines.append("Upstream tables (already described):\n" + "\n".join(ups[:8]))
    # Fine-grained column lineage hints.
    fg = [f"{c.name} <- {', '.join(f'{u}.{col}' for u, col in c.upstreams[:3])}"
          for c in t.columns if c.upstreams]
    if fg:
        lines.append("Column lineage:\n" + "\n".join(fg[:20]))
    lines.append(
        "\nReturn strict JSON: {\"table\": \"<description>\", "
        "\"columns\": {\"<col>\": \"<description>\", ...}}. "
        "Describe only columns you can ground.")
    return "\n".join(lines)[:max_chars]


def _parse(raw: str) -> tuple[str, dict[str, str]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip() if "```" in raw[3:] else raw[3:]
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return (obj.get("table") or "").strip(), {
            k: (v or "").strip() for k, v in (obj.get("columns") or {}).items()}
    except Exception:
        return raw.split("\n")[0].strip(), {}


class Generator:
    def __init__(self, llm: LLMBackend, index: EmbeddingIndex | None = None,
                 cfg: GenerationConfig | None = None):
        self.llm = llm
        self.index = index or EmbeddingIndex()
        self.cfg = cfg or GenerationConfig()

    def run(self, tables: list[Table]) -> list[GenerationResult]:
        upstream_desc: dict[str, str] = {}

        # --- wave generation -------------------------------------------------
        for wave in _topo_waves(tables):
            prompts = [_prompt(t, upstream_desc, self.cfg.model_ctx_chars) for t in wave]
            outs = self.llm.batch(prompts, system=_SYSTEM,
                                  temperature=self.cfg.temperature_start,
                                  workers=self.cfg.workers)
            for t, p, raw in zip(wave, prompts, outs):
                tdesc, cdescs = _parse(raw)
                t.generated_description = tdesc
                t.prompt_context = p
                for c in t.columns:
                    c.generated_description = cdescs.get(c.name, cdescs.get(c.name.lower(), ""))
                    c.prompt_context = p
                if tdesc:
                    upstream_desc[t.fqn] = tdesc

        # --- coverage loop ---------------------------------------------------
        for _ in range(self.cfg.max_coverage_iters):
            missing = [t for t in tables if not t.generated_description]
            if not missing:
                break
            prompts = [_prompt(t, upstream_desc, self.cfg.model_ctx_chars) for t in missing]
            outs = self.llm.batch(prompts, system=_SYSTEM,
                                  temperature=self.cfg.temperature_start + 0.2,
                                  workers=self.cfg.workers)
            for t, raw in zip(missing, outs):
                tdesc, cdescs = _parse(raw)
                t.generated_description = tdesc
                for c in t.columns:
                    if not c.generated_description:
                        c.generated_description = cdescs.get(c.name, "")
                if tdesc:
                    upstream_desc[t.fqn] = tdesc

        # --- accuracy rework loop -------------------------------------------
        temp = self.cfg.temperature_start
        for _ in range(self.cfg.max_rework_iters):
            scored = [(t, cosine(self.index, t.generated_description, t.prompt_context))
                      for t in tables if t.generated_description]
            if not scored:
                break
            mean_acc = sum(s for _, s in scored) / len(scored)
            if mean_acc >= self.cfg.target_accuracy:
                break
            temp += self.cfg.temperature_step
            worst = [t for t, s in scored if s < self.cfg.rework_sim_threshold]
            if not worst:
                break
            prompts = [_prompt(t, upstream_desc, self.cfg.model_ctx_chars)
                       + "\n\nReuse concrete vocabulary from the context above."
                       for t in worst]
            outs = self.llm.batch(prompts, system=_SYSTEM, temperature=temp,
                                  workers=self.cfg.workers)
            for t, raw in zip(worst, outs):
                tdesc, cdescs = _parse(raw)
                if tdesc:
                    t.generated_description = tdesc
                    t.rework_iters += 1
                    upstream_desc[t.fqn] = tdesc

        return self._results(tables)

    def _results(self, tables: list[Table]) -> list[GenerationResult]:
        out: list[GenerationResult] = []
        for t in tables:
            note = self._grounding_note(t)
            t.grounding_notes = note
            if t.generated_description:
                out.append(GenerationResult(
                    "table", t.fqn, "", t.prompt_context, t.generated_description, note,
                    cosine(self.index, t.generated_description, t.prompt_context)))
            for c in t.columns:
                if c.generated_description:
                    cn = ("Column lineage: " +
                          ", ".join(f"{u}.{col}" for u, col in c.upstreams[:3])) \
                        if c.upstreams else "From parent table context"
                    out.append(GenerationResult(
                        "column", f"{t.fqn}.{c.name}", t.fqn, c.prompt_context,
                        c.generated_description, cn))
        return out

    @staticmethod
    def _grounding_note(t: Table) -> str:
        srcs = [n for n, v in (("code", t.code_context), ("docs", t.doc_context),
                               ("view", t.view_definition)) if v]
        has_up = bool(t.upstreams)
        if srcs and has_up:
            tier = "RICH" if len(srcs) >= 2 else "PARTIAL"
            base = f"{tier}: {' + '.join(srcs)} + {len(t.upstreams)} upstream(s)"
        elif srcs:
            base = f"{'RICH' if len(srcs) >= 2 else 'PARTIAL'}: {' + '.join(srcs)}"
        elif has_up:
            base = f"INHERITED-ONLY: {len(t.upstreams)} upstream(s)"
        else:
            base = "NO CONTEXT: schema + column names only"
        if t.rework_iters:
            base += f"  (reworked {t.rework_iters}x)"
        return base
