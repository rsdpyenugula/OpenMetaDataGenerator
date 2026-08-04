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

from .canonicalize import canonical_key, group_columns, high_frequency_concepts
from .config import GenerationConfig
from .context.embedding import EmbeddingIndex, cosine
from .llm.base import LLMBackend
from .model import GenerationResult, Table

# Confidence tags prefixed to every description (BOS = beginning-of-string tags):
#   [AIG | High] — AI-generated with rich grounding (code / docs / lineage)
#   [AIG | Low]  — AI-generated with little or no grounding (schema/names only)
#   [Reviewed]   — human-reviewed; preserved as-is if already present
_KNOWN_TAGS = ("[AIG | High]", "[AIG | Low]", "[Reviewed]")


def _tag(text: str, high: bool) -> str:
    """Prefix a confidence tag, unless the description already carries one."""
    t = (text or "").strip()
    if not t or t.startswith(_KNOWN_TAGS):
        return t
    return f"{'[AIG | High]' if high else '[AIG | Low]'}  {t}"


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

        # --- canonicalize-first: describe frequent concepts once, seed everywhere -
        if self.cfg.canonicalize:
            self._canonical_prepass(tables)

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
                    c.prompt_context = c.prompt_context or p
                    # Do not overwrite a description seeded by the canonical pre-pass.
                    if not c.generated_description:
                        c.generated_description = cdescs.get(c.name, cdescs.get(c.name.lower(), ""))
                if tdesc:
                    upstream_desc[t.fqn] = tdesc

        # --- table-level coverage fill (retry empty tables) ------------------
        self._coverage_fill(tables, upstream_desc)

        # --- planned, controlled agentic loop --------------------------------
        # Each iteration measures coverage + per-item accuracy against the
        # configured targets, then a strategy selector (LLM decision with a
        # deterministic fallback) chooses the next action:
        #   inherit — copy descriptions from same-named columns elsewhere (free)
        #   sibling — infer empty columns from a table's described columns (LLM)
        #   rework  — regenerate the lowest-similarity items, judge-gated (LLM)
        #   stop    — targets met / no candidates remain
        # The loop is bounded by max_agent_iters; inherit/sibling run at most once,
        # rework may recur (each pass targets a fresh low-similarity tail).
        excluded: set[str] = set()   # strategies that are done (ran once, or yielded 0)
        temp = self.cfg.temperature_start
        self.trace: list[tuple[str, dict]] = []  # audit of decisions per iteration
        for _ in range(self.cfg.max_agent_iters):
            gaps = self._measure_gaps(tables)
            strategy = self._agent_decide(gaps, excluded)
            self.trace.append((strategy, gaps))
            if strategy == "stop":
                break
            if strategy == "inherit":
                yielded = self._apply_inherit(tables)
                excluded.add("inherit")             # coverage strategies run once
            elif strategy == "sibling":
                yielded = self._apply_sibling(tables, upstream_desc)
                excluded.add("sibling")
            else:  # rework
                temp += self.cfg.temperature_step
                yielded = self._apply_rework(tables, upstream_desc, temp)
            if yielded == 0:                         # a strategy that stops helping is done
                excluded.add(strategy)

        # --- confidence tagging (last, so it never pollutes similarity scoring) ---
        if self.cfg.tag_confidence:
            self._apply_tags(tables)

        return self._results(tables)

    def _apply_tags(self, tables: list[Table]) -> None:
        """Prefix each description with a High/Low confidence tag based on how well the
        object was grounded. High = had code/doc context, table lineage, or column
        lineage; Low = schema and column names only."""
        for t in tables:
            table_grounded = bool(t.code_context or t.doc_context or t.upstreams)
            if t.generated_description:
                t.generated_description = _tag(t.generated_description, table_grounded)
            for c in t.columns:
                if c.generated_description:
                    c.generated_description = _tag(c.generated_description,
                                                   bool(c.upstreams) or table_grounded)

    # ------------------------------------------------------------------ passes
    def _canonical_prepass(self, tables: list[Table]) -> None:
        """Generate one description per high-frequency canonical concept and seed it
        onto every occurrence, so wave generation only describes the long tail.

        This is the *canonicalize-first* strategy: it turns thousands of repeated
        columns (e.g. ``dw_insert_ts`` in every table) into a single batched call and
        guarantees identical wording for identical concepts.
        """
        concepts = high_frequency_concepts(tables, self.cfg.canonical_min_freq)
        if not concepts:
            return
        keys = list(concepts)
        prompts = []
        for key in keys:
            members = concepts[key]
            example = members[0][1]
            types = sorted({c.data_type for _, c in members if c.data_type})[:3]
            tabs = sorted({t.schema + "." + t.name for t, _ in members})[:6]
            prompts.append(
                f"Concept: a column named '{example.name}' (type(s): {', '.join(types) or 'n/a'}) "
                f"that recurs across tables: {', '.join(tabs)}.\n"
                f"Write ONE concise, general description of this column that is accurate "
                f"wherever it appears. Return just the description text.")
        outs = self.llm.batch(prompts, system=_SYSTEM,
                              temperature=self.cfg.temperature_start, workers=self.cfg.workers)
        for key, raw in zip(keys, outs):
            desc = raw.strip().strip('"')
            if not desc:
                continue
            for _t, c in concepts[key]:
                c.generated_description = desc
                c.prompt_context = prompts[keys.index(key)]

    def _coverage_fill(self, tables: list[Table], upstream_desc: dict[str, str]) -> None:
        """Retry tables that came back without a description (table-level coverage)."""
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

    # -------------------------------------------------- controller: measurement
    def _measure_gaps(self, tables: list[Table]) -> dict:
        """Measure coverage, per-item accuracy, and per-strategy candidate counts —
        the state the strategy selector reasons over each iteration."""
        total_cols = sum(len(t.columns) for t in tables)
        described_cols = sum(1 for t in tables for c in t.columns if c.generated_description)
        described_tables = sum(1 for t in tables if t.generated_description)
        described_keys = {canonical_key(c.name) for t in tables for c in t.columns
                          if c.generated_description}
        inherit_c = sum(1 for t in tables for c in t.columns
                        if not c.generated_description and canonical_key(c.name) in described_keys)
        sibling_c = sum(1 for t in tables
                        if any(not c.generated_description for c in t.columns)
                        and any(c.generated_description for c in t.columns))
        scored = [cosine(self.index, c.generated_description, c.prompt_context)
                  for t in tables for c in t.columns
                  if c.generated_description and c.prompt_context]
        scored += [cosine(self.index, t.generated_description, t.prompt_context)
                   for t in tables if t.generated_description and t.prompt_context]
        accuracy = sum(scored) / len(scored) if scored else 0.0
        rework_c = sum(1 for s in scored if s < self.cfg.rework_sim_threshold)
        return {
            "col_coverage": described_cols / total_cols if total_cols else 1.0,
            "table_coverage": described_tables / len(tables) if tables else 1.0,
            "accuracy": round(accuracy, 4),
            "inherit_candidates": inherit_c,
            "sibling_candidates": sibling_c,
            "rework_candidates": rework_c,
        }

    def _agent_decide(self, gaps: dict, excluded: set) -> str:
        """Pick the next strategy. LLM decision with a deterministic fallback.

        Coverage strategies (inherit, sibling) run at most once; rework may recur while
        it keeps yielding improvements and accuracy is below target. ``excluded`` holds
        strategies already run / exhausted. Returns one of inherit|sibling|rework|stop.
        """
        coverage_avail = [s for s in ("inherit", "sibling")
                          if s not in excluded and gaps[f"{s}_candidates"] > 0]
        rework_avail = ("rework" not in excluded and gaps["rework_candidates"] > 0
                        and gaps["accuracy"] < self.cfg.target_accuracy)
        if not coverage_avail and not rework_avail:
            return "stop"
        options = coverage_avail + (["rework"] if rework_avail else []) + ["stop"]
        choice = self._ask_llm_strategy(gaps, options)
        if choice in options:
            return choice
        # deterministic fallback: fill coverage first, then rework, then stop
        if coverage_avail:
            return coverage_avail[0]
        return "rework" if rework_avail else "stop"

    def _ask_llm_strategy(self, gaps: dict, options: list[str]) -> str:
        prompt = (
            f"Coverage: {gaps['col_coverage']:.0%} columns (target {self.cfg.target_coverage:.0%}). "
            f"Accuracy: {gaps['accuracy']:.2f} (target {self.cfg.target_accuracy:.2f}).\n"
            f"Candidates: inherit={gaps['inherit_candidates']}, sibling={gaps['sibling_candidates']}, "
            f"rework={gaps['rework_candidates']}.\n"
            f"Choose the single next strategy from: {', '.join(options)}. "
            "Prefer filling coverage (inherit, then sibling) before rework; choose stop only "
            "when no candidates remain and accuracy >= target. Reply with exactly one word.")
        try:
            out = self.llm.generate(
                prompt, system="You select the next data-documentation strategy.").lower()
            for w in re.findall(r"[a-z]+", out):
                if w in options:
                    return w
        except Exception:
            pass
        return ""

    # ---------------------------------------------------- controller: strategies
    def _apply_inherit(self, tables: list[Table]) -> int:
        """inherit — copy descriptions from same-named (canonical) columns elsewhere,
        and from fine-grained upstream columns. Free and instant (no LLM call)."""
        by_col: dict[tuple[str, str], str] = {}
        for t in tables:
            for c in t.columns:
                if c.generated_description:
                    by_col[(t.fqn, c.name.lower())] = c.generated_description
        n = 0
        if self.cfg.inherit_lineage:
            for t in tables:
                for c in t.columns:
                    if c.generated_description or not c.upstreams:
                        continue
                    for up_fqn, up_col in c.upstreams:
                        d = by_col.get((up_fqn, up_col.lower())) or next(
                            (v for (fq, cn), v in by_col.items()
                             if cn == up_col.lower() and (fq.endswith(up_fqn) or up_fqn.endswith(fq))), "")
                        if d:
                            c.generated_description = d
                            n += 1
                            break
        for _key, members in group_columns(tables).items():
            described = [c for _, c in members if c.generated_description]
            if not described or len(members) == len(described):
                continue
            best = max(described, key=lambda c: (
                cosine(self.index, c.generated_description, c.prompt_context)
                if c.prompt_context else 0.0, len(c.generated_description)))
            for _t, c in members:
                if not c.generated_description:
                    c.generated_description = best.generated_description
                    c.prompt_context = c.prompt_context or best.prompt_context
                    n += 1
        return n

    def _apply_sibling(self, tables: list[Table], upstream_desc: dict[str, str]) -> int:
        """sibling — infer a table's still-empty columns from its already-described
        columns (one small LLM call per table)."""
        targets = [t for t in tables
                   if any(not c.generated_description for c in t.columns)
                   and any(c.generated_description for c in t.columns)]
        if not targets:
            return 0
        prompts = []
        for t in targets:
            described = [f"{c.name}: {c.generated_description}"
                         for c in t.columns if c.generated_description][:40]
            empties = [c.name for c in t.columns if not c.generated_description]
            prompts.append(
                f"Table {t.fqn} ({t.layer}). Already-described columns:\n" + "\n".join(described) +
                f"\n\nDescribe the remaining columns consistently with the above and the table's "
                f"purpose. Return strict JSON {{\"columns\": {{\"<col>\": \"<desc>\"}}}}. "
                f"Columns to describe: {', '.join(empties)}")
        outs = self.llm.batch(prompts, system=_SYSTEM,
                              temperature=self.cfg.temperature_start, workers=self.cfg.workers)
        n = 0
        for t, p, raw in zip(targets, prompts, outs):
            _td, cdescs = _parse(raw)
            for c in t.columns:
                if not c.generated_description:
                    d = cdescs.get(c.name, cdescs.get(c.name.lower(), ""))
                    if d:
                        c.generated_description = d
                        c.prompt_context = c.prompt_context or p
                        n += 1
        return n

    def _apply_rework(self, tables: list[Table], upstream_desc: dict[str, str], temp: float) -> int:
        """rework — regenerate the lowest-similarity table descriptions with a
        vocab-reuse nudge and a raised temperature; an LLM-judge gates replacement."""
        scored = [(t, cosine(self.index, t.generated_description, t.prompt_context))
                  for t in tables if t.generated_description and t.prompt_context]
        worst = [t for t, s in scored if s < self.cfg.rework_sim_threshold]
        if not worst:
            return 0
        prompts = [_prompt(t, upstream_desc, self.cfg.model_ctx_chars)
                   + "\n\nReuse concrete vocabulary from the context above." for t in worst]
        outs = self.llm.batch(prompts, system=_SYSTEM, temperature=temp, workers=self.cfg.workers)
        pairs = []
        for t, raw in zip(worst, outs):
            tdesc, _c = _parse(raw)
            if tdesc:
                pairs.append((t, t.generated_description, tdesc))
        # Only consider items whose regenerated text actually differs from the current
        # one — an identical regeneration is not an improvement and must not keep the
        # rework strategy alive (otherwise the controller spins).
        pairs = [(t, old, new) for (t, old, new) in pairs if new.strip() != old.strip()]
        if not pairs:
            return 0
        approved = self._judge(pairs) if self.cfg.judge else {id(p[0]) for p in pairs}
        n = 0
        for t, _old, new in pairs:
            if id(t) in approved:
                t.generated_description = new
                t.rework_iters += 1
                upstream_desc[t.fqn] = new
                n += 1
        return n

    def _judge(self, pairs: list[tuple]) -> set:
        """LLM-judge (with a deterministic grounding fallback): approve a reworked
        description only if it is a genuine improvement over the previous one."""
        if not pairs:
            return set()
        approved: set = set()
        listing = "\n".join(f"{i}. OLD: {old}\n   NEW: {new}"
                            for i, (_o, old, new) in enumerate(pairs))
        prompt = ("For each item decide whether NEW is a more accurate, better-grounded "
                  "description than OLD. Reply one line per item as '<index> APPROVE' or "
                  "'<index> REJECT'.\n" + listing)
        try:
            out = self.llm.generate(prompt, system="You are a strict data-documentation reviewer.")
            for line in out.splitlines():
                m = re.match(r"\s*(\d+)\D+(approve|reject)", line.strip().lower())
                if m and m.group(2) == "approve":
                    approved.add(id(pairs[int(m.group(1))][0]))
            if approved:
                return approved
        except Exception:
            pass
        # deterministic fallback: approve if NEW is at least as grounded as OLD
        for obj, old, new in pairs:
            ctx = getattr(obj, "prompt_context", "")
            if not ctx or cosine(self.index, new, ctx) >= cosine(self.index, old, ctx):
                approved.add(id(obj))
        return approved

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
