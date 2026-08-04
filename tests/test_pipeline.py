"""End-to-end tests using the deterministic mock backend (no API keys needed)."""
from __future__ import annotations

from openmetadatagenerator.context.embedding import EmbeddingIndex
from openmetadatagenerator.generation import Generator, _topo_waves
from openmetadatagenerator.config import GenerationConfig
from openmetadatagenerator.output import write_csv
from benchmark.generate import build_benchmark
from benchmark.evaluate import evaluate
from benchmark.mock_llm import MockBackend


def _gen(tables):
    return Generator(MockBackend(), EmbeddingIndex(), GenerationConfig(workers=2)).run(tables)


def test_topo_waves_respect_lineage():
    tables, _ = build_benchmark(n_entities=2, seed=1)
    waves = _topo_waves(tables)
    seen = set()
    for wave in waves:
        for t in wave:
            # every in-scope upstream must appear in an earlier wave
            for up in t.upstreams:
                assert any(up.endswith(s) or s in up for s in seen) or up not in {x.fqn for x in tables}
            seen.add(t.fqn)


def test_full_covers_all_objects():
    tables, gold = build_benchmark(n_entities=5, seed=7, context_prob=0.8)
    n = len(tables) + sum(len(t.columns) for t in tables)
    m = evaluate(_gen(tables), gold, n)
    assert m.coverage == 1.0
    assert m.n == n


def test_context_improves_grain_recovery():
    idx = EmbeddingIndex()
    t_ctx, gold = build_benchmark(n_entities=5, seed=7, context_prob=1.0)
    t_none, _ = build_benchmark(n_entities=5, seed=7, context_prob=0.0)
    n = len(t_ctx) + sum(len(x.columns) for x in t_ctx)
    m_ctx = evaluate(_gen(t_ctx), gold, n, idx)
    m_none = evaluate(_gen(t_none), gold, n, idx)
    assert m_ctx.exact_grain > m_none.exact_grain
    assert m_ctx.accuracy >= m_none.accuracy


def test_agentic_controller_terminates_and_traces():
    tables, _ = build_benchmark(n_entities=4, seed=7, context_prob=0.6)
    g = Generator(MockBackend(), EmbeddingIndex(), GenerationConfig(workers=2, max_agent_iters=6))
    g.run(tables)
    assert g.trace, "controller should record at least one decision"
    # every decision is a valid strategy, and it must not run to the cap forever
    assert all(s in ("inherit", "sibling", "rework", "stop") for s, _ in g.trace)
    assert g.trace[-1][0] == "stop" or len(g.trace) <= 6


def test_judge_gates_replacement():
    g = Generator(MockBackend(), EmbeddingIndex(), GenerationConfig())
    class _Obj:
        prompt_context = "orders table: one row per customer order with amount and status"
    obj = _Obj()
    # NEW is clearly more grounded than OLD -> approved by the deterministic fallback
    approved = g._judge([(obj, "a table", "orders: one row per customer order with amount")])
    assert id(obj) in approved


def test_csv_export(tmp_path):
    tables, _ = build_benchmark(n_entities=2, seed=3)
    out = tmp_path / "d.csv"
    n = write_csv(_gen(tables), str(out))
    assert out.exists() and n > 0
    assert out.read_text().splitlines()[0].startswith("object_type,object_name")
