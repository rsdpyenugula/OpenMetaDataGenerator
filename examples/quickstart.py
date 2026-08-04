"""Minimal end-to-end example using the deterministic mock backend (no API keys).

Run:  python examples/quickstart.py
It builds a tiny synthetic catalog, generates descriptions, and writes a CSV.
Swap ``MockBackend()`` for ``get_backend("anthropic")`` (etc.) to use a real LLM.
"""
from benchmark.generate import build_benchmark
from benchmark.mock_llm import MockBackend
from openmetadatagenerator.context.embedding import EmbeddingIndex
from openmetadatagenerator.generation import Generator
from openmetadatagenerator.output import write_csv

tables, _ = build_benchmark(n_entities=3, seed=42, context_prob=0.9)
results = Generator(MockBackend(), EmbeddingIndex()).run(tables)
n = write_csv(results, "outputs/quickstart.csv")

print(f"generated {n} descriptions -> outputs/quickstart.csv")
for r in results[:6]:
    print(f"  [{r.object_type}] {r.object_name}\n      {r.output}\n      grounding: {r.grounding_notes}")
