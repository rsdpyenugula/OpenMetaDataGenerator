.PHONY: install test benchmark paper clean

install:
	pip install -e ".[all]"

test:
	python -m pytest -q tests/

demo:
	python examples/agentic_demo.py

rag-demo:
	python examples/rag_demo.py

benchmark:
	python -m benchmark.run

# Reproduce with a real cloud LLM, e.g.: make benchmark-llm PROVIDER=anthropic MODEL=claude-3-5-sonnet-latest
benchmark-llm:
	python -m benchmark.run --provider $(PROVIDER) --model $(MODEL)

# Run all three benchmarks with a real provider (writes paper/results_table*.tex).
# Example: GEMINI_API_KEY=... make benchmark-llm-all PROVIDER=gemini MODEL=gemini-2.0-flash
benchmark-llm-all:
	python -m benchmark.run --benchmark synthetic --provider $(PROVIDER) --model $(MODEL)
	python -m benchmark.run --benchmark tpch      --provider $(PROVIDER) --model $(MODEL)
	python -m benchmark.run --benchmark sakila    --provider $(PROVIDER) --model $(MODEL)

paper:
	cd paper && (tectonic main.tex || (pdflatex main && bibtex main && pdflatex main && pdflatex main))

paper-tmlr:
	cd paper && tectonic main_tmlr.tex

clean:
	rm -rf build dist *.egg-info **/__pycache__ paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out
