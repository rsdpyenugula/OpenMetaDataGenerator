.PHONY: install test benchmark paper clean

install:
	pip install -e ".[all]"

test:
	python -m pytest -q tests/

benchmark:
	python -m benchmark.run

# Reproduce with a real cloud LLM, e.g.: make benchmark-llm PROVIDER=anthropic MODEL=claude-3-5-sonnet-latest
benchmark-llm:
	python -m benchmark.run --provider $(PROVIDER) --model $(MODEL)

paper:
	cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

clean:
	rm -rf build dist *.egg-info **/__pycache__ paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out
