# Contributing to OpenMetaDataGenerator

Thanks for your interest in improving OMDG. This project favors small, well-tested,
provider-agnostic contributions.

## Development setup

```bash
git clone https://github.com/rsdpyenugula/OpenMetaDataGenerator
cd OpenMetaDataGenerator
pip install -e ".[all]"
pip install pytest ruff
make test          # unit tests (no API keys needed)
make benchmark     # reproducible ablations (deterministic backend)
```

## Ground rules

- **Keep the core interfaces stable.** New capabilities plug into one of the three
  extension points rather than special-casing the pipeline:
  - a metadata source implements `sources/base.py::MetadataSource`,
  - a context provider implements `context/base.py::ContextProvider`,
  - an LLM backend implements `llm/base.py::LLMBackend`.
- **No provider lock-in.** Nothing in `openmetadatagenerator/` (outside a backend module)
  may import a specific provider SDK. Optional dependencies are declared as extras in
  `pyproject.toml` and imported lazily inside the backend/source that needs them.
- **Determinism where possible.** Generation defaults to temperature 0; anything that adds
  nondeterminism must be opt-in and documented.
- **Tests must pass without network or API keys.** Use the deterministic `MockBackend`
  (see `benchmark/mock_llm.py`) for tests.

## Adding a new metadata source or LLM backend

1. Implement the interface in a new module.
2. Register it in the corresponding factory (`sources/__init__.py` or `llm/__init__.py`).
3. Add an extra in `pyproject.toml` for any new dependency.
4. Add a unit test that exercises normalization / generation via the mock.

## Pull requests

- Run `make test` and `ruff check openmetadatagenerator benchmark` before opening.
- Keep PRs focused; describe the change and its rationale.
- By contributing you agree your contributions are licensed under Apache-2.0.

## Reporting issues

Use the issue templates. For generation-quality issues, include the (sanitized) schema,
the context you provided, and the produced description with its `grounding_notes`.
