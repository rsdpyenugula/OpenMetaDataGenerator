## What & why
Describe the change and its motivation.

## Type
- [ ] Bug fix
- [ ] New metadata source / LLM backend / context provider
- [ ] Benchmark / evaluation
- [ ] Docs / paper

## Checklist
- [ ] `make test` passes (no network / API keys required)
- [ ] `ruff check openmetadatagenerator benchmark` is clean
- [ ] New provider dependency (if any) is declared as an extra in `pyproject.toml` and imported lazily
- [ ] New behavior is covered by a test using the deterministic `MockBackend`
- [ ] No provider-specific SDK imported outside its backend/source module
