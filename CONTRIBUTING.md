# Contributing

Create focused commits with tests for every behavior change. Run these checks
before opening a pull request:

```bash
python -m pytest -q
python -m compileall -q semantic_cache tests demo.py
```

The CI workflow additionally runs Ruff and builds the package. Avoid network
calls in unit tests; use deterministic embedders and compute functions.
