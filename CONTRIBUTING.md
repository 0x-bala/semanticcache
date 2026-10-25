# Contributing

Thanks for your interest in improving semantic-cache. Bug reports, benchmarks,
documentation fixes and new features are all welcome.

## Getting started

```bash
git clone https://github.com/0x-bala/semanticcache.git
cd semanticcache
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run the checks before every pull request:

```bash
python -m ruff check .
python -m pytest -q -m "not live"
python -m build
```

CI runs the same checks on Python 3.10, 3.11, 3.12 and 3.13.

## Making a change

1. For anything larger than a small fix, open an issue first so the approach can be
   discussed.
2. Create a branch from `main`.
3. Keep commits focused, with a short imperative subject ("Fix LFU eviction of new entries").
4. Add or update tests for every behavior change. Bug fixes should include a test that
   fails without the fix.
5. Open a pull request that says what changed, why, and how you tested it.

## Guidelines

- **No network in tests.** Use `HashingEmbedder` or a small deterministic stub. Tests that
  need a real API must be marked `@pytest.mark.live` and must skip cleanly without a key or
  when the quota is exhausted.
- **Never commit secrets.** API keys come from environment variables. Do not paste keys
  into code, tests, logs or benchmark output.
- **Fail safe.** When the cache cannot decide (an error, a timeout, an ambiguous match), it
  must return a miss, never a possibly wrong cached answer.
- **Style.** Ruff with a line length of 100 (`ruff check .`). Public functions need type
  hints and docstrings that describe behavior and parameters.
- **Dependencies.** The core package depends on NumPy only. Anything else belongs in an
  optional extra and must be imported lazily.
- **Do not commit generated files:** benchmark datasets, embeddings (`benchmarks/data/`),
  database files, patches or installers.

## Changing the hit guard

The guard (`semantic_cache/guard.py`) decides when two prompts are interchangeable, so
changes to it need evidence:

- Add test cases that show the new behavior and a case that must still be rejected.
- Run `python benchmarks/guard_rules.py` and the accuracy benchmark
  (`benchmarks/real_embeddings.py --part 2`) and include the before and after numbers in
  the pull request.
- A change that lowers the number of wrong answers served is preferred over one that only
  raises the hit rate.

## Benchmarks

Benchmark scripts live in `benchmarks/` and record results in `benchmarks/results_real.json`.
Report what was measured, on which machine and dataset, and label anything synthetic.

## Reporting bugs

Open an issue with the version (`python -c "import semantic_cache; print(semantic_cache.__version__)"`),
Python version, a minimal reproduction, and the expected and actual behavior.

## Security

Do not report vulnerabilities in public issues. Use GitHub Security Advisories for this
repository. See [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contributions are licensed under the MIT License.
