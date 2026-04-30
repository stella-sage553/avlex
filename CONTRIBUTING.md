# Contributing to avlex

Thanks for taking a look! avlex is a small, focused framework and contributions
that keep it small and focused are very welcome.

## Development setup

```bash
git clone https://github.com/stella-sage553/avlex
cd avlex
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Checks

The same checks CI runs, locally:

```bash
ruff check .          # lint
ruff format .         # format
mypy                  # type-check
pytest                # tests (with coverage in CI)
```

All four must pass. `pre-commit` runs lint/format/mypy on staged files.

## Adding a component

avlex is built from four pluggable abstractions. To add one:

1. Subclass the relevant base — `Encoder`, `Bridge`, `Fusion`, or `LLMClient`.
2. Keep the core dependency-free; put heavy backends behind a lazy import and an
   optional extra (see `avlex/llm/openai_client.py`).
3. Register it (`register_bridge`, `register_encoder`, ...) so it works from a
   config file.
4. Add a test mirroring the existing ones, and a line in `CHANGELOG.md`.

## Style

- Type hints on public functions; docstrings explaining the *why*.
- Small, single-concern commits.
- No new required dependency without a discussion in an issue first.

## Reporting bugs

Open an issue with a minimal reproduction — ideally a few lines using
`synthetic_clip()`.
