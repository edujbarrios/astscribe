# Contributing to ASTScribe

ASTScribe is open source and welcomes contributions through issues and pull requests.

## Development setup

```bash
git clone https://github.com/edujbarrios/astscribe.git
cd astscribe
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Run the quality checks with:

```bash
pytest
ruff check .
mypy src/astscribe
```

Release preparation and Trusted Publishing setup are documented in
[`docs/releasing.md`](docs/releasing.md).

## Adding a semantic rule

A semantic rule should:

1. match source structure statically;
2. emit a structured SIR operation;
3. attach explicit evidence and source locations;
4. avoid unsupported methodological claims;
5. include focused tests;
6. never execute analyzed source code.

Prefer a small, defensible rule over a broad heuristic that can produce misleading explanations.
