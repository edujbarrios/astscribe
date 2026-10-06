# ASTScribe

> Evidence-backed scientific explanations for ML notebooks, without LLMs.

[![PyPI version](https://img.shields.io/pypi/v/astscribe.svg)](https://pypi.org/project/astscribe/)
[![Python versions](https://img.shields.io/pypi/pyversions/astscribe.svg)](https://pypi.org/project/astscribe/)
[![PyPI downloads](https://img.shields.io/pypi/dm/astscribe.svg)](https://pypi.org/project/astscribe/)
[![Tests](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml/badge.svg)](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/edujbarrios/astscribe.svg)](https://github.com/edujbarrios/astscribe/blob/main/LICENSE)

ASTScribe statically analyzes Python and Jupyter notebooks and turns supported ML
operations into deterministic, traceable explanations.

**No LLMs. No API keys. No code execution. No telemetry.**

## Install

```bash
python -m pip install astscribe
```

Optional IPython/Jupyter integration:

```bash
python -m pip install "astscribe[ipython]"
```

## Quick start

```python
from astscribe import explain

print(explain("""
model.eval()
with torch.no_grad():
    outputs = model(inputs)
""", style="scientific"))
```

For notebooks:

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_ipynb("experiment.ipynb")
print(notebook.render_methodology(include_evidence=True))
print(notebook.render_diagnostics())
```

ASTScribe only reports claims supported by the source it can inspect.

## CLI

```bash
astscribe training.py --style concise
astscribe experiment.ipynb --report methodology --evidence
astscribe experiment.ipynb --report diagnostics
astscribe experiment.ipynb --report impact --cell 3

# Read Python source from stdin
printf 'model.eval()\n' | astscribe - --style concise

# Fail if a notebook contains a code cell that is not valid Python
astscribe experiment.ipynb --report diagnostics --strict
```

The same CLI is available as `python -m astscribe`.

## Supported semantics

ASTScribe has first-class static semantics for **PyTorch**, **Hugging Face
Transformers**, **Hugging Face Datasets**, and **PEFT**. It can reconstruct notebook
methodology, experiment pipelines, cross-cell dependencies, diagnostics, impact, and
conservative composite techniques such as QLoRA.

Generated claims retain source provenance and an evidence level. The scientific
renderer uses directly observed, statically resolved, and known-framework claims by
default.

Detailed contracts and limitations live in [docs](https://github.com/edujbarrios/astscribe/tree/main/docs).

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src/astscribe
```

See [CONTRIBUTING.md](https://github.com/edujbarrios/astscribe/blob/main/CONTRIBUTING.md)
and [docs/releasing.md](https://github.com/edujbarrios/astscribe/blob/main/docs/releasing.md).

## License

Apache License 2.0. See [LICENSE](https://github.com/edujbarrios/astscribe/blob/main/LICENSE).
