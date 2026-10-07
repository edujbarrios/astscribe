# ASTScribe

> Evidence-backed scientific explanations for ML notebooks, without LLMs.

[![PyPI](https://img.shields.io/pypi/v/astscribe?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/astscribe/)
[![Python](https://img.shields.io/pypi/pyversions/astscribe?logo=python&logoColor=white)](https://pypi.org/project/astscribe/)
[![CI](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml)
[![Downloads](https://img.shields.io/pypi/dm/astscribe?label=downloads)](https://pypi.org/project/astscribe/)
[![License](https://img.shields.io/pypi/l/astscribe)](https://github.com/edujbarrios/astscribe/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/edujbarrios/astscribe?display_name=tag&label=release)](https://github.com/edujbarrios/astscribe/releases)
[![Stars](https://img.shields.io/github/stars/edujbarrios/astscribe?label=stars)](https://github.com/edujbarrios/astscribe/stargazers)
[![Issues](https://img.shields.io/github/issues/edujbarrios/astscribe)](https://github.com/edujbarrios/astscribe/issues)
[![Last commit](https://img.shields.io/github/last-commit/edujbarrios/astscribe)](https://github.com/edujbarrios/astscribe/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/edujbarrios/astscribe)](https://github.com/edujbarrios/astscribe)

ASTScribe statically analyzes Python and Jupyter notebooks and turns supported ML
operations into deterministic, traceable explanations.

**No LLMs. No API keys. No code execution. No telemetry.**

## Install

```bash
python -m pip install astscribe
```

For Jupyter/IPython:

```bash
python -m pip install "astscribe[ipython]"
```

## Python

```python
from astscribe import explain

print(explain("import torch\nwith torch.no_grad():\n    output = model(inputs)"))
```

ASTScribe analyzes the source without executing it.

## Jupyter / IPython

Load the extension once, then explain any executed input by its `In[n]` number:

```python
%load_ext astscribe.ipython
%scribe 4
%scribe 7 concise
```

`%scribe N` rebuilds forward-only static context from earlier Python inputs and explains
`In[N]`. IPython-only syntax acts as a conservative context boundary because ASTScribe
does not execute or infer its runtime side effects.

To explain the cell you are writing:

```python
%%scribe concise
with torch.no_grad():
    output = model(inputs)
```

The `%%scribe` body is analyzed, not executed.

## Notebook files

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_ipynb("experiment.ipynb")
print(notebook.render_methodology())
```

Other notebook reports include dependencies, diagnostics, impact, impact ranking,
experiment pipelines, and composite techniques.

## CLI

```bash
astscribe training.py --style concise
astscribe experiment.ipynb --report methodology
astscribe experiment.ipynb --report diagnostics --json
astscribe experiment.ipynb --report impact --cell 3
```

The same CLI is available as `python -m astscribe`. Use `--strict` to reject notebook
code cells that are not valid Python and `--fail-on-warning` to return status 1 when
dependency diagnostics contain warnings.

## Supported semantics

ASTScribe has first-class static semantics for **PyTorch**, **Hugging Face
Transformers**, **Hugging Face Datasets**, and **PEFT**. It can reconstruct notebook
methodology, experiment pipelines, cross-cell dependencies, diagnostics, impact, and
conservative composite techniques such as QLoRA.

Generated claims retain source provenance and an evidence level. Detailed contracts and
limitations live in [docs](https://github.com/edujbarrios/astscribe/tree/main/docs).

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
