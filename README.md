# ASTScribe

> Evidence-backed scientific explanations for ML notebooks, without LLMs.

[![PyPI](https://img.shields.io/pypi/v/astscribe?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/astscribe/)
[![Python](https://img.shields.io/pypi/pyversions/astscribe?logo=python&logoColor=white)](https://pypi.org/project/astscribe/)
[![CI](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml)
[![Downloads](https://img.shields.io/pypi/dm/astscribe?label=downloads)](https://pypi.org/project/astscribe/)
[![License](https://img.shields.io/pypi/l/astscribe)](https://github.com/edujbarrios/astscribe/blob/main/LICENSE)

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

ASTScribe analyzes source code without executing it:

```python
from astscribe import explain

source = """
import torch

model.eval()
with torch.no_grad():
    outputs = model(inputs)
"""

print(explain(source, style="scientific"))
```

Output:

```text
Inference procedure

Gradient tracking is disabled for the enclosed operations, so no autograd graph is constructed for computations executed within this context.

The model is explicitly configured in evaluation mode.

A forward pass is performed by invoking the model on the supplied inputs.
```

For notebook-level structure:

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_cells(
    [
        "dataset = load_data()",
        "features = preprocess(dataset)",
        "result = evaluate(features)",
    ]
)

print(notebook.render_dependency_graph())
```

Output:

```text
# Cell dependency graph

Cell 0 -> Cell 1 [dataset]
Cell 1 -> Cell 2 [features]
```

ASTScribe only reports claims supported by the source it can inspect.

## CLI

```bash
astscribe training.py --style concise
astscribe experiment.ipynb --report methodology --evidence
astscribe experiment.ipynb --report diagnostics
astscribe experiment.ipynb --report impact --cell 3
astscribe experiment.ipynb --report ranking
astscribe experiment.ipynb --report dependencies --dot

# Structured output for scripts, CI, and jq
astscribe training.py --json
astscribe experiment.ipynb --report diagnostics --json
astscribe experiment.ipynb --report impact --cell 3 --json

# Read Python source from stdin
printf 'model.eval()\n' | astscribe - --style concise

# Fail if a notebook contains a code cell that is not valid Python
astscribe experiment.ipynb --report diagnostics --strict
```

The same CLI is available as `python -m astscribe`. Use `--json` with Python source
or any notebook report to emit the underlying structured analysis instead of rendered
text. JSON output is deterministic and uses only the Python standard library.

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
