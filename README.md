# ASTScribe

> Evidence-backed scientific explanations for ML notebooks, without LLMs.

[![PyPI](https://img.shields.io/pypi/v/astscribe?label=PyPI)](https://pypi.org/project/astscribe/)
[![Python](https://img.shields.io/pypi/pyversions/astscribe)](https://pypi.org/project/astscribe/)
[![CI](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml)

ASTScribe statically analyzes Python and Jupyter notebooks and produces deterministic,
traceable explanations of supported ML operations in **PyTorch**, **Transformers**,
**Datasets**, and **PEFT**.

**No LLMs. No API keys. No execution of analyzed code. No telemetry.**

## Installation

```bash
python -m pip install astscribe
```

For IPython/Jupyter magics, install the optional extra with
`python -m pip install "astscribe[ipython]"`.

## Explain inference and training

PyTorch is **not required** to run this example: the source inside the string is
*analyzed*, never executed.

```python
from astscribe import explain

source = """import torch
model.eval()
with torch.no_grad():
    outputs = model(inputs)
"""
print(explain(source, style="concise"))
```

Output:

```text
Performs PyTorch inference using evaluation-oriented execution semantics.
```

A training step can be analyzed in the same way:

```python
training = """model.train()
for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
"""
print(explain(training, style="concise"))
```

Output:

```text
Performs a PyTorch gradient-based training step with backward propagation and a parameter update.
```

Use `analyze(source)` to get structured operations, claims, provenance and
inference/training flags, instead of rendered text.

## Notebook dependencies

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_cells([
    "raw = 10",
    "features = raw * 2",
    "prediction = features + 1",
])
print(notebook.render_dependency_graph())
```

Output:

```text
# Cell dependency graph

Cell 0 -> Cell 1 [raw]
Cell 1 -> Cell 2 [features]
```

Find the downstream cells affected by changing an earlier cell:

```python
print(notebook.render_impact_ranking())
```

Output:

```text
# Notebook impact ranking

- Cell 0: 2 affected cell(s), 1 direct dependent(s).
- Cell 1: 1 affected cell(s), 1 direct dependent(s).
- Cell 2: 0 affected cell(s), 0 direct dependent(s).
```

## Analyze an existing notebook file

This command **reads** a committed example notebook; it does not execute its cells.

```python
from astscribe import NotebookAnalyzer

report = NotebookAnalyzer.from_ipynb("examples/notebooks/notebook_audit.ipynb")
print(report.cell_count)
```

Output:

```text
3
```

Further reports: `render_methodology()`, `render_pipeline()`,
`render_diagnostics()`, `render_impact(cell)`, `dependency_dot()`,
and structured `to_dict()` results. Analyzing a real `.ipynb` preserves
original cell indices, even when Markdown or unsupported cells appear between them.

## CLI

```bash
printf 'value = 1\n' | astscribe - --style concise
```

Output (no supported ML operations in this source):

```text
No supported PyTorch semantics were identified in the analyzed source.
```

Notebook reports are also available from the CLI, such as
`astscribe experiment.ipynb --report diagnostics --json` and
`astscribe experiment.ipynb --report dependencies --dot`. Use `--strict`
to reject unsupported notebook cells, and `--fail-on-warning` to report
dependency warnings with a non-zero exit status. Incompatible report and output
options are rejected rather than silently ignored.

## Jupyter / IPython

Install `astscribe[ipython]` and load the extension with
`%load_ext astscribe.ipython`. Then `%scribe N concise` explains
the previous `In[N]` input using earlier static context, and
`%%scribe concise` explains the current cell without running the body.
These magics display the same deterministic text in Markdown form.

## Executed notebooks with recorded outputs

All three example notebooks contain **real code cells and saved outputs** displayed
by GitHub's notebook viewer. They analyze code statically, so no ML frameworks,
datasets, GPUs or network calls are needed to run them.

- [Inference and training](examples/notebooks/inference_and_training.ipynb): concise explanations and structured inference flags.
- [Dependencies and impact](examples/notebooks/dependencies_and_impact.ipynb): dataflow, transitive impact paths, diagnostics.
- [Notebook audit](examples/notebooks/notebook_audit.ipynb): skipped IPython syntax, forward-reference warnings, and adding cells with correct original indices.

Re-run and compare the notebooks against their committed outputs after installing
`nbclient`, `nbformat` and `ipykernel`:

```bash
python scripts/verify_notebooks.py
```

Expected terminal output:

```text
Verified 3 executed example notebooks; all stored outputs match.
```

The verifier fails if any output differs, and runs in CI. To deliberately
refresh the recorded outputs, use
`python scripts/verify_notebooks.py --update`.

## Scope and limitations

ASTScribe provides evidence-backed semantics for PyTorch, Hugging Face
Transformers, Hugging Face Datasets and PEFT. It can infer methodology,
experiment pipelines, cross-cell dependencies, diagnostics, impact and
composite techniques such as QLoRA. Dynamic effects and hidden Jupyter
kernel state are not inferred. See the [technical documentation](docs/).

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src/astscribe
```

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). License: [Apache-2.0](LICENSE).
