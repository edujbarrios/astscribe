# ASTScribe

> Evidence-backed scientific explanations for ML notebooks, without LLMs.

[![PyPI version](https://img.shields.io/pypi/v/astscribe.svg)](https://pypi.org/project/astscribe/)
[![Python versions](https://img.shields.io/pypi/pyversions/astscribe.svg)](https://pypi.org/project/astscribe/)
[![Tests](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml/badge.svg)](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/edujbarrios/astscribe.svg)](https://github.com/edujbarrios/astscribe/blob/main/LICENSE)

ASTScribe statically analyzes Python and Jupyter notebook code and turns supported
machine-learning operations into deterministic, traceable scientific explanations.

**No LLMs. No API keys. No code execution. No telemetry.**

It currently has first-class semantics for **PyTorch**, **Hugging Face Transformers**,
**Hugging Face Datasets**, and **PEFT**.

## Install

```bash
python -m pip install astscribe
```

Optional Jupyter/IPython integration:

```bash
python -m pip install "astscribe[ipython]"
```

## Quick start

```python
from astscribe import explain

code = """
model.eval()
with torch.no_grad():
    outputs = model(inputs)
"""

print(explain(code, style="scientific"))
```

ASTScribe only reports claims supported by the source it can inspect. It does not execute
the code or import ML frameworks at runtime.

## Analyze a notebook

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_ipynb("experiment.ipynb")

print(notebook.render_methodology(include_evidence=True))
print(notebook.render_pipeline())
print(notebook.render_techniques())
print(notebook.render_diagnostics())
```

Notebook context flows forward across cells, so ASTScribe can connect imports,
constructors, datasets, optimizers, adapters, and other supported symbols to later uses.

## Command line

Explain a Python file:

```bash
astscribe training.py --style concise
```

Pipe Python source through stdin with `-`:

```bash
printf 'model.eval()\n' | astscribe - --style concise
```

Inspect a notebook:

```bash
astscribe experiment.ipynb --report methodology --evidence
astscribe experiment.ipynb --report pipeline
astscribe experiment.ipynb --report techniques
astscribe experiment.ipynb --report dependencies
astscribe experiment.ipynb --report diagnostics
astscribe experiment.ipynb --report impact --cell 3
```

The same CLI is available as `python -m astscribe`.

## What ASTScribe detects

ASTScribe focuses on conservative, auditable signals such as:

- training and inference mode, forward passes, objectives, optimizers, schedulers, AMP,
  gradient clipping, devices, DataLoaders, checkpoints, and reproducibility;
- Transformers pretrained components, tokenization, Trainer workflows, generation,
  decoding, and quantization;
- Datasets loading and common preparation transforms;
- PEFT/LoRA configuration, adapter application, k-bit preparation, checkpoints, and
  merging;
- notebook-level methodology, experiment pipelines, dependency graphs, diagnostics,
  impact analysis, and conservative composite techniques such as QLoRA.

Detailed semantic contracts and limitations live in the
[docs](https://github.com/edujbarrios/astscribe/tree/main/docs) directory.

## Evidence model

Generated claims retain source provenance and an evidence level:

- **E1** — directly observed in the AST;
- **E2** — resolved from static symbols or notebook context;
- **E3** — derived from known framework semantics;
- **E4** — general methodological interpretation.

The scientific renderer uses E1-E3 claims by default.

## Development

```bash
git clone https://github.com/edujbarrios/astscribe.git
cd astscribe
python -m pip install -e ".[dev]"

pytest
ruff check .
mypy src/astscribe
```

See [CONTRIBUTING.md](https://github.com/edujbarrios/astscribe/blob/main/CONTRIBUTING.md)
for contribution guidance and
[docs/releasing.md](https://github.com/edujbarrios/astscribe/blob/main/docs/releasing.md)
for the release process.

## Citation

If ASTScribe contributes to academic or scientific work, cite the software using
[CITATION.cff](https://github.com/edujbarrios/astscribe/blob/main/CITATION.cff).

## License

Apache License 2.0. See [LICENSE](https://github.com/edujbarrios/astscribe/blob/main/LICENSE).
