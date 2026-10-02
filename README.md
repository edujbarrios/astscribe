# ASTScribe

> **Evidence-backed scientific explanations for ML notebooks, without LLMs.**

ASTScribe is an open-source, lightweight static semantic analysis library for Python machine-learning notebooks. It transforms ML code into traceable scientific explanations using Python's AST, framework-specific semantic rules, deterministic pattern recognition, and notebook-aware static context.

**No LLMs. No API keys. No code execution. No telemetry.**

ASTScribe is fully open source and designed to remain independent of proprietary AI services.

## Why ASTScribe?

Machine-learning notebooks often contain the complete implementation of an experiment while leaving its methodology distributed across many lines and cells. ASTScribe turns code structure into a conservative scientific description without executing the notebook or sending source code to an external service.

The first development focus is **PyTorch**.

## Features

- Python AST-based static analysis.
- Evidence-backed claims with source line and notebook-cell traceability.
- PyTorch-aware semantic rules.
- Training and inference pattern recognition.
- Scientific, educational, and concise rendering styles.
- Static resolution of imports, aliases, literal variables, and constructor bindings.
- Forward-only cross-cell notebook context through `NotebookAnalyzer`.
- Static recovery of optimizer type and explicit hyperparameters from earlier cells.
- Optional Jupyter/IPython `%%scribe` magic.
- Zero mandatory runtime dependencies outside the Python standard library.
- No telemetry or source-code upload.

## Installation

ASTScribe is not published on PyPI yet. Install the development version directly from the repository:

```bash
git clone https://github.com/edujbarrios/astscribe.git
cd astscribe
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

For optional IPython/Jupyter integration:

```bash
pip install -e ".[ipython]"
```

## Quick Start

```python
from astscribe import explain

code = """
model.eval()

with torch.no_grad():
    outputs = model(inputs)
"""

print(explain(code, style="scientific"))
```

Example output:

```text
Inference procedure

The model is explicitly configured in evaluation mode.

Gradient tracking is disabled for the enclosed operations, so no autograd graph is constructed for computations executed within this context.

A forward pass is performed by invoking the model on the supplied inputs.
```

## Scientific Explanation

```python
from astscribe import explain

code = """
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
"""

print(explain(code, style="scientific"))
```

ASTScribe identifies the sequence as a training step and constructs its explanation from explicit static-analysis rules rather than free-form generation.

## Evidence Model

Every claim is linked to evidence:

- **E1** — directly observed in the AST.
- **E2** — resolved from static symbols or notebook context.
- **E3** — derived from known framework semantics.
- **E4** — general methodological interpretation.

The scientific renderer currently restricts itself to E1-E3 claims.

```python
from astscribe import analyze

result = analyze("model.eval()")

for claim in result.claims:
    print(claim.text)
    print(claim.evidence_level, claim.line_start, claim.rule)
```

Evidence produced by notebook analysis can also identify the originating cell.

## Cross-Cell Notebook Analysis

ASTScribe can incrementally build static context across notebook cells without executing them:

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer()

notebook.add_cell("""
import torch
from torch.optim import AdamW
learning_rate = 2e-5
""")

notebook.add_cell("""
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
)
""")

result = notebook.add_cell("""
optimizer.zero_grad()
loss.backward()
optimizer.step()
""")

print(result.render("scientific"))
```

The final cell can recover that `optimizer` refers to an `AdamW` instance configured in an earlier cell, including statically resolvable hyperparameters such as the learning rate and weight decay.

Context is intentionally **forward-only**. Later cells never retroactively change the interpretation of earlier cells.

## PyTorch Support

The initial rule set covers a focused subset of PyTorch, including:

- `model.train()` / `model.eval()`
- `torch.no_grad()` / `torch.inference_mode()`
- `loss.backward()`
- `optimizer.zero_grad()` / `optimizer.step()`
- Adam, AdamW, and SGD configuration
- common loss constructors
- `torch.manual_seed(...)`
- checkpoint load/save operations
- basic DataLoader configuration
- common training and inference patterns
- optimizer context recovered across notebook cells
- loss-constructor context recovered across notebook cells

The project intentionally prefers a small set of defensible rules over broad heuristics.

## Jupyter Usage

Install the optional IPython extra, then load the extension:

```python
%load_ext astscribe.ipython
```

Analyze a cell without executing it:

```python
%%scribe
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

Choose another renderer by passing the style after the magic:

```python
%%scribe educational
model.eval()
```

## Architecture

```text
Notebook Cells
    │
    ▼
Python AST
    │
    ▼
Import, Symbol & Cross-Cell Resolution
    │
    ▼
Framework Semantic Rules
    │
    ▼
Pattern Recognition
    │
    ▼
SIR
Scientific Interpretation Representation
    │
    ▼
Evidence-Backed Claims
    │
    ▼
Scientific Renderer
```

This architecture is intentionally deterministic and inspectable.

## Open Source

ASTScribe is fully open source from its first commit. The analysis engine, semantic rules, renderers, and notebook integration are all available in this repository under the **Apache License 2.0**.

The project does not require proprietary APIs, external AI services, telemetry systems, or closed-source runtime components.

Community contributions are welcome through issues and pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src/astscribe
```

## Roadmap

### v0.1

- Python AST parser
- PyTorch semantic rules
- training/inference pattern detection
- scientific renderer
- evidence tracking
- forward-only cross-cell notebook context
- static optimizer and loss binding resolution
- `%%scribe` IPython magic

### v0.2

- richer DataLoader and preprocessing analysis
- reproducibility inspection
- richer optimizer/loss support
- notebook-level methodology graph

### v0.3

- Hugging Face Transformers
- notebook-level methodology extraction
- Methods-section generation

### Future

- scikit-learn
- TensorFlow/Keras
- notebook dependency graphs
- richer Jupyter visualization
- community semantic-rule packs

## Citation

If ASTScribe contributes to academic or scientific work, please cite the software using [CITATION.cff](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Author

**Eduardo J. Barrios**  
GitHub: [@edujbarrios](https://github.com/edujbarrios)
