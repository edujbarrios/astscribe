# ASTScribe

> **Evidence-backed scientific explanations for ML notebooks, without LLMs.**

ASTScribe is an open-source, lightweight static semantic analysis library for Python machine-learning notebooks. It transforms ML code into traceable scientific explanations using Python's AST, framework-specific semantic rules, deterministic pattern recognition, and notebook-aware static context.

**No LLMs. No API keys. No code execution. No telemetry.**

ASTScribe is fully open source under the **Apache License 2.0** and is designed to remain independent of proprietary AI services.

## Why ASTScribe?

Machine-learning notebooks often contain the complete implementation of an experiment while leaving its methodology distributed across many cells. ASTScribe turns that implementation into a conservative scientific description without executing the notebook or sending source code to an external service.

The first development focus is **PyTorch**.

## Features

- Python AST-based static analysis.
- Direct `.ipynb` loading using only the Python standard library.
- Evidence-backed claims with source-line and notebook-cell traceability.
- Forward-only cross-cell context through `NotebookAnalyzer`.
- PyTorch-aware semantic rules for training, inference, data pipelines, and evaluation workflows.
- Static recovery of optimizer type and explicit hyperparameters.
- Detection of epochs, devices, DataLoaders, schedulers, gradient clipping, and AMP constructs.
- Torchvision dataset, transform, augmentation, and model semantics.
- Structured experiment-pipeline reconstruction.
- Scientific, educational, and concise cell-level rendering styles.
- Deterministic notebook-level `# Methods` reports.
- Optional evidence appendix linking scientific claims back to source cells and lines.
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

## Analyze a Real Jupyter Notebook

ASTScribe can load a `.ipynb` directly without depending on `nbformat`, Jupyter, or PyTorch:

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_ipynb("training.ipynb")

print(notebook.render_methodology())
```

To include an auditable evidence appendix:

```python
print(notebook.render_methodology(include_evidence=True))
```

Notebook cell indices in the evidence report refer to the original `.ipynb` document, including Markdown cells between code cells.

IPython-specific syntax such as `%matplotlib` or shell commands such as `!pip install ...` is not treated as Python AST. By default, such cells are skipped and recorded explicitly:

```python
for skipped in notebook.skipped_cells:
    print(skipped.index, skipped.reason)
```

Strict parsing is also available:

```python
NotebookAnalyzer.from_ipynb("training.ipynb", skip_invalid_python=False)
```

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

## Notebook Methods Report

A notebook can be reduced to a deterministic paper-like Methods report:

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer()

notebook.add_cell("""
import torch
from torch import nn
from torch.optim import AdamW

torch.manual_seed(42)
epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
""")

notebook.add_cell("""
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
""")

print(notebook.render_methodology(include_evidence=True))
```

The report can include sections such as:

- reproducibility;
- dataset configuration;
- preprocessing and augmentation;
- data loading;
- execution environment;
- model architecture;
- model and objective;
- optimization;
- numerical precision;
- training procedure;
- evaluation and inference;
- metrics;
- checkpointing.

## Experiment Pipeline Reconstruction

ASTScribe also exposes a structured experiment pipeline built from detected operations rather than generated prose:

```python
notebook = NotebookAnalyzer.from_ipynb("training.ipynb")

print(notebook.render_pipeline())
```

Example:

```text
Dataset
    ↓
Preprocessing and augmentation
    ↓
Data loading
    ↓
Model architecture
    ↓
Objective
    ↓
Optimization
    ↓
Training procedure
    ↓
Evaluation and inference
    ↓
Metrics
    ↓
Checkpointing
```

The same information is available as structured data:

```python
pipeline = notebook.pipeline()

for stage in pipeline.stages:
    print(stage.key, stage.operations)

payload = pipeline.to_dict()
```

Only stages supported by detected evidence are included. A missing stage means that ASTScribe did not find sufficient supported evidence; it does not prove that the experiment omitted that stage.

See [`docs/experiment_pipeline.md`](docs/experiment_pipeline.md) for the current semantics and design constraints.

## Evidence Model

Every generated claim is linked to evidence:

- **E1** — directly observed in the AST.
- **E2** — resolved from static symbols or notebook context.
- **E3** — derived from known framework semantics.
- **E4** — general methodological interpretation.

The scientific renderer restricts itself to E1-E3 claims by default.

```python
from astscribe import analyze

result = analyze("model.eval()")

for claim in result.claims:
    print(claim.text)
    print(claim.evidence_level)
    print(claim.evidence.cell, claim.line_start, claim.line_end)
    print(claim.rule)
```

## Current PyTorch Coverage

The current rule set intentionally focuses on common, defensible methodology signals:

- `model.train()` / `model.eval()`;
- `torch.no_grad()` / `torch.inference_mode()`;
- model forward calls and objective evaluation;
- `loss.backward()`;
- `optimizer.zero_grad()` / `optimizer.step()`;
- Adam, AdamW, and SGD configuration;
- static learning rate, weight decay, and momentum extraction;
- epoch loops using statically resolvable `range(...)` counts;
- common loss constructors;
- `torch.manual_seed(...)`;
- DataLoader batch size, shuffling, and worker configuration;
- `torch.device(...)`, `.to(...)`, `.cuda()`, and `.cpu()`;
- `torch.optim.lr_scheduler.*` configuration and scheduler steps;
- `torch.nn.utils.clip_grad_norm_` and `clip_grad_value_`;
- `torch.autocast`, `torch.amp.autocast`, and CUDA autocast;
- `GradScaler` configuration, scaling, stepping, and updating;
- common torchvision dataset constructors;
- `torch.utils.data.random_split(...)`;
- common torchvision preprocessing and stochastic augmentation operations;
- torchvision model constructors and PyTorch model-head replacement;
- explicit `requires_grad = False` parameter freezing;
- TorchMetrics configuration;
- argmax-based prediction selection and softmax normalization;
- checkpoint load/save and `load_state_dict(...)` operations;
- cross-cell optimizer and loss-constructor context.

The project deliberately prefers a small set of defensible rules over broad heuristics that could produce unsupported scientific claims.

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

Choose another renderer:

```python
%%scribe educational
model.eval()
```

## Architecture

```text
.ipynb / Notebook Cells
        │
        ▼
    Python AST
        │
        ▼
Import, Symbol & Cross-Cell Resolution
        │
        ▼
Framework Semantic Analyzers
        │
        ▼
Pattern Recognition
        │
        ▼
       SIR
Scientific Interpretation Representation
        │
        ├────────► Cell-level renderers
        │
        ├────────► Notebook Methods report
        │
        └────────► Structured experiment pipeline
                         │
                         ▼
                  Evidence provenance
```

The architecture is intentionally deterministic and inspectable.

## Open Source

ASTScribe is fully open source. The analysis engine, semantic rules, renderers, and notebook integration are available in this repository under the **Apache License 2.0**.

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

- Python AST parser;
- PyTorch semantic rules;
- training/inference pattern detection;
- scientific renderer;
- evidence tracking;
- direct `.ipynb` loading;
- forward-only cross-cell notebook context;
- optimizer/loss/scheduler configuration analysis;
- epochs, devices, gradient clipping, and AMP analysis;
- deterministic notebook-level Methods reports;
- evidence appendix with cell/line provenance;
- `%%scribe` IPython magic.

### v0.2

- torchvision dataset and model analysis;
- preprocessing and augmentation analysis;
- model-head replacement and parameter-freezing analysis;
- TorchMetrics and prediction-selection analysis;
- structured experiment-pipeline reconstruction;
- more granular notebook-level Methods sections;
- deeper reproducibility inspection;
- more optimizer, loss, and scheduler families.

### v0.3

- Hugging Face Transformers;
- richer Methods-section generation;
- notebook-level experiment summaries.

### Future

- scikit-learn;
- TensorFlow/Keras;
- notebook dependency graphs;
- richer Jupyter visualization;
- community semantic-rule packs.

## Citation

If ASTScribe contributes to academic or scientific work, please cite the software using [CITATION.cff](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Author

**Eduardo J. Barrios**  
GitHub: [@edujbarrios](https://github.com/edujbarrios)
