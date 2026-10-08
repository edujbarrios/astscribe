# ASTScribe

> Understand someone else's ML notebook without leaving Jupyter — evidence-backed, no LLM required.

[![PyPI](https://img.shields.io/pypi/v/astscribe?label=PyPI)](https://pypi.org/project/astscribe/)
[![Python](https://img.shields.io/pypi/pyversions/astscribe)](https://pypi.org/project/astscribe/)
[![CI](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml)
[![Try it in Colab](https://img.shields.io/badge/Try%20it%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/edujbarrios/astscribe/blob/main/examples/notebooks/quick_examples.ipynb)

**Understand what a notebook does without leaving that notebook.**
Someone shared a research notebook with you? Exploring an open-source project
built with PyTorch, Transformers or other ML frameworks? ASTScribe explains the
supported operations **right where you're reading them**, with source-linked
evidence instead of guesses. Understand inference, training and the overall
experiment before you try running unfamiliar code. You can also inspect cell
dependencies, spot out-of-order definitions and see what edits may affect.

Use ASTScribe in Jupyter to explain individual cells or inspect the entire
`.ipynb` with a single notebook overview. **No LLM or API key required.**

**Start hands-on:** [Run the interactive quick examples in Google Colab](https://colab.research.google.com/github/edujbarrios/astscribe/blob/main/examples/notebooks/quick_examples.ipynb).
Open the notebook, choose **Runtime → Run all**, and explore each explanation
alongside its source. It installs ASTScribe automatically if needed.

**Static analysis, not model execution.** ASTScribe never executes the ML code
it explains: the walkthroughs below use Python source **strings**, so you don't
need PyTorch, Transformers, datasets, a GPU, downloads or an API key just to
try them.

## Install

```bash
python -m pip install astscribe
```

## Understand a notebook someone sent you

Download or clone the project notebook (for example, `research.ipynb`). Then
get a **whole-notebook overview**: detected experiment stages, evidence-backed
methodology, dependency diagnostics and a list of cells ASTScribe had to skip.

```bash
astscribe research.ipynb --report overview --evidence
```

Or, **from inside Jupyter**, open the same file using
`NotebookAnalyzer.from_ipynb("research.ipynb")` and display
`notebook.render_overview(include_evidence=True)` as Markdown.
To inspect a particular source cell even when there are Markdown gaps, call
`notebook.explain_notebook_cell(7)` using its **original** `.ipynb` index.
For the current notebook's executed input, use `%scribe 7 scientific`.

ASTScribe **only reads and analyzes** the source: it will not run a stranger's
training code, fetch model weights or execute notebook magics. It reports
unrecognized/malformed cells instead of silently pretending they were analyzed.

## See what ASTScribe produces

**Start with the main feature: understand an unfamiliar notebook.**
Each example is independent; you can copy its Python block into a script,
Jupyter notebook or Colab. The expected output underneath is tested in CI.

### 1. Explain ML code in plain language — right inside your notebook

**Question:** Can I explain the inference procedure in a review or Methods section without guessing?

```python
from astscribe import explain

source = """import torch
model.eval()
with torch.no_grad():
    outputs = model(inputs)
"""
print(explain(source, style="scientific"))
```

**Output**

```text
Inference procedure

Gradient tracking is disabled for the enclosed operations, so no autograd graph is constructed for computations executed within this context.

The model is explicitly configured in evaluation mode.

A forward pass is performed by invoking the model on the supplied inputs.
```

**Why it matters:** Explanations are based on supported source evidence, not claims about model accuracy or performance.
### 2. Audit a training step and trace each claim to code

**Question:** Does this snippet actually do backpropagation and an optimizer update? Where?

```python
from astscribe import analyze, explain

source = """import torch
torch.manual_seed(42)
model.train()
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
"""
result = analyze(source)

print(explain(source, style="concise"))
print("Backward pass:", result.training_step.backward_pass)
print("Parameter update:", result.training_step.parameter_update)
for claim in result.claims:
    if claim.rule == "pytorch.optimizer_step":
        print(f"Evidence: line {claim.line_start} ({claim.rule})")
```

**Output**

```text
Performs a PyTorch gradient-based training step with backward propagation and a parameter update.
Backward pass: True
Parameter update: True
Evidence: line 8 (pytorch.optimizer_step)
```

**Why it matters:** The result is structured (`result.training_step`), while individual claims trace back to source lines. It describes *detected code*, not whether training converged.

### 3. Reconstruct an unfamiliar Hugging Face experiment

**Question:** Which stages appear in this multi-cell ML workflow? Imports and constructors are represented as strings, so nothing is downloaded or trained.

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_cells([
    """from datasets import load_dataset
train_data = load_dataset("imdb", split="train")""",
    """from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")""",
    """from transformers import Trainer, TrainingArguments
args = TrainingArguments(output_dir="runs", num_train_epochs=2)
trainer = Trainer(model=model, args=args, train_dataset=train_data)
trainer.train()""",
])
print(notebook.render_pipeline())
```

**Output**

```text
Dataset
    ↓
Model architecture
    ↓
Training procedure
```

**Why it matters:** You can understand the workflow before running it. Only stages supported by ASTScribe's static rules appear; a missing stage is not proof that it never happens.

### 4. Detect cells that rely on later definitions

**Question:** Could a notebook work only because someone ran its cells out of order?

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_cells([
    "features = preprocess(raw_data)",   # Cell 0 reads raw_data too early
    "raw_data = load_data()",             # Cell 1 defines it later
    "predictions = predict(features)",
])
for issue in notebook.diagnostics().by_code("dependency.forward_reference"):
    print(f"Cell {issue.cell}: {issue.symbol} defined later in cell {issue.related_cell}")
```

**Output**

```text
Cell 0: raw_data defined later in cell 1
```

**Why it matters:** The diagnostic points to a source-order dependency. Other undefined symbols are separately reported as potentially external or hidden kernel state.

### 5. Find which results depend on a changed preprocessing cell

**Question:** If you edit cell 1, which downstream results might be stale?

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_cells([
    "raw = load_data()",                    # Cell 0
    "cleaned = normalize(raw)",             # Cell 1: changed preprocessing
    "features = make_features(cleaned)",    # Cell 2
    "model = fit(features)",                 # Cell 3
    "score = evaluate(model)",               # Cell 4
])
report = notebook.impact(1)
print("Revisit cells:", report.affected_cells)
print("Cells needed first:", report.required_ancestors)
print("Blast radius:", report.blast_radius)
```

**Output**

```text
Revisit cells: (2, 3, 4)
Cells needed first: (0,)
Blast radius: 3
```

**Why it matters:** ASTScribe follows static symbol dependencies, so it can estimate which cells need review without executing their code. This is a conservative impact estimate, not a Jupyter execution scheduler.


## Use it on your own notebook

Pass your existing `.ipynb` file to the analyzer. Markdown and unsupported IPython
syntax are skipped conservatively, while original notebook cell indices are preserved.

- `NotebookAnalyzer.from_ipynb('experiment.ipynb').render_methodology(include_evidence=True)` — draft an evidence-linked Methods report.
- `NotebookAnalyzer.from_ipynb('experiment.ipynb').render_diagnostics()` — flag forward references and unresolved names.
- `NotebookAnalyzer.from_ipynb('experiment.ipynb').render_impact(3)` — see what depends on original notebook cell 3.

Or use the CLI on a Python snippet:

```bash
printf 'import torch\nmodel.eval()\nwith torch.no_grad():\n    outputs = model(inputs)\n' | astscribe - --style concise
```

**Output**

```text
Performs PyTorch inference using evaluation-oriented execution semantics.
```

For real notebook files, you can also run `astscribe experiment.ipynb --report pipeline`,
`--report diagnostics --json`, or `--report dependencies --dot`.
Use `--strict` to reject unsupported cells and `--fail-on-warning` to return
a non-zero exit code when dependency warnings are detected.

## Explain a cell without leaving Jupyter

Install `astscribe[ipython]`, then load `%load_ext astscribe.ipython`.
Run `%scribe 4 concise` to explain an earlier `In[4]` cell, or start a cell
with `%%scribe concise` to **explain its contents without executing them**.
Both display the explanation in the notebook itself; no switching to a
separate app or external service.

[Try these features in the quick-start Colab notebook](https://colab.research.google.com/github/edujbarrios/astscribe/blob/main/examples/notebooks/quick_examples.ipynb).

## Explore executed notebooks

These Jupyter notebooks include saved outputs you can inspect on GitHub:

- [Quick examples (open in Colab)](examples/notebooks/quick_examples.ipynb) — hands-on start with code explanations, training evidence, diagnostics and notebook magic.
- [Inference and training](examples/notebooks/inference_and_training.ipynb) — inspect detected ML operations.
- [Dependencies and impact](examples/notebooks/dependencies_and_impact.ipynb) — symbol flow and blast radius.
- [Notebook audit](examples/notebooks/notebook_audit.ipynb) — source-order warnings and skipped cells.

To re-execute and verify that their stored outputs are still correct:

```bash
python -m pip install nbclient nbformat ipykernel
python scripts/verify_notebooks.py
```

Expected final line:

```text
Verified 4 executed example notebooks; all stored outputs match.
```

## Scope and limits

Supported static semantics include **PyTorch**, **Hugging Face Transformers**,
**Hugging Face Datasets**, and **PEFT**, as well as composite techniques such
as QLoRA. ASTScribe cannot infer arbitrary dynamic values, actual execution
order, hidden kernel history, or performance results. A diagnostic is a
reason to inspect your code, not proof of a runtime error.

See [technical documentation](docs/), [contributing](CONTRIBUTING.md)
and [release notes](CHANGELOG.md). Apache-2.0 license: [LICENSE](LICENSE).
