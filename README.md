# ASTScribe

> Evidence-backed explanations of unfamiliar ML notebooks, directly inside Jupyter.

[![PyPI](https://img.shields.io/pypi/v/astscribe?label=PyPI)](https://pypi.org/project/astscribe/)
[![Python](https://img.shields.io/pypi/pyversions/astscribe)](https://pypi.org/project/astscribe/)
[![CI](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/edujbarrios/astscribe/actions/workflows/tests.yml)
[![Try it in Colab](https://img.shields.io/badge/Try%20it%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/edujbarrios/astscribe/blob/main/examples/notebooks/quick_examples.ipynb)

**What does this notebook actually do — and what in the code supports that interpretation?**

A collaborator sends you an `.ipynb`. You discover a PyTorch experiment in an
open-source repository. Before trying to reproduce its results, you need to
understand its methods: **Which model is configured? Where are gradients
computed? What is evaluated, and how do the cells depend on one another?**

**ASTScribe turns supported source-level operations into explanations you can
inspect alongside the notebook itself.** It identifies training and inference
patterns, reconstructs an evidence-backed experiment outline, and connects
methodological claims to their **original cells and source lines**. Explore
individual cells using Jupyter magics or inspect an entire notebook from its
`.ipynb` file — without switching to a separate AI service.

**Why this approach?** Scientific interpretation requires traceability.
ASTScribe uses deterministic static analysis of recognized **PyTorch,
Transformers, Datasets and PEFT** constructs. It does **not** execute the
analyzed code, infer unobserved runtime behavior, or claim that an experiment
produced particular results. Unknown code and skipped cells are reported;
their possible effects are not silently treated as established facts.

**Try the workflow yourself:** [Open the runnable examples in Google Colab](https://colab.research.google.com/github/edujbarrios/astscribe/blob/main/examples/notebooks/quick_examples.ipynb)
and select **Runtime → Run all**. The notebook installs ASTScribe if needed.
The source snippets are analyzed as strings, so you can examine the
explanations **without downloading ML models, running training, using a GPU,
or supplying an LLM API key**.

## Install

```bash
python -m pip install astscribe
```

## First experiment: read an unfamiliar notebook

**Research question:** What methods and dataflow are *actually visible* in a
notebook you received from a colleague or found in an open-source project?

Download or clone `research.ipynb`, then ask ASTScribe for a **source-grounded
overview**: experiment stages, methodology, dependency diagnostics and skipped
cells. It does not execute any notebook cell.

```bash
astscribe research.ipynb --report overview --evidence
```

Or, **from inside Jupyter**, open the same file using
`NotebookAnalyzer.from_ipynb("research.ipynb")` and display
`notebook.render_overview(include_evidence=True)` as Markdown.
To inspect a particular source cell even when there are Markdown gaps, call
`notebook.explain_notebook_cell(7)` using its **original** `.ipynb` index.
For the current notebook's executed input, use `%scribe 7 scientific`.

**Interpretation boundary:** ASTScribe cannot establish model accuracy, numerical
results or hidden kernel state from static source. Unsupported code (including
IPython magics) forms a conservative context boundary: later explanations and
symbol dependencies do not reuse earlier bindings that the unknown code may
have modified. Unknown or malformed cells appear in the report.

## From source code to scientific explanation

Each example asks a concrete question and shows ASTScribe's **actual output**.
Copy any Python block into Jupyter, Colab or a script; the documented results
are checked automatically by tests. No ML framework needs to be installed to
analyze the strings below.

### 1. Explain ML code in plain language — right inside your notebook

**Research question:** Does this cell contain inference-specific execution semantics, and how can they be described without speculation?

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

**Interpretation:** Each statement is grounded in a recognized source operation.
This is a description of the code, not a claim about predictive performance.

### 2. Audit a training step and trace each claim to code

**Research question:** Which optimization steps are explicitly present, and what is the traceable source evidence?

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

**Research question:** Which stages of an unfamiliar Hugging Face experiment are supported by the source? No dataset or model is downloaded.

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

**Research question:** Are there source-order dependencies that may rely on a previous kernel session?

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

**Research question:** Which downstream cells are statically dependent on a changed preprocessing step?

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

## Scope and limitations

A scientific description should distinguish what the source **shows** from what
would require **running an experiment**. ASTScribe does not measure accuracy,
reproducibility or causal effects, and it cannot reconstruct hidden notebook
execution order. Diagnostics are prompts for review, not proof of runtime errors.

Supported static semantics include **PyTorch**, **TorchVision**, **TorchAudio**,
**Hugging Face Transformers**, **Hugging Face Datasets** and **PEFT**.
This includes common neural architectures, tensor and autograd APIs, image
transforms, vision datasets and operators, speech feature extraction, audio
datasets and VLM workflows; see the
[PyTorch ecosystem coverage guide](docs/pytorch_ecosystem_coverage.md),
[vision-language guide](docs/vision_language_models.md) and
[TorchAudio guide](docs/torchaudio.md).

**Important:** ASTScribe recognizes a broad, curated set of operations, *not
every symbol in these libraries*. It does not infer arbitrary dynamic values,
actual execution order, hidden kernel history, model accuracy, image contents
or audio contents. Unsupported APIs and dependency versions require review.

See [technical documentation](docs/), [contributing](CONTRIBUTING.md)
and [release notes](CHANGELOG.md). Apache-2.0 license: [LICENSE](LICENSE).
