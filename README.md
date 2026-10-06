# ASTScribe

> **Evidence-backed scientific explanations for ML notebooks, without LLMs.**

ASTScribe is an open-source, lightweight static semantic analysis library for Python machine-learning notebooks. It transforms ML code into traceable scientific explanations using Python's AST, framework-specific semantic rules, deterministic pattern recognition, and notebook-aware static context.

**No LLMs. No API keys. No code execution. No telemetry.**

ASTScribe is fully open source under the **Apache License 2.0** and is designed to remain independent of proprietary AI services.

## Why ASTScribe?

Machine-learning notebooks often contain the complete implementation of an experiment while leaving its methodology distributed across many cells. ASTScribe turns that implementation into a conservative scientific description without executing the notebook or sending source code to an external service.

The current first-class semantic frameworks are **PyTorch**, **Hugging Face Transformers**, **Hugging Face Datasets**, and **PEFT**.

ASTScribe deliberately favors a smaller number of auditable semantic rules over broad heuristics that could produce unsupported claims.

## Features

- Python AST-based static analysis.
- Direct `.ipynb` loading using only the Python standard library.
- Evidence-backed claims with source-line and notebook-cell traceability.
- Forward-only cross-cell context through `NotebookAnalyzer`.
- PyTorch-aware semantic rules for training, inference, data pipelines, and evaluation workflows.
- Hugging Face Transformers semantics for pretrained components, tokenization, Trainer workflows, generation, and quantization.
- Hugging Face Datasets semantics for loading, mapping, filtering, splitting, shuffling, selection, and schema transforms.
- PEFT semantics for LoRA-family configuration, adapter application, k-bit preparation, adapter checkpoints, and merging.
- Composite notebook-level technique detection for patterns that require evidence across cells, including conservative QLoRA detection.
- Static recovery of optimizer type and explicit hyperparameters.
- Detection of epochs, devices, DataLoaders, schedulers, gradient clipping, and AMP constructs.
- Torchvision dataset, transform, augmentation, and model semantics.
- Structured experiment-pipeline reconstruction.
- Scientific, educational, and concise cell-level rendering styles.
- Deterministic notebook-level `# Methods` reports.
- Optional evidence appendix linking scientific claims back to source cells and lines.
- Optional Jupyter/IPython `%%scribe` magic.
- Zero mandatory runtime dependencies outside the Python standard library.
- CI-enforced wheel size below 1,000,000 bytes.
- No telemetry or source-code upload.

## Installation

Install ASTScribe from PyPI:

```bash
python -m pip install astscribe
```

For optional IPython/Jupyter integration:

```bash
python -m pip install "astscribe[ipython]"
```

To work on the latest development version:

```bash
git clone https://github.com/edujbarrios/astscribe.git
cd astscribe
python -m pip install -e ".[dev]"
```

Verify the installation with:

```bash
astscribe --version
```

Release maintainers can follow the
[release checklist](https://github.com/edujbarrios/astscribe/blob/main/docs/releasing.md).

## Command Line

The installed package includes an `astscribe` command. Explain a Python file:

```bash
astscribe training.py --style concise
```

Render notebook-level reports without executing the notebook:

```bash
astscribe experiment.ipynb --report methodology --evidence
astscribe experiment.ipynb --report diagnostics
astscribe experiment.ipynb --report impact --cell 3
```

The same interface is available as `python -m astscribe`.

PyTorch, Transformers, Datasets, PEFT, and bitsandbytes are **not** mandatory ASTScribe dependencies. ASTScribe recognizes supported source-level APIs without importing or executing those frameworks.

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

## Modern Hugging Face Fine-Tuning Example

ASTScribe can connect dataset preparation, quantized loading, PEFT adaptation, and training across separate notebook cells without importing any of those libraries:

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer()

notebook.add_cell("""
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
dataset = dataset.map(tokenize, batched=True)
splits = dataset.train_test_split(test_size=0.1, seed=42)
""")

notebook.add_cell("""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "example/model",
    quantization_config=bnb,
)
""")

notebook.add_cell("""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora)
""")

notebook.add_cell("""
from transformers import Trainer

trainer = Trainer(model=model, train_dataset=splits["train"])
trainer.train()
""")

print(notebook.render_methodology(include_evidence=True))
print(notebook.render_pipeline())
print(notebook.render_techniques())
```

The reconstructed pipeline can contain:

```text
Dataset
    ↓
Dataset preparation
    ↓
Model architecture
    ↓
Quantization
    ↓
Parameter-efficient fine-tuning
    ↓
Training procedure
```

If the source contains explicit evidence for 4-bit quantized model loading, LoRA application, and training, `render_techniques()` can additionally report a QLoRA finding and list the rules/cells supporting it.

## Composite Technique Detection

Some methodological labels cannot be justified from a single AST node. ASTScribe keeps those separate from ordinary cell-level claims:

```python
findings = notebook.techniques()

for finding in findings:
    print(finding.key)
    print(finding.summary)
    for evidence in finding.evidence:
        print(evidence.cell, evidence.line_start, evidence.rule)
```

QLoRA is intentionally conservative. ASTScribe does **not** call a notebook QLoRA merely because it contains `prepare_model_for_kbit_training(...)`, a LoRA config, or a 4-bit config in isolation.

## Transformers Example

ASTScribe can reconstruct common Hugging Face inference workflows without importing Transformers:

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer()

notebook.add_cell("""
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
""")

notebook.add_cell("""
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
""")

print(notebook.render_methodology(include_evidence=True))
print(notebook.render_pipeline())
```

Because context flows forward across cells, ASTScribe can connect `model.generate(...)` and `tokenizer.decode(...)` back to the `from_pretrained(...)` constructors observed earlier in the notebook.

## Analyze a Real Jupyter Notebook

ASTScribe can load a `.ipynb` directly without depending on `nbformat`, Jupyter, or any supported ML framework:

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_ipynb("training.ipynb")

print(notebook.render_methodology())
print(notebook.render_pipeline())
print(notebook.render_techniques())
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

Fluent self-reassignment preserves constructor lineage when possible, so patterns such as `dataset = dataset.map(...)` do not automatically discard the known dataset origin.

## Notebook Methods Report

A notebook can be reduced to a deterministic paper-like Methods report:

```python
print(notebook.render_methodology(include_evidence=True))
```

The report can include sections such as:

- reproducibility;
- dataset configuration;
- dataset preparation;
- preprocessing and augmentation;
- tokenization and input preparation;
- data loading;
- execution environment;
- model architecture;
- quantization;
- parameter-efficient fine-tuning;
- model and objective;
- optimization;
- numerical precision;
- training procedure;
- evaluation and inference;
- metrics;
- checkpointing.

## Experiment Pipeline Reconstruction

ASTScribe exposes a structured experiment pipeline built from detected operations rather than generated prose:

```python
notebook = NotebookAnalyzer.from_ipynb("training.ipynb")

pipeline = notebook.pipeline()
print(pipeline.render())

for stage in pipeline.stages:
    print(stage.key, stage.operations)
```

Only stages supported by detected evidence are included. A missing stage means that ASTScribe did not find sufficient supported evidence; it does not prove that the experiment omitted that stage.

Transformers tokenization and processor operations retain the existing structured `preprocessing` stage for backwards compatibility, while the Methods report gives them the more precise `Tokenization and input preparation` section.

See the documentation for [experiment pipelines](https://github.com/edujbarrios/astscribe/blob/main/docs/experiment_pipeline.md), [Transformers semantics](https://github.com/edujbarrios/astscribe/blob/main/docs/transformers_semantics.md), [PEFT semantics](https://github.com/edujbarrios/astscribe/blob/main/docs/peft_semantics.md), and [Datasets and quantization](https://github.com/edujbarrios/astscribe/blob/main/docs/datasets_quantization.md) for the current evidence contracts and explicit non-goals.

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

Composite technique findings keep a tuple of their supporting evidence objects instead of pretending the conclusion came from one source line.

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
- gradient clipping;
- autocast and GradScaler workflows;
- common torchvision datasets, transforms, and models;
- model-head replacement and explicit parameter freezing;
- TorchMetrics configuration;
- prediction selection and softmax normalization;
- checkpoint operations;
- cross-cell optimizer and loss-constructor context.

## Current Transformers Coverage

The Transformers rule set currently recognizes:

- tokenizer, processor, feature-extractor, config, and model `from_pretrained(...)` calls;
- model classes whose names do not literally contain `Model`, including common `*ForConditionalGeneration`-style classes;
- cross-cell tokenizer and processor calls;
- supported model forward calls and explicit `labels=` supervision;
- `TrainingArguments` / `Seq2SeqTrainingArguments`;
- `Trainer` / `Seq2SeqTrainer` configuration, training, evaluation, and prediction;
- common data collators;
- `set_seed(...)`;
- inference pipelines;
- `generate(...)` and tokenizer decoding;
- `save_pretrained(...)`;
- `BitsAndBytesConfig(...)` and explicit 4-bit/8-bit quantized model loading.

## Current Datasets Coverage

The Datasets analyzer recognizes supported source-level uses of:

- `load_dataset(...)`;
- `load_from_disk(...)`;
- `map(...)`;
- `filter(...)`;
- `train_test_split(...)`;
- `shuffle(...)`;
- `select(...)`;
- common column/schema transforms.

ASTScribe does not download datasets, inspect examples, infer remote schemas, count records, or judge dataset quality.

## Current PEFT Coverage

The PEFT analyzer recognizes:

- `LoraConfig`, `AdaLoraConfig`, `IA3Config`, and other explicit `peft.*Config` constructors;
- statically resolvable adapter hyperparameters;
- `get_peft_model(...)`;
- `prepare_model_for_kbit_training(...)`;
- `PeftModel*.from_pretrained(...)`;
- adapter save/load/activation/addition operations;
- adapter state extraction;
- `merge_and_unload()`.

ASTScribe does not infer trainable-parameter counts, memory savings, convergence effects, model quality, or adapter correctness from these calls.

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
 ┌──────────┬──────────────┬──────────┬───────┐
 ▼          ▼              ▼          ▼
PyTorch  Transformers    Datasets    PEFT
 └──────────┴───────┬──────┴──────────┘
                    ▼
            Pattern Recognition
                    │
                    ▼
                   SIR
      Scientific Interpretation Representation
                    │
          ┌─────────┼──────────┬─────────────┐
          ▼         ▼          ▼             ▼
        Cell     Methods    Pipeline    Composite techniques
      renderers   report                 (e.g. QLoRA)
                    │
                    ▼
             Evidence provenance
```

The architecture is intentionally deterministic and inspectable.

## Open Source

ASTScribe is fully open source. The analysis engine, semantic rules, renderers, and notebook integration are available in this repository under the **Apache License 2.0**.

The project does not require proprietary APIs, external AI services, telemetry systems, or closed-source runtime components.

Community contributions are welcome through issues and pull requests. See the
[contribution guide](https://github.com/edujbarrios/astscribe/blob/main/CONTRIBUTING.md).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src/astscribe
```

The GitHub Actions workflow builds the wheel and fails if it reaches 1,000,000 bytes. This keeps the lightweight constraint measurable rather than aspirational.

## Roadmap

### v0.1

- AST parser, PyTorch semantics, evidence tracking, direct notebook loading, cross-cell context, Methods reporting, and IPython magic.

### v0.2

- torchvision experiment semantics, reproducibility analysis, and structured pipeline reconstruction.

### v0.3

- Hugging Face Transformers pretrained components, tokenization, Trainer workflows, generation, and decoding.

### v0.4

- PEFT/LoRA semantics, k-bit preparation, adapter lifecycle analysis, and parameter-efficient adaptation pipeline stage.

### v0.5

- Hugging Face Datasets semantics;
- bitsandbytes 4-bit/8-bit quantization analysis;
- broader Transformers/VLM model-class recognition;
- dataset-preparation and quantization pipeline stages;
- composite notebook techniques and conservative QLoRA detection.

### Future

- scikit-learn;
- TensorFlow/Keras;
- more quantization backends;
- notebook dependency graphs;
- richer Jupyter visualization;
- notebook-level experiment diagnostics;
- published semantic-rule extension API;
- community semantic-rule packs.

## Citation

If ASTScribe contributes to academic or scientific work, please cite the software using
the repository's [citation metadata](https://github.com/edujbarrios/astscribe/blob/main/CITATION.cff).

## License

Apache License 2.0. See the [license text](https://github.com/edujbarrios/astscribe/blob/main/LICENSE).

## Author

**Eduardo J. Barrios**  
GitHub: [@edujbarrios](https://github.com/edujbarrios)
