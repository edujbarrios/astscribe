# Experiment pipeline reconstruction

ASTScribe can reconstruct a coarse, evidence-backed machine-learning experiment pipeline from a notebook without executing the notebook and without using an LLM.

The reconstruction is built from the same structured operations used by the scientific Methods renderer.

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_ipynb("training.ipynb")

print(notebook.render_pipeline())
```

Example output:

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

## Structured API

The pipeline is available as data rather than only rendered text:

```python
pipeline = notebook.pipeline()

for stage in pipeline.stages:
    print(stage.key)
    print(stage.title)
    print(stage.operations)
```

It can also be converted into JSON-serializable data:

```python
payload = notebook.pipeline().to_dict()
```

This makes pipeline reconstruction useful for future Jupyter interfaces, documentation generators, editor integrations, and scientific auditing tools.

## Current experiment semantics

The v0.2 analysis layer recognizes a focused set of constructs:

- common `torchvision.datasets` dataset constructors;
- `torch.utils.data.random_split(...)`;
- common `torchvision.transforms` preprocessing operations;
- composed transform pipelines;
- stochastic augmentation operations such as random crops and flips;
- torchvision model constructors;
- replacement of PyTorch model heads with `torch.nn` modules;
- explicit `requires_grad = False` parameter freezing;
- TorchMetrics constructors;
- argmax-based prediction selection;
- softmax normalization;
- `load_state_dict(...)` parameter restoration.

These rules complement the lower-level PyTorch analysis for optimizers, losses, schedulers, AMP, devices, DataLoaders, training loops, inference, and checkpoint serialization.

## Conservative interpretation

ASTScribe describes what can be supported by static evidence.

For example, given:

```python
transforms.RandomHorizontalFlip()
```

ASTScribe may state that random horizontal flipping is included as a stochastic data-augmentation operation.

It does **not** claim that the transform improves generalization, prevents overfitting, or increases accuracy, because those effects are not established by the source code.

Likewise, given:

```python
for parameter in model.parameters():
    parameter.requires_grad = False
```

ASTScribe reports that gradient computation is disabled for those parameters. It does not invent a fine-tuning strategy or motivation that is not encoded in the notebook.

## Pipeline stages

The structured pipeline currently uses the following ordered stages:

1. Dataset
2. Preprocessing and augmentation
3. Data loading
4. Model architecture
5. Objective
6. Optimization
7. Numerical precision
8. Training procedure
9. Evaluation and inference
10. Metrics
11. Checkpointing

Only stages backed by detected operations are included.

A missing stage therefore means that ASTScribe did not find sufficient supported evidence for that stage; it does not prove that the underlying experiment omitted it.
