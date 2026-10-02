# Notebook diagnostics

ASTScribe can derive conservative notebook diagnostics from its static cell dependency graph.

The diagnostics are intended to expose structural risks that are common in exploratory notebooks without pretending to reconstruct historical kernel state or runtime execution.

## Public API

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_ipynb("experiment.ipynb")

diagnostics = notebook.diagnostics()
print(diagnostics.render())
print(diagnostics.to_dict())
```

Convenience rendering is also available:

```python
print(notebook.render_diagnostics())
```

Diagnostics can be selected by stable code:

```python
for item in diagnostics.by_code("dependency.forward_reference"):
    print(item.cell, item.symbol, item.related_cell)
```

## Diagnostic model

Each `NotebookDiagnostic` contains:

- a stable `code`;
- `severity` (`warning` or `info`);
- a conservative message;
- the notebook cell index;
- the symbol involved;
- an optional related cell.

Original `.ipynb` cell indices are preserved.

## `dependency.forward_reference`

A forward reference is reported when a symbol is read without a supported prior producer but ASTScribe can see a supported definition in a later notebook cell.

Example:

```python
# Cell 2
outputs = model(inputs)

# Cell 8
model = build_model()
```

The finding means that source-order static analysis cannot resolve `model` in cell 2 and that a later cell defines it.

It does **not** prove that the notebook failed at runtime. The kernel may already have contained `model` because cells were executed out of source order.

This distinction is useful precisely because notebook source order and kernel execution history can diverge.

## `dependency.unresolved_symbol`

An unresolved-symbol warning is reported when a non-builtin symbol is read without a supported prior definition and ASTScribe cannot find a supported later definition either.

Possible explanations include:

- hidden kernel state;
- environment injection;
- skipped notebook magic or shell syntax;
- dynamic execution;
- an unsupported Python construct;
- a genuine missing definition.

Therefore this diagnostic means **unresolved statically**, not `NameError` proven.

## `dependency.symbol_redefinition`

A redefinition is informational. It records that a later cell becomes the new static producer of a name previously produced by another cell.

Example:

```python
# Cell 4
model = baseline_model()

# Cell 11
model = tuned_model()
```

Redefinition is common and legitimate in exploratory notebooks. ASTScribe exposes it because repeated rebinding of names such as `model`, `dataset`, `optimizer`, or `config` can make experiments difficult to audit.

## `dependency.overwritten_before_cross_cell_use`

This informational diagnostic is emitted when a symbol definition is replaced by a later cell before any later analyzed cell consumes that specific definition.

The wording is intentionally narrow. ASTScribe does not call the definition dead code because:

- it may have been used inside its own cell;
- it may have produced side effects;
- it may have been inspected interactively;
- unsupported/dynamic code may have consumed it.

Read-before-write patterns such as:

```python
x = x + 1
```

count as consumption of the previous definition before the new definition replaces it.

## Severity policy

Warnings identify source-order dependencies that ASTScribe cannot resolve safely:

- forward references;
- unresolved symbols.

Informational findings describe potentially confusing but valid notebook structure:

- symbol redefinitions;
- definitions overwritten before cross-cell consumption.

No current diagnostic is classified as an error because ASTScribe does not observe runtime kernel state.

## Explicit non-goals

Notebook diagnostics do not prove:

- actual execution order;
- runtime success or failure;
- existence or absence of hidden kernel variables;
- dead code in the compiler sense;
- branch/path feasibility;
- object identity or mutation aliasing;
- scientific correctness of an experiment.

The diagnostic layer is a static notebook-audit aid, not a replacement for executing or testing the notebook.
