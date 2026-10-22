# Notebook execution metadata audit

Jupyter notebook files may preserve an `execution_count` for each code cell. ASTScribe now exposes this **saved metadata only** through a report, without running any notebook cells:

```sh
astscribe experiment.ipynb --report execution
astscribe experiment.ipynb --report execution --json
```

The report includes original cell indices, saved positive integer counts, and deterministic findings:

- `execution.nonmonotonic_count` (warning): saved counter decreases between code cells in source order.
- `execution.duplicate_count` (warning): two code cells share a positive saved counter.
- `execution.no_saved_count` (info): a cell lacks a positive integer counter (including `null`, invalid metadata and skipped Python cells).

The report works on an ASTScribe notebook analyzer created from `from_ipynb_data` or `from_ipynb`. Cells with Jupyter magics or unsupported Python syntax still contribute execution metadata even when excluded from semantic Python analysis.

**Important:** Execution counters can be edited, cleared or left stale by Jupyter clients. Neither an increasing sequence nor a warning-free report proves that a notebook was reproducibly executed from a clean kernel; notebook outputs, timestamps, dependencies and runtime state are not verified. Static source analysis continues to follow notebook source order, **not** the claimed count order.

The public API is `NotebookAnalyzer.execution_report()`, returning an immutable `NotebookExecutionReport` with `to_dict()` and `render()`. In-memory `from_cells` analyzers lack persisted execution counters and therefore return an empty metadata report.
