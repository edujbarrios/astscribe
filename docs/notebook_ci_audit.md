# Static notebook CI audit

ASTScribe can combine conservative notebook source and saved-metadata findings in one deterministic report:

```sh
astscribe experiment.ipynb --report audit
astscribe experiment.ipynb --report audit --json
astscribe experiment.ipynb --report audit --json --fail-on-warning
```

The third command is suitable for CI: it always emits the report and exits with status **1** when the unified audit contains warnings, otherwise **0**. Without `--fail-on-warning`, the report is advisory and exits with status 0 for valid notebook files.

Supported findings include:

- `audit.skipped_code`: unsupported Python, notebook magics, or malformed code cells skipped by the analyzer; their effects on the kernel state are not known.
- Dependency diagnostics: unresolved/forward symbol reads and informational redefinitions.
- Execution metadata diagnostics: duplicate/out-of-order positive execution counts (warnings) and missing or invalid counters (informational).

The report also lists sorted **non-standard-library import roots** discovered by static AST analysis, but **does not flag external imports as warnings** because they may be local modules and are not automatically evidence of an unsafe environment. See [Import inventory](import_inventory.md).

The JSON contract reports `kind: notebook_static_audit`, `schema_version: 1`, `verification: source_and_saved_metadata_only`, `analyzed_cells`, `skipped_cells`, `warning_count`, sorted `findings`, and `non_stdlib_roots`. Each finding records the original notebook cell index.

The API is `NotebookAnalyzer.audit_report()`, returning a frozen `NotebookAuditReport`. ASTScribe never executes analyzed cells. An audit with zero warnings **does not certify reproducibility, installed dependencies, saved output correctness or clean-kernel execution**.

For backwards compatibility, `--fail-on-warning` with other reports retains its previous behavior: it gates on dependency warnings only. Combined audit warning gating is specific to `--report audit`.
