# Notebook import inventory

ASTScribe can inventory Python import statements from supported code cells and record **original notebook cell indices and source lines** without installing, importing, or running code.

```sh
astscribe experiment.ipynb --report imports
astscribe experiment.ipynb --report imports --json
```

The public API is `NotebookAnalyzer.import_report()`, returning a structured `NotebookImportReport` with immutable `ImportEvidence` records and `to_dict()` / `render()` views.

The JSON envelope includes `kind: notebook_import_inventory`, `schema_version: 1`, `verification: source_only`, `imports` and sorted `non_stdlib_roots`. Each import records `cell`, `line`, `module`, `root`, `kind`, optional `distribution_hint`, and whether the import occurs inside a nested or conditional statement.

Classification is intentionally conservative:

- `stdlib`: imports whose root is in Python's own standard-library list.
- `relative`: explicit relative imports such as `from . import helper`.
- `non_stdlib`: unknown import roots; these may be external packages **or local modules**. No network or package index lookup occurs.

Distribution names are **hints**, not verified project requirements: `sklearn` → `scikit-learn`, `cv2` → `opencv-python`, `PIL` → `Pillow`, and `yaml` → `PyYAML`. ASTScribe cannot determine exact installed versions, environment markers, local import paths, dynamic `importlib` calls, or whether an import is ever executed.

Skipped/unsupported notebook code cells do not contribute import claims. Nested/conditional imports are visible but should not be interpreted as unconditional dependencies. This source-only report is a useful starting point for reproducibility review—not an automatically validated lockfile.
