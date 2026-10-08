# Changelog

All notable changes to ASTScribe will be documented in this file.

Versions 0.1.0 through 0.7.0 were internal development milestones and were not
published to PyPI.

## [0.10.1] - 2026-10-08

### Fixed

- Notebooks containing unsupported code or IPython magics now invalidate stale imports, constants and constructor bindings before subsequent analysis. This prevents claims based on context that unknown code might have changed.
- Static symbol-flow edges no longer cross skipped/unsupported notebook cells, so impact analysis does not incorrectly assume a dependency survived a potentially state-changing cell.
- Non-string/unhashable notebook cell types are reported as skipped (or rejected in strict mode) instead of raising an uncaught `TypeError`.
- Regression tests cover context invalidation, safe reintroduction of imports, skipped-cell dependency barriers, trailing unsupported cells, and Markdown/empty cells that should preserve context.

### Changed

- The README now leads with the scientific question of understanding unfamiliar research notebooks, emphasizing traceable evidence and the boundary between supported static interpretation and measured outcomes.
- Package metadata describes the notebook understanding and audit use case more precisely.

## [0.10.0] - 2026-10-08

### Added

- A whole-notebook overview via `NotebookAnalyzer.render_overview(include_evidence=True)`, combining the experiment pipeline, supported methodology, dependency diagnostics, and skipped-cell details without executing the notebook.
- `astscribe project.ipynb --report overview` and `--report overview --json` for readable and machine-readable summaries of notebooks from colleagues or open-source repositories.
- `NotebookAnalyzer.analyze_notebook_cell(index)` and `explain_notebook_cell(index)` for addressing an original `.ipynb` cell even when Markdown, raw, or skipped cells occur between Python cells.

### Changed

- The README now leads with understanding **someone else's ML notebook without leaving Jupyter**, rather than describing one's own code, and documents the new overview and original-index lookup workflow.
- The permissive `.ipynb` reader now records malformed notebook cells as skipped; strict mode rejects them with clear errors.

### Fixed

- Malformed notebook entries (non-object cells, unknown cell types, or code cells with no `source`) were previously skipped without any diagnostic, potentially giving an incomplete audit while appearing successful.
- Looking up a real notebook cell by index was awkward with interleaved Markdown: the existing `analyze_cell()` API addresses analyzed-cell ordinal positions, whereas the new original-index APIs cannot silently select the wrong cell.

## [0.9.1] - 2026-10-08

### Added

- Three self-contained, executed example Jupyter notebooks with recorded outputs covering PyTorch inference/training semantics, notebook dependencies and impact, and static notebook auditing.
- A CI notebook-reproducibility job that re-executes example notebooks and checks stored outputs and execution counts.
- Additional regression tests for source-order cell indexing, augmented assignment, and CLI option validation.

### Changed

- Simplified the README from ten badges to three and documented the exact output alongside each runnable usage example.
- Reject CLI options that were silently ignored: non-notebook report selections, `--cell` without an impact report, and `--evidence` outside a methodology text report.

### Fixed

- Appending notebook cells after Markdown, empty code, or syntax-skipped cells now preserves original `.ipynb` coordinate positions even when the skipped content occurs at the end of the notebook.
- Explicit notebook cell indices may not reuse positions already occupied by earlier notebook cells, even if those cells were not analyzed.
- Augmented attribute and subscript assignments no longer produce duplicate static reads of their target base and index expressions.

## [0.9.0] - 2026-10-07

### Added

- A `%scribe N` IPython line magic that explains a specific `In[N]` entry using forward-only static context reconstructed from earlier Python inputs.
- Context-aware `%%scribe` analysis for the cell being written, while preserving the existing short style syntax such as `%%scribe concise`.
- A reusable `astscribe.ipython.explain_history_cell()` helper for deterministic history-cell explanations without executing notebook code.

### Changed

- IPython-only inputs such as magics and shell escapes now form conservative context boundaries so unknown runtime side effects cannot leak stale bindings into later explanations.
- The README examples are shorter and focus on Python, live notebook usage, notebook files, and the CLI.
- The README badge set has doubled from five to ten badges.

### Release

- Prepared ASTScribe 0.9.0 for production publication to PyPI and GitHub Releases.

## [0.8.8] - 2026-10-07

### Changed

- Notebook context propagation now treats module-scope rebindings conservatively, invalidating stale import aliases, constants, and constructor metadata when names are reassigned, deleted, shadowed by definitions, or made ambiguous by control flow.
- Persistent notebook imports are now tracked from module scope instead of leaking imports declared inside functions or classes into later cells.
- Strict notebook loading now rejects unsupported code-cell source representations instead of silently skipping them.

### Fixed

- Reassigning an import alias such as `torch as t` could leave the old framework mapping active in later cells and produce false PyTorch claims.
- Imports declared only inside a function could incorrectly leak into subsequent notebook cells.
- Deleted or rebound constructor variables could retain stale optimizer/loss metadata and cause later calls to be misclassified.
- Loop and other runtime bindings could leave stale literal constants available to later semantic analysis.
- Building a symbol table for a constructor assignment without passing the original source text could raise `IndexError` when the call appeared after the first line.

## [0.8.7] - 2026-10-07

### Added

- A `--fail-on-warning` CLI option for notebooks that preserves the requested report while returning exit status 1 when dependency diagnostics contain warnings, making ASTScribe easier to gate in CI.
- Static dependency support for Python structural pattern matching, including capture bindings, mapping/rest bindings, class patterns, OR patterns, guards, and value-pattern reads.

### Changed

- Dependency analysis now follows Python's assignment-expression scoping rules inside comprehensions, so `:=` targets correctly bind in the containing scope.
- Exception-handler aliases are now modeled as temporary bindings and cleared after the handler, matching Python's runtime lifetime rules.

### Fixed

- Notebook cells that consumed a name assigned with `:=` inside a comprehension could previously miss the producer edge.
- Pattern-captured names in `match/case` guards, bodies, and later cells could previously be reported as unresolved.

## [0.8.6] - 2026-10-06

### Added

- Structured CLI output with `--json` for Python analysis and every notebook report, including methodology, pipeline, techniques, dependencies, diagnostics, impact, and impact ranking.
- Deterministic JSON wrappers for notebook technique and ranking collections, making them easier to consume from CI, shell scripts, and tools such as `jq`.
- Regression coverage for JSON CLI output and Python's comprehension execution-scope behavior inside class bodies.

### Changed

- The README now documents JSON output examples for source analysis, diagnostics, and per-cell impact reports.
- Dependency analysis now models the outermost comprehension iterable separately from the nested comprehension scope.

### Fixed

- A comprehension inside a class could incorrectly treat a class-local name used by its first iterable as an outer notebook dependency.

## [0.8.5] - 2026-10-06

### Added

- Exact generated output beneath the Python Quick Start example so the library's behavior is visible directly from the README.
- A notebook dependency-graph Quick Start example with its deterministic output.
- A regression test that locks the documented scientific Quick Start output.

### Changed

- Refreshed the README badges with clearer PyPI, Python, CI, downloads, and license labels.
- Made the primary Quick Start source self-contained by including the PyTorch import.
- Improved the README flow so installation, generated output, notebook analysis, and CLI usage are visible earlier.

## [0.8.4] - 2026-10-06

### Added

- A `ranking` CLI report that renders notebook cells ordered by downstream static blast radius.
- `NotebookAnalyzer.render_impact_ranking()` for deterministic text output of impact rankings.
- Regression coverage for annotation-only assignments, class-local deletion, and class-comprehension scope behavior.

### Changed

- Class-scope tracking now models the fact that class namespaces are not enclosing lexical scopes for nested comprehensions or nested classes.
- The guarded release workflow now reacts to version metadata changes, so future releases do not require an unrelated edit to `release.yml`.

### Fixed

- Annotation-only statements such as `x: int` no longer create a false runtime producer for `x`.
- Deleting a class-local name now exposes an outer notebook producer to later reads in the same class body when appropriate.
- Comprehensions inside class bodies no longer incorrectly capture class-local names as lexical closures.

## [0.8.3] - 2026-10-06

### Added

- CLI Graphviz DOT export for notebook dependency graphs with `--report dependencies --dot`.
- Regression coverage for class-body dependency execution, UTF-8 BOM notebooks, malformed notebook source arrays, and DOT CLI output.

### Changed

- Class bodies now participate in notebook dependency analysis because their statements execute when the class is defined, while names bound inside the class remain class-local.
- Notebook files are opened with `utf-8-sig`, accepting both ordinary UTF-8 and UTF-8 files with a byte-order mark.

### Fixed

- Cross-cell dependencies referenced only from class bodies were previously missed.
- Imports and assignments inside a class body could be misclassified as notebook-global definitions once class-body analysis was added.
- Code-cell source arrays containing non-string values are no longer silently coerced into Python source.

## [0.8.2] - 2026-10-06

### Added

- A `--strict` CLI option for notebooks, allowing CI and validation workflows to fail when a code cell is not valid Python instead of silently skipping it.
- A PyPI downloads badge alongside the existing PyPI version and supported-Python badges.

### Changed

- Simplified the README further around installation, quick-start usage, notebook reports, CLI examples, and supported semantics.
- Made the release checklist version-agnostic and aligned it with the actual dynamic version source.
- Made the production release workflow self-contained: it can be run from Actions, derives the version, validates/builds first, creates the annotated tag, publishes to PyPI, and creates the GitHub Release.

### Fixed

- Removed the misleading release-checklist instruction to compare `pyproject.toml` with the package version even though the project declares its version dynamically.

## [0.8.1] - 2026-10-06

### Added

- Command-line support for reading Python source from standard input with `astscribe -`, making ASTScribe easier to use in pipes and CI workflows.

### Changed

- Simplified the README around installation, core usage, notebook analysis, CLI examples, and semantic scope.
- Added PyPI version and supported-Python badges to the README.

### Fixed

- Invalid notebook files whose JSON root is not an object now raise a clear validation error instead of leaking an `AttributeError` through the CLI.

## [0.8.0] - 2026-10-06

### Added

- Transitive notebook impact analysis, propagation paths, prerequisite discovery, and impact ranking.
- Command-line interface for Python explanations and notebook methodology, pipeline, technique, dependency, diagnostic, and impact reports.
- `py.typed` marker for typed-library consumers.
- Complete source distributions containing project documentation, examples, and community files.
- Trusted Publishing workflow for tagged PyPI releases, including distribution validation.
- Manual TestPyPI rehearsal workflow and a release checklist with exact publisher settings.
- Python 3.14 support in package metadata and the CI test matrix.

### Changed

- Package metadata now uses `astscribe.__version__` as its single version source.
- CI validates both wheel and source distributions with Twine before release.
- Distribution CI installs the built wheel and exercises its command-line entry point.
- Successful PyPI publication creates the matching GitHub Release automatically.

## [0.7.0] - 2026-10-02

### Added

- Dependency-aware notebook diagnostics built on the framework-independent cell dependency graph.
- `dependency.forward_reference` warnings for reads that cannot be resolved in source order but have a supported definition in a later cell.
- `dependency.unresolved_symbol` warnings for non-builtin reads with no supported notebook definition, explicitly allowing for hidden kernel state or external injection.
- `dependency.symbol_redefinition` informational findings for cross-cell rebinding of notebook-global names.
- `dependency.overwritten_before_cross_cell_use` informational findings when a definition is replaced before any later analyzed cell consumes that specific definition.
- Read-before-write handling so patterns such as `x = x + 1` count as consuming the previous definition before rebinding it.
- `NotebookAnalyzer.diagnostics()` and `NotebookAnalyzer.render_diagnostics()`.
- Structured `NotebookDiagnostic` / `NotebookDiagnostics` APIs, stable diagnostic codes, `by_code(...)`, and `to_dict()` export.
- Original `.ipynb` cell-index preservation in diagnostic locations and related-cell references.
- Dedicated diagnostic semantics and limitations in `docs/notebook_diagnostics.md`.
- Tests covering forward references, hidden/external state, redefinitions, conservative overwrite detection, deterministic rendering, and structured output.

### Changed

- ASTScribe can now distinguish a likely source-order notebook dependency from a symbol that remains unresolved across the analyzed notebook, rather than collapsing both cases into the same static finding.
- Dependency diagnostics use warnings only for unresolved source-order structure and informational severity for potentially confusing but valid notebook rebinding patterns.

## [0.6.0] - 2026-10-02

### Added

- Framework-agnostic notebook cell dependency graph based on ordered Python symbol reads, writes, and deletions.
- Static resolution of cross-cell symbol reads to the latest previously observed producer.
- Aggregated producer-cell to consumer-cell edges with the symbols responsible for each dependency.
- Per-cell unresolved-read reporting for non-builtin symbols with no supported prior producer.
- Cross-cell symbol-redefinition tracking for names repeatedly rebound during notebook experimentation.
- `NotebookAnalyzer.dependency_graph()`, `render_dependency_graph()`, and `dependency_dot()`.
- `NotebookDependencyGraph.parents(...)`, `children(...)`, structured `to_dict()`, deterministic text rendering, and Graphviz DOT source export.
- Original `.ipynb` cell-index preservation in graph nodes and edges.
- Conservative scope handling for imports, assignments, augmented assignments, loops, context-manager bindings, comprehensions, function defaults, classes, and deletion.
- Dedicated dependency-graph semantics and limitations in `docs/cell_dependency_graph.md`.
- Tests covering read-before-write reassignment, producer replacement, comprehension scope, function-definition behavior, deletion, unresolved reads, original notebook indices, and DOT export.

### Changed

- ASTScribe now includes a framework-independent notebook structural/dataflow layer in addition to framework semantic analysis, experiment reconstruction, and composite-technique detection.
- Dependency analysis is flow-sensitive to supported syntactic event order while remaining intentionally non-branch-sensitive and independent of historical Jupyter kernel execution state.

## [0.5.0] - 2026-10-02

### Added

- First Hugging Face Datasets semantic analyzer with no runtime `datasets` dependency.
- Static `load_dataset(...)` and `load_from_disk(...)` analysis with dataset identifier, split, streaming, and other statically resolvable options.
- Dataset `map`, `filter`, `train_test_split`, `shuffle`, `select`, and common schema-transformation semantics.
- Dataset lineage resolution across notebook cells.
- Dedicated `Dataset preparation` Methods and experiment-pipeline stage.
- First bitsandbytes-oriented Transformers quantization analyzer with no runtime `bitsandbytes` dependency.
- `BitsAndBytesConfig(...)` analysis for explicit 4-bit/8-bit settings, NF4/FP4-related configuration, compute dtype expressions, double quantization, and int8 threshold when statically available.
- Cross-cell linking of named `BitsAndBytesConfig` instances to model `from_pretrained(..., quantization_config=...)` calls.
- Dedicated `Quantization` Methods and experiment-pipeline stage.
- `NotebookAnalyzer.techniques()` and `NotebookAnalyzer.render_techniques()` for techniques requiring evidence across multiple cells.
- Conservative composite QLoRA detection requiring explicit 4-bit model loading, LoRA application, and an invoked training procedure.
- Broader Transformers model-class coverage for classes such as VLM/generative implementations whose names do not literally contain `Model`.
- Dataset/quantization/QLoRA evidence contract in `docs/datasets_quantization.md`.
- Tests for dataset lineage, fluent reassignment, quantized model loading, QLoRA positive/negative cases, VLM model classes, Methods output, and pipeline reconstruction.

### Changed

- Fluent self-reassignment such as `dataset = dataset.map(...)` now preserves constructor lineage, improving cross-cell semantic context without framework-specific parser logic.
- ASTScribe now supports PyTorch, Hugging Face Transformers, Hugging Face Datasets, and PEFT as first-class semantic frameworks.
- The experiment pipeline can distinguish dataset preparation, model architecture, quantization, and parameter-efficient adaptation as separate evidence-backed stages.

## [0.4.0] - 2026-10-02

### Added

- First PEFT semantic analyzer with no runtime `peft` dependency.
- Static analysis for `peft.*Config` adapter configurations, including LoRA, AdaLoRA, and IA3 family labeling when explicit from the config class.
- Recovery of statically resolvable LoRA settings such as rank, alpha, dropout, bias, literal target modules, and PEFT task type expressions.
- `get_peft_model(...)` adapter-application analysis with forward-only cross-cell recovery of adapter configuration.
- `prepare_model_for_kbit_training(...)` semantics without automatically claiming QLoRA.
- `PeftModel*.from_pretrained(...)` adapter checkpoint loading with explicit adapter identifier and trainability settings when statically resolvable.
- PEFT adapter save, load, activation, addition, merge, and state-dictionary extraction semantics.
- Dedicated `Parameter-efficient fine-tuning` notebook Methods section.
- Optional `adaptation` experiment-pipeline stage between model architecture and objective/optimization.
- PEFT evidence contract and explicit non-goals in `docs/peft_semantics.md`.
- Tests for LoRA hyperparameters, cross-cell adapter context, adapter checkpoint operations, k-bit preparation, and pipeline reconstruction.

### Changed

- ASTScribe now supports PyTorch, Hugging Face Transformers, and PEFT as first-class semantic frameworks.
- Experiment pipelines can distinguish base-model construction from parameter-efficient adaptation without changing pipelines that contain no PEFT evidence.

## [0.3.0] - 2026-10-02

### Added

- First Hugging Face Transformers semantic analyzer with no runtime Transformers dependency.
- `AutoTokenizer*`, processor, image-processor, feature-extractor, config, and `AutoModel*` `from_pretrained(...)` analysis.
- Cross-cell tokenizer and processor invocation semantics.
- Transformers model forward-pass analysis with explicit `labels=` supervision evidence.
- `TrainingArguments` and `Seq2SeqTrainingArguments` static hyperparameter recovery.
- `Trainer` and `Seq2SeqTrainer` configuration, training, evaluation, and prediction semantics.
- Common `DataCollator*` configuration analysis.
- `transformers.set_seed(...)` reproducibility analysis.
- `transformers.pipeline(...)` inference-pipeline analysis.
- `generate(...)` analysis for statically resolvable generation parameters including token limits, beam search, and sampling controls.
- Tokenizer `decode(...)` and `batch_decode(...)` output-decoding analysis.
- Transformers `save_pretrained(...)` checkpoint/export semantics.
- Dedicated notebook Methods section for tokenization and input preparation.
- Transformers-aware structured experiment-pipeline reconstruction.
- Tests for import alias resolution, cross-cell Transformers context, generation, Trainer workflows, Methods reporting, and pipeline reconstruction.
- A documented Transformers evidence contract and explicit non-goals in `docs/transformers_semantics.md`.

### Changed

- ASTScribe now supports PyTorch and Hugging Face Transformers as first-class semantic frameworks.
- Transformers tokenization and processor operations share the existing structured preprocessing pipeline stage while receiving a dedicated Methods section, preserving the public pipeline title used by v0.2.

## [0.2.0] - 2026-10-02

### Added

- Experiment-level semantic analysis for torchvision datasets and model constructors.
- Static detection of `torch.utils.data.random_split(...)` dataset partitions.
- Preprocessing and data-augmentation analysis for common torchvision transforms.
- Scientific descriptions for composed preprocessing pipelines without claiming unsupported performance effects.
- Model architecture reconstruction for torchvision models and PyTorch head replacement.
- Explicit parameter-freezing analysis through `requires_grad = False`.
- TorchMetrics configuration analysis.
- Argmax prediction-selection and softmax-normalization analysis.
- `load_state_dict(...)` checkpoint restoration semantics.
- Reproducibility analysis for CUDA seeds, deterministic algorithms, and cuDNN deterministic/benchmark flags.
- Structured `ExperimentPipeline` / `PipelineStage` API.
- `NotebookAnalyzer.pipeline()` and `NotebookAnalyzer.render_pipeline()`.
- More granular Methods sections for dataset, preprocessing, model architecture, objective, evaluation, metrics, and reproducibility.
- GitHub Actions quality gates for Ruff and strict mypy checks.
- A CI-enforced wheel-size limit of less than 1,000,000 bytes.

### Changed

- The semantic registry now supports multiple analyzers per framework, allowing experiment-level rules to evolve independently from the core PyTorch analyzer.
- Notebook methodology reports now follow the order of a typical ML experiment more closely.
- Static analyzer typing is validated in CI while keeping optional IPython typing exceptions narrowly scoped.

## [0.1.0] - 2026-10-02

### Added

- Initial Python AST parsing and import alias resolution.
- Lightweight static symbol resolution for literal values and constructors.
- Scientific Interpretation Representation (SIR) with evidence-backed claims.
- Concise, educational, and scientific renderers.
- Forward-only notebook context through `NotebookAnalyzer`.
- Direct `.ipynb` loading using the Python standard library.
- Preservation of original notebook cell indices in evidence provenance.
- Explicit tracking of skipped non-Python/IPython-specific cells.
- Cross-cell recovery of optimizer constructors and statically resolvable hyperparameters.
- Cell-aware evidence provenance for claims derived from earlier notebook cells.
- PyTorch semantic rules for training, inference, optimizers, losses, seeds, checkpoints, and DataLoaders.
- Static epoch-count extraction for common `range(...)` training loops.
- PyTorch device configuration and transfer analysis.
- Learning-rate scheduler configuration and scheduler-step analysis.
- Gradient clipping analysis.
- Automatic mixed-precision and GradScaler analysis.
- Deterministic notebook-level `# Methods` reports with optional evidence appendix.
- Optional `%%scribe` IPython cell magic.
- Static safety tests guaranteeing analyzed code is not executed.
- Apache License 2.0 licensing metadata and NOTICE file.

### Changed

- Optimizer-step detection now distinguishes optimizer, scheduler, and GradScaler `.step()` calls to reduce false methodological claims.
