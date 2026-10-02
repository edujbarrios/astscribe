# Changelog

All notable changes to ASTScribe will be documented in this file.

## [0.1.0] - Unreleased

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
