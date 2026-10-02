# Changelog

All notable changes to ASTScribe will be documented in this file.

## [0.1.0] - Unreleased

### Added

- Initial Python AST parsing and import alias resolution.
- Lightweight static symbol resolution for literal values and constructors.
- PyTorch semantic rules for training, inference, optimizers, losses, seeds, checkpoints, and DataLoaders.
- Scientific Interpretation Representation (SIR) with evidence-backed claims.
- Training and inference pattern detection.
- Concise, educational, and scientific renderers.
- Forward-only notebook context through `NotebookAnalyzer`.
- Cross-cell recovery of optimizer constructors and statically resolvable hyperparameters.
- Cell-aware evidence provenance for claims derived from earlier notebook cells.
- Deterministic notebook-level `# Methods` reports with optional evidence appendix.
- Optional `%%scribe` IPython cell magic.
- Static safety tests guaranteeing analyzed code is not executed.
- Apache License 2.0 licensing metadata and NOTICE file.
