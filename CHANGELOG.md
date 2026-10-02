# Changelog

All notable changes to ASTScribe will be documented in this file.

## [0.3.0] - Unreleased

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
- Tests for alias resolution, cross-cell Transformers context, generation, Trainer workflows, Methods reporting, and pipeline reconstruction.

### Changed

- ASTScribe now supports PyTorch and Hugging Face Transformers as first-class semantic frameworks.
- The experiment pipeline uses the broader `Preprocessing and input preparation` stage title so vision transforms and tokenization can share a structured stage without conflating their scientific descriptions.

## [0.2.0] - Unreleased

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
