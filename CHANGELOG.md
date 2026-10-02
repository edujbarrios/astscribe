# Changelog

All notable changes to ASTScribe will be documented in this file.

## [0.5.0] - Unreleased

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

## [0.4.0] - Unreleased

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
- Tests for import alias resolution, cross-cell Transformers context, generation, Trainer workflows, Methods reporting, and pipeline reconstruction.
- A documented Transformers evidence contract and explicit non-goals in `docs/transformers_semantics.md`.

### Changed

- ASTScribe now supports PyTorch and Hugging Face Transformers as first-class semantic frameworks.
- Transformers tokenization and processor operations share the existing structured preprocessing pipeline stage while receiving a dedicated Methods section, preserving the public pipeline title used by v0.2.

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
