# Datasets, quantization, and composite-technique semantics

ASTScribe analyzes Hugging Face Datasets and quantized Transformers workflows without importing, executing, or contacting those libraries or services.

This document defines the evidence contract for the v0.5 rule set.

## Hugging Face Datasets

Supported source-level signals include:

- `datasets.load_dataset(...)`;
- `datasets.load_from_disk(...)`;
- `Dataset.map(...)`;
- `Dataset.filter(...)`;
- `Dataset.train_test_split(...)`;
- `Dataset.shuffle(...)`;
- `Dataset.select(...)`;
- common column/schema transformations such as `remove_columns(...)`, `select_columns(...)`, and rename operations.

ASTScribe may report literal arguments such as dataset identifiers, splits, `streaming`, `test_size`, seeds, `batched`, and `num_proc` when those values can be resolved statically.

It does not download datasets, inspect rows, infer schemas from remote repositories, count examples, validate labels, assess class balance, or judge dataset suitability.

## Bitsandbytes quantization

ASTScribe recognizes `transformers.BitsAndBytesConfig(...)` and links a named configuration to a later model load when it is passed through `quantization_config=`.

Supported explicit settings include:

- `load_in_4bit`;
- `load_in_8bit`;
- `bnb_4bit_quant_type`;
- `bnb_4bit_compute_dtype` when statically represented;
- `bnb_4bit_use_double_quant`;
- `llm_int8_threshold`.

A model is described as loaded with 4-bit or 8-bit bitsandbytes quantization only when that bit-width is explicit in the source-level configuration attached to the load operation.

ASTScribe does not measure memory usage, inspect runtime module replacement, verify hardware/backend support, or claim accuracy/speed effects from quantization.

## Composite QLoRA finding

QLoRA cannot be established from one AST node. ASTScribe therefore treats it as a notebook-level composite technique.

`NotebookAnalyzer.techniques()` reports `qlora` only when all of the following are observed:

1. a Transformers model load with explicit 4-bit bitsandbytes quantization;
2. LoRA applied to a model through PEFT; and
3. an invoked supported training procedure, such as `Trainer.train()` or a detected optimizer parameter update.

`prepare_model_for_kbit_training(...)` is included as additional supporting evidence when present, but it is neither necessary nor sufficient by itself to label a notebook as QLoRA.

This distinction intentionally avoids the unsafe shortcut:

```text
prepare_model_for_kbit_training(...) => QLoRA
```

which is not enough evidence.

## Evidence provenance

Composite findings retain the individual evidence objects that established them. The rendered technique report points back to source cells, lines, and rule identifiers rather than generating an unsupported narrative.

## Explicit non-goals

ASTScribe does not infer:

- hidden framework defaults;
- remote model or dataset contents;
- actual tensor dtypes at runtime unless directly established by supported source evidence;
- whether a quantized kernel was successfully loaded;
- exact trainable-parameter counts;
- memory savings or speedups;
- model quality, convergence, or scientific validity;
- QLoRA merely from the presence of 4-bit quantization;
- QLoRA merely from the presence of a LoRA configuration;
- QLoRA merely from k-bit preparation.

The guiding rule is unchanged: prefer a missing claim over an unsupported one.
