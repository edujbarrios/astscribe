# Hugging Face Transformers semantics

ASTScribe analyzes Hugging Face Transformers source code without importing or executing the `transformers` package.

The analyzer is intentionally conservative. Every scientific statement must be traceable to source syntax, statically resolved notebook context, or a narrow documented API semantic.

## Supported evidence

### Pretrained components

ASTScribe recognizes supported `transformers.*.from_pretrained(...)` calls and records statically resolvable values such as the model identifier and explicit keyword arguments.

Examples include:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

The analyzer can carry these constructor bindings forward to later notebook cells.

### Tokenization and input preparation

Calls through a statically known tokenizer or processor can report explicit options such as:

- `padding`;
- `truncation`;
- `max_length`;
- `return_tensors`;
- `add_special_tokens`.

ASTScribe does not infer tokenizer defaults that are absent from source.

### Model calls and supervision

A call through a statically known Transformers model is represented as a model forward pass.

If the source explicitly includes `labels=...`, ASTScribe records that target labels were supplied. It does not claim a particular loss formulation unless that formulation is independently supported by source evidence.

### Trainer workflows

The analyzer recognizes:

- `TrainingArguments`;
- `Seq2SeqTrainingArguments`;
- `Trainer`;
- `Seq2SeqTrainer`;
- `trainer.train()`;
- `trainer.evaluate()`;
- `trainer.predict()`.

Only statically resolvable argument values are reported. The presence of `train_dataset=` or `eval_dataset=` is recorded without inspecting dataset contents.

### Generation

For `model.generate(...)`, ASTScribe can report explicit generation controls including:

- `max_new_tokens`;
- `max_length`;
- `num_beams`;
- `do_sample`;
- `temperature`;
- `top_p`;
- `top_k`.

No hidden generation defaults are reconstructed.

### Output decoding

Calls to `decode(...)` and `batch_decode(...)` on a statically known tokenizer are represented as conversion of generated token IDs back into text.

### Reproducibility and persistence

The analyzer recognizes `transformers.set_seed(...)` and `save_pretrained(...)` when their subjects can be identified statically.

## Evidence boundaries

ASTScribe does **not** infer:

- whether a remote model identifier exists;
- which files were actually downloaded;
- model quality or suitability;
- dataset quality or label correctness;
- tokenizer or model defaults omitted from source;
- behavior introduced by `trust_remote_code=True`;
- runtime tensor shapes or values;
- GPU/CPU placement unless separately explicit in supported source;
- whether training converged;
- whether evaluation metrics are scientifically appropriate;
- scientific conclusions from observed code.

## Cross-cell resolution

Notebook context flows forward only.

```python
# Cell 2
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

```python
# Cell 8
generated = model.generate(**inputs, max_new_tokens=32)
```

Cell 8 can use the constructor evidence established in cell 2. A later cell never changes the interpretation of an earlier cell.

## Why static rules instead of an LLM?

The purpose of this layer is not unrestricted natural-language code explanation. It is deterministic methodology reconstruction:

```text
source syntax
    ↓
AST + import resolution
    ↓
forward-only symbol context
    ↓
framework semantic rule
    ↓
SIR operation + claim + evidence
    ↓
Methods report / experiment pipeline
```

This keeps every generated statement inspectable and attributable to an explicit rule.