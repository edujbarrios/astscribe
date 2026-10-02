# PEFT and LoRA semantics

ASTScribe analyzes parameter-efficient fine-tuning code without importing or executing `peft`.

The PEFT analyzer follows the same evidence policy as the rest of ASTScribe: report explicit source configuration, resolve notebook context forward across cells, and attach every methodological statement to a narrow semantic rule.

## Adapter configuration

ASTScribe recognizes `peft.*Config` constructor calls. `LoraConfig` is described as LoRA; `AdaLoraConfig` as AdaLoRA; `IA3Config` as IA3. Other PEFT config classes retain their source class name rather than being mapped to an inferred algorithm.

For statically resolvable values, the analyzer can preserve adapter settings such as:

- `r`;
- `lora_alpha`;
- `lora_dropout`;
- `bias`;
- `target_modules` when written as a literal list or tuple;
- explicit PEFT task type expressions.

Example:

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
```

ASTScribe reports these values because they are present in source. It does not infer missing defaults.

## Applying an adapter

`get_peft_model(base_model, config)` is represented as parameter-efficient adaptation of a base model. When `config` was created in an earlier notebook cell, ASTScribe can use the forward-only symbol table to recover its constructor and statically known scalar hyperparameters.

```python
# Earlier cell
config = LoraConfig(r=8, lora_alpha=16)

# Later cell
model = get_peft_model(model, config)
```

This produces a dedicated parameter-efficient fine-tuning stage rather than treating the adapter as part of base model construction.

## K-bit training preparation

`prepare_model_for_kbit_training(...)` is reported narrowly as use of PEFT's k-bit training preparation helper.

ASTScribe does **not** label this as QLoRA by itself. QLoRA is a compound methodology and should only be identified when sufficient independent evidence exists for both quantized base-model loading and LoRA-style adaptation.

## Adapter checkpoints

ASTScribe recognizes supported `PeftModel*.from_pretrained(...)` calls as adapter checkpoint loading. Explicit adapter identifiers and `is_trainable` values are recorded when statically resolvable.

The analyzer also recognizes these operations on statically known PEFT models:

- `save_pretrained(...)`;
- `load_adapter(...)`;
- `set_adapter(...)`;
- `add_adapter(...)`;
- `merge_and_unload()`;
- `get_peft_model_state_dict(...)`.

## Methods and pipeline semantics

PEFT configuration and application are rendered in a dedicated Methods section:

```text
## Parameter-efficient fine-tuning
```

The structured experiment pipeline gains an optional stage:

```text
Model architecture
    ↓
Parameter-efficient fine-tuning
    ↓
Objective
```

The stage is omitted when no supported PEFT evidence is present, so existing non-PEFT pipelines remain unchanged.

## Evidence boundaries

ASTScribe does **not** infer:

- that an experiment is QLoRA from k-bit preparation alone;
- effective trainable-parameter counts unless explicitly available in source;
- memory savings, speedups, or quality improvements;
- which modules actually matched a target-module pattern at runtime;
- quantization state that is not separately explicit in supported source;
- whether an adapter checkpoint exists remotely;
- whether merging an adapter changes model quality;
- convergence or scientific validity.

The goal is methodology reconstruction, not runtime simulation or performance prediction.