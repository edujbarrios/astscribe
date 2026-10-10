from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from astscribe.parser import ImportTable, ParsedSource, SymbolTable
from astscribe.sir import Claim, Evidence, EvidenceLevel, Operation


@dataclass(frozen=True)
class SemanticOutput:
    operations: list[Operation]
    claims: list[Claim]


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _call_path(call: ast.Call, imports: ImportTable) -> str | None:
    name = _dotted_name(call.func)
    return imports.resolve_dotted(name) if name else None


def _evidence(
    parsed: ParsedSource,
    node: ast.AST,
    level: EvidenceLevel,
    rule: str,
) -> Evidence:
    return Evidence(
        level=level,
        rule=rule,
        source=parsed.source_segment(node),
        line_start=getattr(node, "lineno", None),
        line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
        cell=parsed.cell,
    )


def _literal(symbols: SymbolTable, node: ast.AST) -> Any | None:
    return symbols.resolve_constant(node)


def _keyword_values(call: ast.Call, symbols: SymbolTable) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = _literal(symbols, keyword.value)
        if value is not None:
            values[keyword.arg] = value
    return values


def _root_name(node: ast.AST) -> str | None:
    name = _dotted_name(node)
    return name.split(".", 1)[0] if name else None


def _constructor_for_subject(node: ast.AST, symbols: SymbolTable) -> str | None:
    root = _root_name(node)
    return symbols.resolve_constructor(root) if root else None


def _loader_class(path: str) -> str:
    return path.removesuffix(".from_pretrained").rsplit(".", 1)[-1]


def _is_transformers_loader(path: str | None) -> bool:
    return bool(path and path.startswith("transformers.") and path.endswith(".from_pretrained"))


def _is_tokenizer_loader(path: str | None) -> bool:
    return bool(_is_transformers_loader(path) and "Tokenizer" in _loader_class(path or ""))


def _is_processor_loader(path: str | None) -> bool:
    if not _is_transformers_loader(path):
        return False
    name = _loader_class(path or "")
    return any(marker in name for marker in ("Processor", "ImageProcessor", "FeatureExtractor"))


def _is_config_loader(path: str | None) -> bool:
    return bool(_is_transformers_loader(path) and "Config" in _loader_class(path or ""))


def _is_model_loader(path: str | None) -> bool:
    if not _is_transformers_loader(path):
        return False
    name = _loader_class(path or "")
    return "Model" in name and "Config" not in name


def _is_trainer_constructor(path: str | None) -> bool:
    return path in {"transformers.Trainer", "transformers.Seq2SeqTrainer"}


def _pretrained_id(call: ast.Call, symbols: SymbolTable) -> str | None:
    if call.args:
        value = _literal(symbols, call.args[0])
        if isinstance(value, str):
            return value
    for keyword in call.keywords:
        if keyword.arg in {"pretrained_model_name_or_path", "model_name_or_path"}:
            value = _literal(symbols, keyword.value)
            if isinstance(value, str):
                return value
    return None


def _explicit_keyword_names(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


def _format_settings(attributes: dict[str, Any], keys: tuple[str, ...]) -> str:
    settings = [f"{key}={attributes[key]!r}" for key in keys if key in attributes]
    return ", ".join(settings)


def _known_subject_constructor(call: ast.Call, symbols: SymbolTable) -> str | None:
    if isinstance(call.func, ast.Name):
        return symbols.resolve_constructor(call.func.id)
    return None


def analyze_transformers(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []

    for node in ast.walk(parsed.tree):
        if not isinstance(node, ast.Call):
            continue

        path = _call_path(node, imports)

        if _is_tokenizer_loader(path):
            tokenizer_class = _loader_class(path or "")
            model_id = _pretrained_id(node, symbols)
            attributes = _keyword_values(node, symbols)
            attributes["tokenizer_class"] = tokenizer_class
            if model_id is not None:
                attributes["pretrained_model_name_or_path"] = model_id
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.tokenizer_from_pretrained")
            operations.append(
                Operation(
                    "tokenizer_configuration",
                    "transformers",
                    subject=tokenizer_class,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            source = f" from `{model_id}`" if model_id is not None else ""
            claims.append(
                Claim(
                    f"A {tokenizer_class} tokenizer is loaded with `from_pretrained`{source}.",
                    ev,
                )
            )
            continue

        if _is_processor_loader(path):
            processor_class = _loader_class(path or "")
            model_id = _pretrained_id(node, symbols)
            attributes = _keyword_values(node, symbols)
            attributes["processor_class"] = processor_class
            if model_id is not None:
                attributes["pretrained_model_name_or_path"] = model_id
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.processor_from_pretrained")
            operations.append(
                Operation(
                    "processor_configuration",
                    "transformers",
                    subject=processor_class,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            source = f" from `{model_id}`" if model_id is not None else ""
            claims.append(
                Claim(
                    f"A {processor_class} input processor is loaded with `from_pretrained`{source}.",
                    ev,
                )
            )
            continue

        if _is_config_loader(path):
            config_class = _loader_class(path or "")
            model_id = _pretrained_id(node, symbols)
            attributes = _keyword_values(node, symbols)
            attributes["config_class"] = config_class
            if model_id is not None:
                attributes["pretrained_model_name_or_path"] = model_id
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.config_from_pretrained")
            operations.append(
                Operation(
                    "model_config_load",
                    "transformers",
                    subject=config_class,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            source = f" for `{model_id}`" if model_id is not None else ""
            claims.append(
                Claim(f"A Transformers model configuration is loaded{source}.", ev)
            )
            continue

        if _is_model_loader(path):
            model_class = _loader_class(path or "")
            model_id = _pretrained_id(node, symbols)
            attributes = _keyword_values(node, symbols)
            attributes["model_class"] = model_class
            if model_id is not None:
                attributes["pretrained_model_name_or_path"] = model_id
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.model_from_pretrained")
            operations.append(
                Operation(
                    "pretrained_model_configuration",
                    "transformers",
                    subject=model_class,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            source = f" from `{model_id}`" if model_id is not None else ""
            claims.append(
                Claim(
                    f"A {model_class} model is loaded with `from_pretrained`{source}.",
                    ev,
                )
            )
            continue

        if path == "transformers.TrainingArguments" or path == "transformers.Seq2SeqTrainingArguments":
            attributes = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.training_arguments")
            operations.append(
                Operation(
                    "training_arguments_configuration",
                    "transformers",
                    subject=path.rsplit(".", 1)[-1],
                    attributes=attributes,
                    evidence=ev,
                )
            )
            details = _format_settings(
                attributes,
                (
                    "num_train_epochs",
                    "learning_rate",
                    "per_device_train_batch_size",
                    "per_device_eval_batch_size",
                    "weight_decay",
                    "gradient_accumulation_steps",
                    "eval_strategy",
                    "evaluation_strategy",
                    "save_strategy",
                    "fp16",
                    "bf16",
                ),
            )
            suffix = f" with {details}" if details else ""
            claims.append(Claim(f"Transformers training arguments are configured{suffix}.", ev))
            continue

        if _is_trainer_constructor(path):
            keywords = _explicit_keyword_names(node)
            attributes = _keyword_values(node, symbols)
            attributes.update(
                {
                    "has_train_dataset": "train_dataset" in keywords,
                    "has_eval_dataset": "eval_dataset" in keywords,
                    "has_data_collator": "data_collator" in keywords,
                    "has_processing_component": bool(
                        {"tokenizer", "processing_class"}.intersection(keywords)
                    ),
                }
            )
            trainer_class = path.rsplit(".", 1)[-1] if path is not None else "Trainer"
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.trainer_configuration")
            operations.append(
                Operation(
                    "trainer_configuration",
                    "transformers",
                    subject=trainer_class,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            dataset_parts: list[str] = []
            if "train_dataset" in keywords:
                dataset_parts.append("a training dataset")
            if "eval_dataset" in keywords:
                dataset_parts.append("an evaluation dataset")
            suffix = f" with {' and '.join(dataset_parts)}" if dataset_parts else ""
            claims.append(Claim(f"A {trainer_class} is configured{suffix}.", ev))
            continue

        if path and path.startswith("transformers.DataCollator"):
            collator_class = path.rsplit(".", 1)[-1]
            attributes = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.data_collator")
            operations.append(
                Operation(
                    "data_collator_configuration",
                    "transformers",
                    subject=collator_class,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(Claim(f"A {collator_class} data collator is configured.", ev))
            continue

        if path == "transformers.set_seed":
            seed = _literal(symbols, node.args[0]) if node.args else None
            attributes = {"seed": seed} if seed is not None else {}
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.set_seed")
            operations.append(
                Operation("seed_configuration", "transformers", attributes=attributes, evidence=ev)
            )
            suffix = f" to {seed}" if seed is not None else ""
            claims.append(Claim(f"The Transformers random seed is explicitly set{suffix}.", ev))
            continue

        if path == "transformers.pipeline":
            task = _literal(symbols, node.args[0]) if node.args else None
            attributes = _keyword_values(node, symbols)
            if task is None and "task" in attributes:
                task = attributes["task"]
            if task is not None:
                attributes["task"] = task
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.pipeline_configuration")
            operations.append(
                Operation(
                    "inference_pipeline_configuration",
                    "transformers",
                    subject=str(task) if task is not None else None,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            suffix = f" for the `{task}` task" if task is not None else ""
            claims.append(Claim(f"A Transformers inference pipeline is configured{suffix}.", ev))
            continue

        constructor = _known_subject_constructor(node, symbols)
        if _is_tokenizer_loader(constructor):
            attributes = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.tokenization")
            operations.append(
                Operation(
                    "tokenization",
                    "transformers",
                    subject=node.func.id if isinstance(node.func, ast.Name) else None,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            details = _format_settings(
                attributes,
                ("padding", "truncation", "max_length", "return_tensors", "add_special_tokens"),
            )
            suffix = f" with {details}" if details else ""
            claims.append(Claim(f"Tokenizer preprocessing is applied{suffix}.", ev))
            continue

        if _is_processor_loader(constructor):
            attributes = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.input_processing")
            operations.append(
                Operation(
                    "input_processing",
                    "transformers",
                    subject=node.func.id if isinstance(node.func, ast.Name) else None,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(Claim("A Transformers processor prepares model inputs.", ev))
            continue

        if _is_model_loader(constructor):
            keywords = _explicit_keyword_names(node)
            attributes: dict[str, Any] = {}
            if "labels" in keywords:
                attributes["labels_supplied"] = True
            if any(keyword.arg is None for keyword in node.keywords):
                attributes["unpacked_inputs"] = True
            ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.forward_pass")
            operations.append(
                Operation(
                    "forward_pass",
                    "transformers",
                    subject=node.func.id if isinstance(node.func, ast.Name) else None,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(
                Claim("A forward pass is performed through the configured Transformers model.", ev)
            )
            if "labels" in keywords:
                label_ev = _evidence(
                    parsed,
                    node,
                    EvidenceLevel.E1,
                    "transformers.labels_supplied",
                )
                operations.append(
                    Operation("supervision_labels", "transformers", evidence=label_ev)
                )
                claims.append(Claim("Target labels are explicitly supplied to the model call.", label_ev))
            continue

        if isinstance(node.func, ast.Attribute):
            subject_constructor = _constructor_for_subject(node.func.value, symbols)

            if node.func.attr == "generate" and _is_model_loader(subject_constructor):
                attributes = _keyword_values(node, symbols)
                ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.generate")
                operations.append(
                    Operation(
                        "generation",
                        "transformers",
                        subject=_dotted_name(node.func.value),
                        attributes=attributes,
                        evidence=ev,
                    )
                )
                details = _format_settings(
                    attributes,
                    (
                        "max_new_tokens",
                        "max_length",
                        "num_beams",
                        "do_sample",
                        "temperature",
                        "top_p",
                        "top_k",
                    ),
                )
                suffix = f" with {details}" if details else ""
                claims.append(Claim(f"Autoregressive or sequence generation is invoked{suffix}.", ev))
                continue

            if node.func.attr in {"decode", "batch_decode"} and _is_tokenizer_loader(
                subject_constructor
            ):
                attributes = _keyword_values(node, symbols)
                ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.output_decoding")
                operations.append(
                    Operation(
                        "output_decoding",
                        "transformers",
                        subject=_dotted_name(node.func.value),
                        attributes=attributes,
                        evidence=ev,
                    )
                )
                claims.append(Claim("Generated token IDs are decoded back into text.", ev))
                continue

            if _is_trainer_constructor(subject_constructor):
                if node.func.attr == "train":
                    ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.trainer_train")
                    operations.append(Operation("trainer_train", "transformers", evidence=ev))
                    claims.append(Claim("The configured Transformers Trainer starts training.", ev))
                    continue
                if node.func.attr in {"evaluate", "predict"}:
                    rule = f"transformers.trainer_{node.func.attr}"
                    ev = _evidence(parsed, node, EvidenceLevel.E3, rule)
                    operations.append(
                        Operation(
                            "trainer_evaluation",
                            "transformers",
                            subject=node.func.attr,
                            evidence=ev,
                        )
                    )
                    action = "evaluation" if node.func.attr == "evaluate" else "prediction"
                    claims.append(
                        Claim(f"The configured Transformers Trainer runs {action}.", ev)
                    )
                    continue

            if node.func.attr == "save_pretrained" and (
                _is_model_loader(subject_constructor)
                or _is_tokenizer_loader(subject_constructor)
                or _is_processor_loader(subject_constructor)
            ):
                ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.save_pretrained")
                operations.append(
                    Operation(
                        "checkpoint_save",
                        "transformers",
                        subject=_dotted_name(node.func.value),
                        evidence=ev,
                    )
                )
                claims.append(
                    Claim("A Transformers component is saved with `save_pretrained`.", ev)
                )

    return SemanticOutput(operations=operations, claims=claims)
