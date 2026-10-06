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
        return f"{parent}.{node.attr}" if parent else node.attr
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


def _static_value(
    node: ast.AST,
    symbols: SymbolTable,
    imports: ImportTable,
) -> Any | None:
    literal = symbols.resolve_constant(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.List | ast.Tuple):
        values: list[Any] = []
        for element in node.elts:
            value = _static_value(element, symbols, imports)
            if value is None:
                return None
            values.append(value)
        return values
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                return None
            key = _static_value(key_node, symbols, imports)
            value = _static_value(value_node, symbols, imports)
            if not isinstance(key, str) or value is None:
                return None
            result[key] = value
        return result
    if isinstance(node, ast.Attribute):
        name = _dotted_name(node)
        return imports.resolve_dotted(name) if name else None
    return None


def _keyword_values(
    call: ast.Call,
    symbols: SymbolTable,
    imports: ImportTable,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = _static_value(keyword.value, symbols, imports)
        if value is not None:
            values[keyword.arg] = value
    return values


def _root_name(node: ast.AST) -> str | None:
    name = _dotted_name(node)
    return name.split(".", 1)[0] if name else None


def _constructor_for_subject(node: ast.AST, symbols: SymbolTable) -> str | None:
    root = _root_name(node)
    return symbols.resolve_constructor(root) if root else None


def _is_peft_config(path: str | None) -> bool:
    return bool(path and path.startswith("peft.") and path.endswith("Config"))


def _config_family(path: str) -> str:
    name = path.rsplit(".", 1)[-1]
    if name == "LoraConfig":
        return "LoRA"
    if name == "AdaLoraConfig":
        return "AdaLoRA"
    if name == "IA3Config":
        return "IA3"
    return name.removesuffix("Config") or name


def _known_peft_model_constructor(path: str | None) -> bool:
    return path in {
        "peft.get_peft_model",
        "peft.PeftModel.from_pretrained",
        "peft.PeftModelForCausalLM.from_pretrained",
        "peft.PeftModelForSequenceClassification.from_pretrained",
        "peft.PeftModelForSeq2SeqLM.from_pretrained",
        "peft.PeftModelForTokenClassification.from_pretrained",
        "peft.PeftModelForQuestionAnswering.from_pretrained",
        "peft.PeftModelForFeatureExtraction.from_pretrained",
    }


def _adapter_identifier(call: ast.Call, symbols: SymbolTable) -> str | None:
    if len(call.args) >= 2:
        value = symbols.resolve_constant(call.args[1])
        if isinstance(value, str):
            return value
    for keyword in call.keywords:
        if keyword.arg in {"model_id", "adapter_name", "peft_model_id"}:
            value = symbols.resolve_constant(keyword.value)
            if isinstance(value, str):
                return value
    return None


def _config_context(
    node: ast.AST,
    symbols: SymbolTable,
) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(node, ast.Name):
        return None
    context = symbols.constructor_context(node.id)
    if context is None:
        return None
    constructor, attributes, _ = context
    if not _is_peft_config(constructor):
        return None
    return constructor, attributes


def analyze_peft(
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

        if _is_peft_config(path):
            attributes = _keyword_values(node, symbols, imports)
            family = _config_family(path or "")
            ev = _evidence(parsed, node, EvidenceLevel.E2, "peft.adapter_configuration")
            operations.append(
                Operation(
                    "adapter_configuration",
                    "peft",
                    subject=family,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            details: list[str] = []
            for key in ("r", "lora_alpha", "lora_dropout", "bias", "task_type"):
                if key in attributes:
                    details.append(f"{key}={attributes[key]!r}")
            if "target_modules" in attributes:
                details.append(f"target_modules={attributes['target_modules']!r}")
            suffix = f" with {', '.join(details)}" if details else ""
            claims.append(
                Claim(f"A {family} parameter-efficient adapter is configured{suffix}.", ev)
            )
            continue

        if path == "peft.get_peft_model":
            attributes: dict[str, Any] = {}
            if len(node.args) >= 2:
                context = _config_context(node.args[1], symbols)
                if context is not None:
                    constructor, config_attributes = context
                    attributes["adapter_config"] = constructor.rsplit(".", 1)[-1]
                    for key in ("r", "lora_alpha", "lora_dropout", "bias"):
                        if key in config_attributes:
                            attributes[key] = config_attributes[key]
            ev = _evidence(parsed, node, EvidenceLevel.E3, "peft.get_peft_model")
            operations.append(
                Operation(
                    "adapter_application",
                    "peft",
                    attributes=attributes,
                    evidence=ev,
                )
            )
            config = attributes.get("adapter_config")
            suffix = f" using {config}" if isinstance(config, str) else ""
            claims.append(
                Claim(
                    f"PEFT wraps the base model with a parameter-efficient adapter{suffix}.",
                    ev,
                )
            )
            continue

        if path == "peft.prepare_model_for_kbit_training":
            attributes = _keyword_values(node, symbols, imports)
            ev = _evidence(parsed, node, EvidenceLevel.E3, "peft.prepare_model_for_kbit_training")
            operations.append(
                Operation(
                    "kbit_training_preparation",
                    "peft",
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(
                Claim(
                    "The model is passed through PEFT's preparation helper for k-bit training.",
                    ev,
                )
            )
            continue

        if path and path.startswith("peft.PeftModel") and path.endswith(".from_pretrained"):
            attributes = _keyword_values(node, symbols, imports)
            adapter_id = _adapter_identifier(node, symbols)
            if adapter_id is not None:
                attributes["adapter_id"] = adapter_id
            peft_class = path.removesuffix(".from_pretrained").rsplit(".", 1)[-1]
            ev = _evidence(parsed, node, EvidenceLevel.E2, "peft.model_from_pretrained")
            operations.append(
                Operation(
                    "adapter_checkpoint_load",
                    "peft",
                    subject=peft_class,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            source = f" from `{adapter_id}`" if adapter_id is not None else ""
            trainable = attributes.get("is_trainable")
            trainable_suffix = " as trainable" if trainable is True else ""
            claims.append(
                Claim(
                    f"A PEFT adapter checkpoint is loaded{source}{trainable_suffix}.",
                    ev,
                )
            )
            continue

        if path == "peft.get_peft_model_state_dict":
            ev = _evidence(parsed, node, EvidenceLevel.E3, "peft.get_peft_model_state_dict")
            operations.append(Operation("adapter_state_export", "peft", evidence=ev))
            claims.append(
                Claim("The PEFT adapter state dictionary is extracted from the model.", ev)
            )
            continue

        if not isinstance(node.func, ast.Attribute):
            continue

        subject_constructor = _constructor_for_subject(node.func.value, symbols)
        if not _known_peft_model_constructor(subject_constructor):
            continue

        if node.func.attr == "merge_and_unload":
            ev = _evidence(parsed, node, EvidenceLevel.E3, "peft.merge_and_unload")
            operations.append(
                Operation(
                    "adapter_merge",
                    "peft",
                    subject=_dotted_name(node.func.value),
                    evidence=ev,
                )
            )
            claims.append(
                Claim(
                    "The active PEFT adapter weights are merged into the base model and the adapter wrapper is removed.",
                    ev,
                )
            )
            continue

        if node.func.attr == "save_pretrained":
            ev = _evidence(parsed, node, EvidenceLevel.E3, "peft.save_pretrained")
            operations.append(
                Operation(
                    "adapter_checkpoint_save",
                    "peft",
                    subject=_dotted_name(node.func.value),
                    evidence=ev,
                )
            )
            claims.append(
                Claim("The PEFT model or adapter state is saved with `save_pretrained`.", ev)
            )
            continue

        if node.func.attr == "load_adapter":
            adapter_id = _adapter_identifier(node, symbols)
            attributes = {"adapter_id": adapter_id} if adapter_id is not None else {}
            ev = _evidence(parsed, node, EvidenceLevel.E3, "peft.load_adapter")
            operations.append(
                Operation(
                    "adapter_checkpoint_load",
                    "peft",
                    subject=_dotted_name(node.func.value),
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(Claim("An additional PEFT adapter is loaded into the model.", ev))
            continue

        if node.func.attr == "set_adapter":
            adapter_name = symbols.resolve_constant(node.args[0]) if node.args else None
            attributes = (
                {"adapter_name": adapter_name} if isinstance(adapter_name, str) else {}
            )
            ev = _evidence(parsed, node, EvidenceLevel.E3, "peft.set_adapter")
            operations.append(
                Operation(
                    "adapter_activation",
                    "peft",
                    subject=_dotted_name(node.func.value),
                    attributes=attributes,
                    evidence=ev,
                )
            )
            suffix = f" `{adapter_name}`" if isinstance(adapter_name, str) else ""
            claims.append(Claim(f"The PEFT adapter{suffix} is activated.", ev))
            continue

        if node.func.attr == "add_adapter":
            attributes = _keyword_values(node, symbols, imports)
            ev = _evidence(parsed, node, EvidenceLevel.E3, "peft.add_adapter")
            operations.append(
                Operation(
                    "adapter_addition",
                    "peft",
                    subject=_dotted_name(node.func.value),
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(Claim("An additional PEFT adapter is attached to the model.", ev))

    return SemanticOutput(operations=operations, claims=claims)
