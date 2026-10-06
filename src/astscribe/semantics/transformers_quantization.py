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


def _static_value(node: ast.AST, symbols: SymbolTable, imports: ImportTable) -> Any | None:
    literal = symbols.resolve_constant(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.Attribute):
        name = _dotted_name(node)
        return imports.resolve_dotted(name) if name else None
    if isinstance(node, ast.List | ast.Tuple):
        values: list[Any] = []
        for element in node.elts:
            value = _static_value(element, symbols, imports)
            if value is None:
                return None
            values.append(value)
        return values
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


def _bitsandbytes_bits(attributes: dict[str, Any]) -> int | None:
    if attributes.get("load_in_4bit") is True:
        return 4
    if attributes.get("load_in_8bit") is True:
        return 8
    return None


def _is_transformers_loader(path: str | None) -> bool:
    return bool(path and path.startswith("transformers.") and path.endswith(".from_pretrained"))


def _loader_class(path: str) -> str:
    return path.removesuffix(".from_pretrained").rsplit(".", 1)[-1]


def _is_non_model_loader(path: str | None) -> bool:
    if not _is_transformers_loader(path):
        return True
    name = _loader_class(path or "")
    return any(marker in name for marker in ("Tokenizer", "Processor", "FeatureExtractor", "Config"))


def _quantization_context(
    node: ast.AST,
    symbols: SymbolTable,
    imports: ImportTable,
) -> tuple[str, dict[str, Any]] | None:
    if isinstance(node, ast.Name):
        context = symbols.constructor_context(node.id)
        if context is None:
            return None
        constructor, attributes, _ = context
        if constructor == "transformers.BitsAndBytesConfig":
            return constructor, attributes
        return None
    if isinstance(node, ast.Call):
        path = _call_path(node, imports)
        if path == "transformers.BitsAndBytesConfig":
            return path, _keyword_values(node, symbols, imports)
    return None


def analyze_transformers_quantization(
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
        if path == "transformers.BitsAndBytesConfig":
            config_attributes = _keyword_values(node, symbols, imports)
            bits = _bitsandbytes_bits(config_attributes)
            if bits is not None:
                config_attributes["bits"] = bits
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.bitsandbytes_config")
            operations.append(
                Operation(
                    "quantization_configuration",
                    "transformers",
                    subject="bitsandbytes",
                    attributes=config_attributes,
                    evidence=ev,
                )
            )
            if bits is None:
                text = "A bitsandbytes quantization configuration is declared."
            else:
                text = f"A bitsandbytes {bits}-bit quantization configuration is declared."
            details: list[str] = []
            for key in (
                "bnb_4bit_quant_type",
                "bnb_4bit_compute_dtype",
                "bnb_4bit_use_double_quant",
                "llm_int8_threshold",
            ):
                if key in config_attributes:
                    details.append(f"{key}={config_attributes[key]!r}")
            if details:
                text = f"{text[:-1]} with {', '.join(details)}."
            claims.append(Claim(text, ev))
            continue

        if not _is_transformers_loader(path) or _is_non_model_loader(path):
            continue

        load_attributes: dict[str, Any] = {}
        config_context: tuple[str, dict[str, Any]] | None = None
        for keyword in node.keywords:
            if keyword.arg == "quantization_config":
                config_context = _quantization_context(keyword.value, symbols, imports)
                break

        if config_context is not None:
            constructor, config_attributes = config_context
            load_attributes.update(config_attributes)
            load_attributes["quantization_config"] = constructor.rsplit(".", 1)[-1]
        else:
            direct = _keyword_values(node, symbols, imports)
            for key in ("load_in_4bit", "load_in_8bit"):
                if key in direct:
                    load_attributes[key] = direct[key]

        bits = _bitsandbytes_bits(load_attributes)
        if bits is None:
            continue
        load_attributes["bits"] = bits

        model_class = _loader_class(path or "")
        ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.quantized_model_load")
        operations.append(
            Operation(
                "quantized_model_load",
                "transformers",
                subject=model_class,
                attributes=load_attributes,
                evidence=ev,
            )
        )
        claims.append(
            Claim(
                f"The {model_class} model is loaded with explicit {bits}-bit bitsandbytes quantization.",
                ev,
            )
        )

    return SemanticOutput(operations=operations, claims=claims)
