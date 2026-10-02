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


def _loader_class(path: str) -> str:
    return path.removesuffix(".from_pretrained").rsplit(".", 1)[-1]


def _is_transformers_loader(path: str | None) -> bool:
    return bool(path and path.startswith("transformers.") and path.endswith(".from_pretrained"))


def _is_nonstandard_model_loader(path: str | None) -> bool:
    if not _is_transformers_loader(path):
        return False
    name = _loader_class(path or "")
    if "Model" in name:
        return False
    return not any(
        marker in name
        for marker in (
            "Tokenizer",
            "Processor",
            "FeatureExtractor",
            "Config",
        )
    )


def _pretrained_id(call: ast.Call, symbols: SymbolTable) -> str | None:
    if call.args:
        value = symbols.resolve_constant(call.args[0])
        if isinstance(value, str):
            return value
    for keyword in call.keywords:
        if keyword.arg in {"pretrained_model_name_or_path", "model_name_or_path"}:
            value = symbols.resolve_constant(keyword.value)
            if isinstance(value, str):
                return value
    return None


def _keyword_values(call: ast.Call, symbols: SymbolTable) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = symbols.resolve_constant(keyword.value)
        if value is not None:
            values[keyword.arg] = value
    return values


def _root_name(node: ast.AST) -> str | None:
    name = _dotted_name(node)
    return name.split(".", 1)[0] if name else None


def _constructor_for_subject(node: ast.AST, symbols: SymbolTable) -> str | None:
    root = _root_name(node)
    return symbols.resolve_constructor(root) if root else None


def analyze_transformers_models(
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
        if _is_nonstandard_model_loader(path):
            model_class = _loader_class(path or "")
            model_id = _pretrained_id(node, symbols)
            load_attributes = _keyword_values(node, symbols)
            load_attributes["model_class"] = model_class
            if model_id is not None:
                load_attributes["pretrained_model_name_or_path"] = model_id
            ev = _evidence(parsed, node, EvidenceLevel.E2, "transformers.model_from_pretrained")
            operations.append(
                Operation(
                    "pretrained_model_configuration",
                    "transformers",
                    subject=model_class,
                    attributes=load_attributes,
                    evidence=ev,
                )
            )
            source = f" from `{model_id}`" if model_id is not None else ""
            claims.append(
                Claim(f"A {model_class} model is loaded with `from_pretrained`{source}.", ev)
            )
            continue

        if isinstance(node.func, ast.Name):
            constructor = symbols.resolve_constructor(node.func.id)
            if _is_nonstandard_model_loader(constructor):
                keywords = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
                forward_attributes: dict[str, Any] = {}
                if "labels" in keywords:
                    forward_attributes["labels_supplied"] = True
                if any(keyword.arg is None for keyword in node.keywords):
                    forward_attributes["unpacked_inputs"] = True
                ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.forward_pass")
                operations.append(
                    Operation(
                        "forward_pass",
                        "transformers",
                        subject=node.func.id,
                        attributes=forward_attributes,
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
                    claims.append(
                        Claim("Target labels are explicitly supplied to the model call.", label_ev)
                    )
                continue

        if isinstance(node.func, ast.Attribute) and node.func.attr == "generate":
            constructor = _constructor_for_subject(node.func.value, symbols)
            if _is_nonstandard_model_loader(constructor):
                generation_attributes = _keyword_values(node, symbols)
                ev = _evidence(parsed, node, EvidenceLevel.E3, "transformers.generate")
                operations.append(
                    Operation(
                        "generation",
                        "transformers",
                        subject=_dotted_name(node.func.value),
                        attributes=generation_attributes,
                        evidence=ev,
                    )
                )
                claims.append(
                    Claim("Autoregressive generation is invoked on the configured Transformers model.", ev)
                )

    return SemanticOutput(operations=operations, claims=claims)
