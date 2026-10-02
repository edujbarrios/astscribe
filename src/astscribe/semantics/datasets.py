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
    if isinstance(node, (ast.List, ast.Tuple)):
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


def _dataset_source(call: ast.Call, symbols: SymbolTable) -> str | None:
    if call.args:
        value = symbols.resolve_constant(call.args[0])
        if isinstance(value, str):
            return value
    for keyword in call.keywords:
        if keyword.arg == "path":
            value = symbols.resolve_constant(keyword.value)
            if isinstance(value, str):
                return value
    return None


def _root_name(node: ast.AST) -> str | None:
    name = _dotted_name(node)
    return name.split(".", 1)[0] if name else None


def _has_dataset_lineage(name: str, symbols: SymbolTable, seen: set[str] | None = None) -> bool:
    visited = set() if seen is None else seen
    if name in visited:
        return False
    visited.add(name)
    constructor = symbols.resolve_constructor(name)
    if constructor is None:
        return False
    if constructor.startswith("datasets."):
        return True
    root = constructor.split(".", 1)[0]
    if root != name:
        return _has_dataset_lineage(root, symbols, visited)
    return False


def _dataset_method(call: ast.Call, symbols: SymbolTable) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    root = _root_name(call.func.value)
    if root is None or not _has_dataset_lineage(root, symbols):
        return None
    return call.func.attr


def analyze_datasets(
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
        if path == "datasets.load_dataset":
            attributes = _keyword_values(node, symbols, imports)
            source = _dataset_source(node, symbols)
            if source is not None:
                attributes["path"] = source
            ev = _evidence(parsed, node, EvidenceLevel.E2, "datasets.load_dataset")
            operations.append(
                Operation(
                    "dataset_configuration",
                    "datasets",
                    subject=source,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            source_text = f" `{source}`" if source is not None else ""
            split = attributes.get("split")
            split_text = f" using split `{split}`" if isinstance(split, str) else ""
            streaming_text = " in streaming mode" if attributes.get("streaming") is True else ""
            claims.append(
                Claim(
                    f"A Hugging Face dataset{source_text} is loaded{split_text}{streaming_text}.",
                    ev,
                )
            )
            continue

        if path == "datasets.load_from_disk":
            attributes = _keyword_values(node, symbols, imports)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "datasets.load_from_disk")
            operations.append(
                Operation("dataset_configuration", "datasets", attributes=attributes, evidence=ev)
            )
            claims.append(Claim("A previously saved Hugging Face dataset is loaded from disk.", ev))
            continue

        method = _dataset_method(node, symbols)
        if method is None:
            continue

        attributes = _keyword_values(node, symbols, imports)
        subject = _dotted_name(node.func.value) if isinstance(node.func, ast.Attribute) else None

        if method == "map":
            ev = _evidence(parsed, node, EvidenceLevel.E3, "datasets.map")
            operations.append(
                Operation(
                    "dataset_mapping",
                    "datasets",
                    subject=subject,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            details: list[str] = []
            if attributes.get("batched") is True:
                details.append("batched=True")
            if "num_proc" in attributes:
                details.append(f"num_proc={attributes['num_proc']!r}")
            suffix = f" with {', '.join(details)}" if details else ""
            claims.append(Claim(f"A mapping transformation is applied to the dataset{suffix}.", ev))
            continue

        if method == "filter":
            ev = _evidence(parsed, node, EvidenceLevel.E3, "datasets.filter")
            operations.append(
                Operation("dataset_filter", "datasets", subject=subject, attributes=attributes, evidence=ev)
            )
            claims.append(Claim("The dataset is filtered using an explicit predicate.", ev))
            continue

        if method == "train_test_split":
            ev = _evidence(parsed, node, EvidenceLevel.E3, "datasets.train_test_split")
            operations.append(
                Operation("dataset_split", "datasets", subject=subject, attributes=attributes, evidence=ev)
            )
            details = []
            for key in ("test_size", "train_size", "seed", "shuffle"):
                if key in attributes:
                    details.append(f"{key}={attributes[key]!r}")
            suffix = f" with {', '.join(details)}" if details else ""
            claims.append(Claim(f"The dataset is partitioned into train/test splits{suffix}.", ev))
            continue

        if method == "shuffle":
            ev = _evidence(parsed, node, EvidenceLevel.E3, "datasets.shuffle")
            operations.append(
                Operation("dataset_shuffle", "datasets", subject=subject, attributes=attributes, evidence=ev)
            )
            seed = attributes.get("seed")
            suffix = f" with seed={seed!r}" if seed is not None else ""
            claims.append(Claim(f"The dataset order is shuffled{suffix}.", ev))
            continue

        if method == "select":
            ev = _evidence(parsed, node, EvidenceLevel.E3, "datasets.select")
            operations.append(Operation("dataset_selection", "datasets", subject=subject, evidence=ev))
            claims.append(Claim("A subset of dataset rows is selected explicitly.", ev))
            continue

        if method in {"remove_columns", "rename_column", "rename_columns", "select_columns"}:
            ev = _evidence(parsed, node, EvidenceLevel.E3, f"datasets.{method}")
            operations.append(
                Operation(
                    "dataset_schema_transform",
                    "datasets",
                    subject=subject,
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(Claim("The dataset column schema is transformed explicitly.", ev))

    return SemanticOutput(operations=operations, claims=claims)
