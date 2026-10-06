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


def _evidence(parsed: ParsedSource, node: ast.AST, level: EvidenceLevel, rule: str) -> Evidence:
    return Evidence(
        level=level,
        rule=rule,
        source=parsed.source_segment(node),
        line_start=getattr(node, "lineno", None),
        line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
        cell=parsed.cell,
    )


def _static_value(symbols: SymbolTable, node: ast.AST) -> Any | None:
    return symbols.resolve_constant(node)


def _assigned_value(node: ast.Assign | ast.AnnAssign) -> tuple[ast.AST | None, ast.AST | None]:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return None, None
        return node.targets[0], node.value
    return node.target, node.value


def analyze_pytorch_reproducibility(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []

    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            target, value_node = _assigned_value(node)
            if target is None or value_node is None:
                continue
            dotted = _dotted_name(target)
            path = imports.resolve_dotted(dotted) if dotted else None
            value = _static_value(symbols, value_node)

            if path == "torch.backends.cudnn.deterministic" and isinstance(value, bool):
                ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.cudnn_deterministic")
                operations.append(
                    Operation(
                        "determinism_configuration",
                        "pytorch",
                        subject="cudnn.deterministic",
                        attributes={"enabled": value},
                        evidence=ev,
                    )
                )
                state = "enabled" if value else "disabled"
                claims.append(
                    Claim(f"cuDNN deterministic execution is explicitly {state}.", ev)
                )
                continue

            if path == "torch.backends.cudnn.benchmark" and isinstance(value, bool):
                ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.cudnn_benchmark")
                operations.append(
                    Operation(
                        "cudnn_benchmark_configuration",
                        "pytorch",
                        attributes={"enabled": value},
                        evidence=ev,
                    )
                )
                state = "enabled" if value else "disabled"
                claims.append(Claim(f"cuDNN benchmarking is explicitly {state}.", ev))
                continue

        if not isinstance(node, ast.Call):
            continue

        path = _call_path(node, imports)
        if path == "torch.cuda.manual_seed_all":
            seed = _static_value(symbols, node.args[0]) if node.args else None
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.cuda_manual_seed_all")
            attrs = {"seed": seed} if seed is not None else {}
            operations.append(
                Operation("cuda_seed_configuration", "pytorch", attributes=attrs, evidence=ev)
            )
            suffix = f" with the value {seed}" if seed is not None else ""
            claims.append(
                Claim(f"All CUDA pseudorandom number generators are seeded{suffix}.", ev)
            )
            continue

        if path == "torch.use_deterministic_algorithms":
            enabled = _static_value(symbols, node.args[0]) if node.args else None
            warn_only = None
            for keyword in node.keywords:
                if keyword.arg == "warn_only":
                    warn_only = _static_value(symbols, keyword.value)
            ev = _evidence(
                parsed,
                node,
                EvidenceLevel.E3,
                "pytorch.use_deterministic_algorithms",
            )
            attrs: dict[str, Any] = {}
            if isinstance(enabled, bool):
                attrs["enabled"] = enabled
            if isinstance(warn_only, bool):
                attrs["warn_only"] = warn_only
            operations.append(
                Operation("deterministic_algorithms", "pytorch", attributes=attrs, evidence=ev)
            )
            if enabled is True:
                text = "PyTorch deterministic algorithms are explicitly requested"
            elif enabled is False:
                text = "PyTorch deterministic-algorithm enforcement is explicitly disabled"
            else:
                text = "PyTorch deterministic-algorithm behavior is explicitly configured"
            if warn_only is True:
                text += " in warning-only mode"
            claims.append(Claim(text + ".", ev))

    return SemanticOutput(operations=operations, claims=claims)
