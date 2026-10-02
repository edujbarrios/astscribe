from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from .imports import ImportTable

_LITERAL_TYPES = (str, int, float, bool, type(None))


@dataclass
class SymbolTable:
    constants: dict[str, Any] = field(default_factory=dict)
    constructors: dict[str, str] = field(default_factory=dict)

    def resolve_constant(self, node: ast.AST) -> Any | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, _LITERAL_TYPES):
            return node.value
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self.resolve_constant(node.operand)
            if isinstance(value, (int, float)):
                return -value
        return None


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def build_symbol_table(tree: ast.Module, imports: ImportTable) -> SymbolTable:
    table = SymbolTable()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target: ast.AST | None
        value: ast.AST | None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            target = node.target if isinstance(node, ast.AnnAssign) else None
            value = node.value if isinstance(node, ast.AnnAssign) else None
        if not isinstance(target, ast.Name) or value is None:
            continue
        constant = table.resolve_constant(value)
        if constant is not None:
            table.constants[target.id] = constant
            continue
        if isinstance(value, ast.Call):
            called = _dotted_name(value.func)
            if called:
                table.constructors[target.id] = imports.resolve_dotted(called)
    return table
