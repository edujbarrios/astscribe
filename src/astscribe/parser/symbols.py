from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from .imports import ImportTable

_LITERAL_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class SymbolOrigin:
    source: str
    line_start: int | None
    line_end: int | None
    cell: int | None = None


@dataclass
class SymbolTable:
    constants: dict[str, Any] = field(default_factory=dict)
    constructors: dict[str, str] = field(default_factory=dict)
    constructor_arguments: dict[str, dict[str, Any]] = field(default_factory=dict)
    constructor_origins: dict[str, SymbolOrigin] = field(default_factory=dict)

    def copy(self) -> SymbolTable:
        return SymbolTable(
            constants=dict(self.constants),
            constructors=dict(self.constructors),
            constructor_arguments={
                name: dict(arguments) for name, arguments in self.constructor_arguments.items()
            },
            constructor_origins=dict(self.constructor_origins),
        )

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

    def resolve_constructor(self, name: str) -> str | None:
        return self.constructors.get(name)

    def constructor_context(
        self,
        name: str,
    ) -> tuple[str, dict[str, Any], SymbolOrigin | None] | None:
        constructor = self.constructors.get(name)
        if constructor is None:
            return None
        return (
            constructor,
            dict(self.constructor_arguments.get(name, {})),
            self.constructor_origins.get(name),
        )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _keyword_values(call: ast.Call, table: SymbolTable) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = table.resolve_constant(keyword.value)
        if value is not None:
            values[keyword.arg] = value
    return values


def build_symbol_table(
    tree: ast.Module,
    imports: ImportTable,
    *,
    base: SymbolTable | None = None,
    source: str = "",
    cell: int | None = None,
) -> SymbolTable:
    table = base.copy() if base is not None else SymbolTable()

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

        name = target.id
        constant = table.resolve_constant(value)
        if constant is not None:
            table.constants[name] = constant
            table.constructors.pop(name, None)
            table.constructor_arguments.pop(name, None)
            table.constructor_origins.pop(name, None)
            continue

        table.constants.pop(name, None)

        if isinstance(value, ast.Call):
            called = _dotted_name(value.func)
            if called:
                table.constructors[name] = imports.resolve_dotted(called)
                table.constructor_arguments[name] = _keyword_values(value, table)
                table.constructor_origins[name] = SymbolOrigin(
                    source=ast.get_source_segment(source, value) or "",
                    line_start=getattr(value, "lineno", None),
                    line_end=getattr(value, "end_lineno", getattr(value, "lineno", None)),
                    cell=cell,
                )
                continue

        table.constructors.pop(name, None)
        table.constructor_arguments.pop(name, None)
        table.constructor_origins.pop(name, None)

    return table
