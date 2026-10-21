from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from .bindings import collect_binding_effects
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
            if isinstance(value, int | float):
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
        return f"{parent}.{node.attr}" if parent else None
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


def _forget_symbol(table: SymbolTable, name: str) -> None:
    table.constants.pop(name, None)
    table.constructors.pop(name, None)
    table.constructor_arguments.pop(name, None)
    table.constructor_origins.pop(name, None)


def _clear_symbol_table(table: SymbolTable) -> None:
    table.constants.clear()
    table.constructors.clear()
    table.constructor_arguments.clear()
    table.constructor_origins.clear()


# These PyTorch-style methods retain the receiver's semantic type.
# Unknown methods (e.g. state_dict()) may return an unrelated object.
_SELF_RETURNING_METHODS = frozenset({
    "to", "cpu", "cuda", "float", "double", "half", "bfloat16",
    "train", "eval", "requires_grad_", "share_memory",
})
# Hugging Face Dataset transformations return another Dataset with equivalent
# provenance even though they are not identity-preserving operations.
_DATASET_TRANSFORMS = frozenset({
    "map", "filter", "shuffle", "select", "sort", "flatten",
    "rename_column", "rename_columns", "remove_columns",
    "select_columns", "cast", "cast_column", "add_column",
    "with_format", "with_transform",
})


def _is_fluent_self_assignment(name: str, value: ast.Call, table: SymbolTable) -> bool:
    if (
        not isinstance(value.func, ast.Attribute)
        or not isinstance(value.func.value, ast.Name)
        or value.func.value.id != name
    ):
        return False
    if value.func.attr in _SELF_RETURNING_METHODS:
        return True
    constructor = table.resolve_constructor(name)
    return bool(
        constructor
        and constructor.startswith("datasets.")
        and value.func.attr in _DATASET_TRANSFORMS
    )


def _bound_values(node: ast.stmt) -> list[tuple[str, ast.AST]]:
    """Extract safe name/value pairs from parallel assignments."""
    def unpack(target: ast.AST, value: ast.AST) -> list[tuple[str, ast.AST]]:
        if isinstance(target, ast.Name):
            return [(target.id, value)]
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            pairs: list[tuple[str, ast.AST]] = []
            for child, item in zip(target.elts, value.elts, strict=True):
                pairs.extend(unpack(child, item))
            return pairs
        return []

    if isinstance(node, ast.Assign):
        pairs: list[tuple[str, ast.AST]] = []
        for target in node.targets:
            pairs.extend(unpack(target, node.value))
        return pairs
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return unpack(node.target, node.value)
    return []


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
        # Evaluate every RHS before invalidating any assignment target.
        # In a parallel swap, both RHS values refer to the original bindings.
        prepared: list[tuple[str, ast.AST, Any, Any, dict[str, Any], str | None]] = []
        for name, value in _bound_values(node):
            constant = table.resolve_constant(value)
            context = (
                table.constructor_context(value.id)
                if isinstance(value, ast.Name)
                else None
            )
            if isinstance(value, ast.Call) and _is_fluent_self_assignment(name, value, table):
                context = table.constructor_context(name)
            call_arguments = (
                _keyword_values(value, table) if isinstance(value, ast.Call) else {}
            )
            called = _dotted_name(value.func) if isinstance(value, ast.Call) else None
            prepared.append((name, value, constant, context, call_arguments, called))

        effects = collect_binding_effects(node)
        if effects.clears_all:
            _clear_symbol_table(table)
        for rebound in effects.names:
            _forget_symbol(table, rebound)

        for name, value, constant, context, call_arguments, called in prepared:
            if constant is not None:
                table.constants[name] = constant
                continue
            if context is not None:
                constructor, arguments, origin = context
                table.constructors[name] = constructor
                table.constructor_arguments[name] = arguments
                if origin is not None:
                    table.constructor_origins[name] = origin
                continue
            if called:
                table.constructors[name] = imports.resolve_dotted(called)
                table.constructor_arguments[name] = call_arguments
                table.constructor_origins[name] = SymbolOrigin(
                    source=(ast.get_source_segment(source, value) or "") if source else "",
                    line_start=getattr(value, "lineno", None),
                    line_end=getattr(value, "end_lineno", getattr(value, "lineno", None)),
                    cell=cell,
                )

    return table
