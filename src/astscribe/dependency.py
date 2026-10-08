from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any, Literal

EventKind = Literal["read", "write", "delete"]
ScopeKind = Literal["class", "comprehension"]
_BUILTIN_NAMES = frozenset(dir(__import__("builtins"))) | {
    "__name__",
    "__file__",
    "__package__",
}


@dataclass(frozen=True)
class SymbolEvent:
    kind: EventKind
    symbol: str
    line: int | None
    column: int | None


@dataclass(frozen=True)
class CellDependencyNode:
    cell: int
    defines: tuple[str, ...]
    reads: tuple[str, ...]
    unresolved_reads: tuple[str, ...]


@dataclass(frozen=True)
class CellDependencyEdge:
    producer_cell: int
    consumer_cell: int
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class SymbolRedefinition:
    symbol: str
    previous_cell: int
    new_cell: int


@dataclass(frozen=True)
class NotebookDependencyGraph:
    nodes: tuple[CellDependencyNode, ...]
    edges: tuple[CellDependencyEdge, ...]
    redefinitions: tuple[SymbolRedefinition, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parents(self, cell: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    edge.producer_cell
                    for edge in self.edges
                    if edge.consumer_cell == cell
                }
            )
        )

    def children(self, cell: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    edge.consumer_cell
                    for edge in self.edges
                    if edge.producer_cell == cell
                }
            )
        )

    def render(self) -> str:
        if not self.nodes:
            return "No analyzable notebook cells are available."
        if not self.edges:
            return "No cross-cell symbol dependencies were detected."

        lines = ["# Cell dependency graph", ""]
        for edge in self.edges:
            symbols = ", ".join(edge.symbols)
            lines.append(
                f"Cell {edge.producer_cell} -> Cell {edge.consumer_cell} [{symbols}]"
            )
        return "\n".join(lines)

    def to_dot(self) -> str:
        lines = ["digraph ASTScribeNotebook {", "  rankdir=LR;"]
        for node in self.nodes:
            lines.append(f'  c{node.cell} [label="Cell {node.cell}"];')
        for edge in self.edges:
            symbols = ", ".join(edge.symbols).replace('"', '\\"')
            lines.append(
                f'  c{edge.producer_cell} -> c{edge.consumer_cell} [label="{symbols}"];'
            )
        lines.append("}")
        return "\n".join(lines)


class _EventCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.events: list[SymbolEvent] = []
        self._locals: list[set[str]] = []
        self._scope_kinds: list[ScopeKind] = []

    def _emit(self, kind: EventKind, symbol: str, node: ast.AST) -> None:
        self.events.append(
            SymbolEvent(
                kind=kind,
                symbol=symbol,
                line=getattr(node, "lineno", None),
                column=getattr(node, "col_offset", None),
            )
        )

    def _is_local(self, name: str) -> bool:
        for depth, (scope, kind) in enumerate(
            zip(reversed(self._locals), reversed(self._scope_kinds), strict=True)
        ):
            # Class namespaces are not enclosing lexical scopes for nested
            # classes or comprehensions. The current class body can use its
            # own names, but nested scopes must fall back past outer classes.
            if depth > 0 and kind == "class":
                continue
            if name in scope:
                return True
        return False

    def _define_name(self, name: str, node: ast.AST) -> None:
        if self._locals:
            self._locals[-1].add(name)
        else:
            self._emit("write", name, node)

    def _write_target(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self._define_name(node.id, node)
            return
        if isinstance(node, ast.Tuple | ast.List):
            for element in node.elts:
                self._write_target(element)
            return
        if isinstance(node, ast.Starred):
            self._write_target(node.value)
            return
        # Attribute/subscript assignment reads the base/index expressions but
        # does not create a new notebook-global symbol.
        self.visit(node)

    def _write_named_expr_target(self, node: ast.Name) -> None:
        if self._scope_kinds and self._scope_kinds[-1] == "comprehension":
            # Assignment expressions inside comprehensions bind in the nearest
            # containing non-comprehension scope (PEP 572), not in the
            # comprehension's implicit nested scope.
            for index in range(len(self._scope_kinds) - 2, -1, -1):
                if self._scope_kinds[index] != "comprehension":
                    self._locals[index].add(node.id)
                    return
            self._emit("write", node.id, node)
            return
        self._write_target(node)

    def _delete_target(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if self._locals:
                self._locals[-1].discard(node.id)
            else:
                self._emit("delete", node.id, node)
            return
        if isinstance(node, ast.Tuple | ast.List):
            for element in node.elts:
                self._delete_target(element)
            return
        self.visit(node)

    def _visit_function_definition(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._define_name(node.name, node)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._write_target(node.target)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._write_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._is_local(node.id):
            self._emit("read", node.id, node)
        elif isinstance(node.ctx, ast.Store):
            self._write_target(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._write_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._write_target(node.target)
        elif not isinstance(node.target, ast.Name):
            # Annotation-only names do not bind a runtime value. Attribute
            # and subscript targets can still evaluate their base/index.
            self.visit(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            # Augmented assignment reads the old value before rebinding it.
            if not self._is_local(node.target.id):
                self._emit("read", node.target.id, node.target)
            self.visit(node.value)
            self._write_target(node.target)
        else:
            # Attribute/subscript targets are evaluated once and do not bind
            # notebook-global names. Visiting the target twice would duplicate
            # reads of its base object and index expressions.
            self._write_target(node.target)
            self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._write_named_expr_target(node.target)
        else:
            self._write_target(node.target)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._delete_target(target)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".", 1)[0]
            self._define_name(name, node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self._define_name(alias.asname or alias.name, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._define_name(node.name, node)
        for statement in node.body:
            self.visit(statement)
        if node.name is not None:
            # Python clears the exception target after the handler to break
            # reference cycles, so it must not remain a future producer.
            if self._locals:
                self._locals[-1].discard(node.name)
            else:
                self._emit("delete", node.name, node)

    def _visit_match_pattern(self, pattern: ast.pattern) -> None:
        if isinstance(pattern, ast.MatchValue):
            self.visit(pattern.value)
            return
        if isinstance(pattern, ast.MatchSingleton):
            return
        if isinstance(pattern, ast.MatchSequence):
            for child in pattern.patterns:
                self._visit_match_pattern(child)
            return
        if isinstance(pattern, ast.MatchStar):
            if pattern.name is not None:
                self._define_name(pattern.name, pattern)
            return
        if isinstance(pattern, ast.MatchMapping):
            for key in pattern.keys:
                self.visit(key)
            for child in pattern.patterns:
                self._visit_match_pattern(child)
            if pattern.rest is not None:
                self._define_name(pattern.rest, pattern)
            return
        if isinstance(pattern, ast.MatchClass):
            self.visit(pattern.cls)
            for child in (*pattern.patterns, *pattern.kwd_patterns):
                self._visit_match_pattern(child)
            return
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                self._visit_match_pattern(pattern.pattern)
            if pattern.name is not None:
                self._define_name(pattern.name, pattern)
            return
        if isinstance(pattern, ast.MatchOr):
            for child in pattern.patterns:
                self._visit_match_pattern(child)
            return
        self.generic_visit(pattern)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            self._visit_match_pattern(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

        # A class body executes immediately, unlike a function body. Track reads
        # from notebook globals while keeping names bound inside the class local.
        self._locals.append(set())
        self._scope_kinds.append("class")
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scope_kinds.pop()
            self._locals.pop()

        self._define_name(node.name, node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        final_nodes: tuple[ast.AST, ...],
    ) -> None:
        first, *remaining = generators

        # Python evaluates the outermost iterable in the surrounding scope
        # before entering the comprehension's nested scope. This matters in
        # class bodies, whose namespace is visible to that first iterable but
        # is not an enclosing lexical scope for the rest of the comprehension.
        self.visit(first.iter)

        self._locals.append(set())
        self._scope_kinds.append("comprehension")
        try:
            self._write_target(first.target)
            for condition in first.ifs:
                self.visit(condition)

            for generator in remaining:
                self.visit(generator.iter)
                self._write_target(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)

            for node in final_nodes:
                self.visit(node)
        finally:
            self._scope_kinds.pop()
            self._locals.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, (node.key, node.value))


def collect_symbol_events(source: str) -> tuple[SymbolEvent, ...]:
    tree = ast.parse(source)
    collector = _EventCollector()
    collector.visit(tree)
    return tuple(collector.events)


def build_dependency_graph(
    cells: tuple[tuple[int, str], ...],
) -> NotebookDependencyGraph:
    producer: dict[str, int] = {}
    nodes: list[CellDependencyNode] = []
    edge_symbols: dict[tuple[int, int], set[str]] = {}
    redefinitions: list[SymbolRedefinition] = []

    for cell, source in cells:
        events = collect_symbol_events(source)
        defines: set[str] = set()
        reads: set[str] = set()
        unresolved: set[str] = set()

        for event in events:
            if event.kind == "read":
                reads.add(event.symbol)
                source_cell = producer.get(event.symbol)
                if source_cell is None:
                    if event.symbol not in _BUILTIN_NAMES:
                        unresolved.add(event.symbol)
                elif source_cell != cell:
                    edge_symbols.setdefault((source_cell, cell), set()).add(event.symbol)
                continue

            if event.kind == "delete":
                producer.pop(event.symbol, None)
                continue

            defines.add(event.symbol)
            previous_cell = producer.get(event.symbol)
            if previous_cell is not None and previous_cell != cell:
                redefinitions.append(
                    SymbolRedefinition(
                        symbol=event.symbol,
                        previous_cell=previous_cell,
                        new_cell=cell,
                    )
                )
            producer[event.symbol] = cell

        nodes.append(
            CellDependencyNode(
                cell=cell,
                defines=tuple(sorted(defines)),
                reads=tuple(sorted(reads)),
                unresolved_reads=tuple(sorted(unresolved)),
            )
        )

    edges = tuple(
        CellDependencyEdge(
            producer_cell=producer_cell,
            consumer_cell=consumer_cell,
            symbols=tuple(sorted(symbols)),
        )
        for (producer_cell, consumer_cell), symbols in sorted(edge_symbols.items())
    )
    return NotebookDependencyGraph(
        nodes=tuple(nodes),
        edges=edges,
        redefinitions=tuple(redefinitions),
    )
