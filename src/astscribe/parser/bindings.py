from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class BindingEffects:
    names: frozenset[str]
    clears_all: bool = False


class _BindingCollector(ast.NodeVisitor):
    """Collect names that can be rebound in the current execution scope."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.clears_all = False

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.clears_all = True
            else:
                self.names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
            self.visit(node.target)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.names.add(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # Comprehension targets live in the implicit comprehension scope.
        self.visit(node.iter)
        for condition in node.ifs:
            self.visit(condition)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.elt)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.key)
        self.visit(node.value)

    def _visit_match_pattern(self, pattern: ast.pattern) -> None:
        if isinstance(pattern, ast.MatchValue):
            self.visit(pattern.value)
        elif isinstance(pattern, ast.MatchSequence):
            for child in pattern.patterns:
                self._visit_match_pattern(child)
        elif isinstance(pattern, ast.MatchStar):
            if pattern.name is not None:
                self.names.add(pattern.name)
        elif isinstance(pattern, ast.MatchMapping):
            for key in pattern.keys:
                self.visit(key)
            for child in pattern.patterns:
                self._visit_match_pattern(child)
            if pattern.rest is not None:
                self.names.add(pattern.rest)
        elif isinstance(pattern, ast.MatchClass):
            self.visit(pattern.cls)
            for child in (*pattern.patterns, *pattern.kwd_patterns):
                self._visit_match_pattern(child)
        elif isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                self._visit_match_pattern(pattern.pattern)
            if pattern.name is not None:
                self.names.add(pattern.name)
        elif isinstance(pattern, ast.MatchOr):
            for child in pattern.patterns:
                self._visit_match_pattern(child)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            self._visit_match_pattern(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)


def collect_binding_effects(node: ast.AST) -> BindingEffects:
    collector = _BindingCollector()
    collector.visit(node)
    return BindingEffects(
        names=frozenset(collector.names),
        clears_all=collector.clears_all,
    )
