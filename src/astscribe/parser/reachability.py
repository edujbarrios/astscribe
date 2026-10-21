"""Remove only control-flow branches whose reachability follows from literal booleans.

This deliberately does not interpret names, compute conditions, or execute code.
The AST is copied so the source parser's original tree remains unchanged.
"""

from __future__ import annotations

import ast
import copy
from typing import cast


class _StaticUnreachablePruner(ast.NodeTransformer):
    def _selected_statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        module = ast.Module(body=statements, type_ignores=[])
        return cast(ast.Module, self.visit(module)).body

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
            return self._selected_statements(node.body if node.test.value else node.orelse)
        return self.generic_visit(node)

    def visit_While(self, node: ast.While) -> ast.AST | list[ast.stmt]:
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            # A while-else body is reached when the loop condition is initially false.
            return self._selected_statements(node.orelse)
        return self.generic_visit(node)


def prune_statically_unreachable(tree: ast.Module) -> ast.Module:
    """Copy a Python module, excluding only provably unreachable statements."""
    return cast(ast.Module, _StaticUnreachablePruner().visit(copy.deepcopy(tree)))
