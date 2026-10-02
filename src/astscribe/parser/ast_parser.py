from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSource:
    source: str
    tree: ast.Module
    lines: tuple[str, ...]

    def source_segment(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.source, node) or ""


def parse_source(source: str) -> ParsedSource:
    return ParsedSource(source=source, tree=ast.parse(source), lines=tuple(source.splitlines()))
