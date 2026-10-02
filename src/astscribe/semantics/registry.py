from __future__ import annotations

from collections.abc import Callable

from astscribe.parser import ImportTable, ParsedSource, SymbolTable

from .pytorch import SemanticOutput, analyze_pytorch

SemanticAnalyzer = Callable[[ParsedSource, ImportTable, SymbolTable], SemanticOutput]


class SemanticRegistry:
    def __init__(self) -> None:
        self._analyzers: dict[str, SemanticAnalyzer] = {"pytorch": analyze_pytorch}

    def analyzers(self) -> dict[str, SemanticAnalyzer]:
        return dict(self._analyzers)


DEFAULT_REGISTRY = SemanticRegistry()
