from __future__ import annotations

from collections.abc import Callable

from astscribe.parser import ImportTable, ParsedSource, SymbolTable

from .pytorch import SemanticOutput, analyze_pytorch
from .pytorch_experiment import analyze_pytorch_experiment

SemanticAnalyzer = Callable[[ParsedSource, ImportTable, SymbolTable], SemanticOutput]


class SemanticRegistry:
    def __init__(self) -> None:
        self._analyzers: dict[str, list[SemanticAnalyzer]] = {
            "pytorch": [analyze_pytorch, analyze_pytorch_experiment]
        }

    def analyzers(self) -> dict[str, tuple[SemanticAnalyzer, ...]]:
        return {
            framework: tuple(analyzers)
            for framework, analyzers in self._analyzers.items()
        }


DEFAULT_REGISTRY = SemanticRegistry()
