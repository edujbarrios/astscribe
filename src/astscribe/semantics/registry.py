from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from astscribe.parser import ImportTable, ParsedSource, SymbolTable
from astscribe.sir import Claim, Operation

from .pytorch import analyze_pytorch
from .pytorch_experiment import analyze_pytorch_experiment


class SemanticOutputProtocol(Protocol):
    operations: list[Operation]
    claims: list[Claim]


SemanticAnalyzer = Callable[
    [ParsedSource, ImportTable, SymbolTable],
    SemanticOutputProtocol,
]


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
