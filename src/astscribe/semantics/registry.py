from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from astscribe.parser import ImportTable, ParsedSource, SymbolTable
from astscribe.sir import Claim, Operation

from .pytorch import analyze_pytorch
from .pytorch_experiment import analyze_pytorch_experiment
from .pytorch_reproducibility import analyze_pytorch_reproducibility
from .transformers import analyze_transformers


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
            "pytorch": [
                cast(SemanticAnalyzer, analyze_pytorch),
                cast(SemanticAnalyzer, analyze_pytorch_experiment),
                cast(SemanticAnalyzer, analyze_pytorch_reproducibility),
            ],
            "transformers": [cast(SemanticAnalyzer, analyze_transformers)],
        }

    def analyzers(self) -> dict[str, tuple[SemanticAnalyzer, ...]]:
        return {
            framework: tuple(analyzers)
            for framework, analyzers in self._analyzers.items()
        }


DEFAULT_REGISTRY = SemanticRegistry()
