from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from astscribe.parser import ImportTable, ParsedSource, SymbolTable
from astscribe.sir import Claim, Operation

from .datasets import analyze_datasets
from .multimodal import analyze_multimodal
from .peft import analyze_peft
from .pytorch import analyze_pytorch
from .pytorch_ecosystem import analyze_pytorch_ecosystem
from .pytorch_experiment import analyze_pytorch_experiment
from .pytorch_networks import analyze_pytorch_networks
from .pytorch_reproducibility import analyze_pytorch_reproducibility
from .torchaudio import analyze_torchaudio
from .torchmetrics import analyze_torchmetrics
from .torchvision import analyze_torchvision
from .transformers import analyze_transformers
from .transformers_models import analyze_transformers_models
from .transformers_quantization import analyze_transformers_quantization


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
                cast(SemanticAnalyzer, analyze_pytorch_ecosystem),
                cast(SemanticAnalyzer, analyze_pytorch_networks),
                cast(SemanticAnalyzer, analyze_pytorch_reproducibility),
            ],
            "transformers": [
                cast(SemanticAnalyzer, analyze_transformers),
                cast(SemanticAnalyzer, analyze_transformers_models),
                cast(SemanticAnalyzer, analyze_transformers_quantization),
            ],
            "datasets": [cast(SemanticAnalyzer, analyze_datasets)],
            "peft": [cast(SemanticAnalyzer, analyze_peft)],
            "multimodal": [cast(SemanticAnalyzer, analyze_multimodal)],
            "torchaudio": [cast(SemanticAnalyzer, analyze_torchaudio)],
            "torchvision": [cast(SemanticAnalyzer, analyze_torchvision)],
            "torchmetrics": [cast(SemanticAnalyzer, analyze_torchmetrics)],
        }

    def analyzers(self) -> dict[str, tuple[SemanticAnalyzer, ...]]:
        return {
            framework: tuple(analyzers)
            for framework, analyzers in self._analyzers.items()
        }


DEFAULT_REGISTRY = SemanticRegistry()
