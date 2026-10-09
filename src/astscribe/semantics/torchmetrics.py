"""Static TorchMetrics functional evaluation and stateful metric usage."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from astscribe.parser import ImportTable, ParsedSource, SymbolTable
from astscribe.sir import Claim, Evidence, EvidenceLevel, Operation


@dataclass(frozen=True)
class SemanticOutput:
    operations: list[Operation]
    claims: list[Claim]


_METRICS = {
    "Accuracy", "Precision", "Recall", "F1Score", "FBetaScore", "AUROC",
    "AveragePrecision", "ConfusionMatrix", "PrecisionRecallCurve",
    "Specificity", "MatthewsCorrCoef", "CohenKappa", "CalibrationError",
    "JaccardIndex", "StatScores", "HammingDistance", "HingeLoss",
    "MeanMetric", "SumMetric", "MaxMetric", "MinMetric",
    "MeanSquaredError", "MeanAbsoluteError", "MeanAbsolutePercentageError",
    "R2Score", "CosineSimilarity", "PearsonCorrCoef", "SpearmanCorrCoef",
    "StructuralSimilarityIndexMeasure", "PeakSignalNoiseRatio",
    "MultiScaleStructuralSimilarityIndexMeasure", "LearnedPerceptualImagePatchSimilarity",
    "FrechetInceptionDistance", "KernelInceptionDistance",
    "MeanAveragePrecision", "WordErrorRate", "CharErrorRate", "BLEUScore",
    "SacreBLEUScore", "BERTScore", "Perplexity", "RetrievalMAP",
    "RetrievalMRR", "RetrievalPrecision", "RetrievalRecall",
    "MetricCollection",
}
_PREFIXES = {"Binary", "Multiclass", "Multilabel"}
_FUNCTIONAL = {
    "accuracy", "precision", "recall", "f1_score", "fbeta_score",
    "auroc", "average_precision", "confusion_matrix", "precision_recall_curve",
    "specificity", "matthews_corrcoef", "cohen_kappa",
    "calibration_error", "jaccard_index", "stat_scores", "hamming_distance",
    "mean_squared_error", "mean_absolute_error", "r2_score",
    "structural_similarity_index_measure", "peak_signal_noise_ratio",
    "mean_average_precision", "word_error_rate", "char_error_rate",
    "bleu_score", "perplexity",
}


def is_metric_constructor(path: str) -> bool:
    if not path.startswith("torchmetrics."):
        return False
    parts = path.split(".")
    if len(parts) not in {2, 3} or "functional" in parts:
        return False
    name = parts[-1]
    if name in _METRICS:
        return True
    return any(
        name.startswith(prefix) and name[len(prefix):] in _METRICS
        for prefix in _PREFIXES
    )


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _keyword_values(node: ast.Call, symbols: SymbolTable) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is not None:
            value = symbols.resolve_constant(keyword.value)
            if value is not None:
                data[keyword.arg] = value
    return data


def analyze_torchmetrics(
    parsed: ParsedSource, imports: ImportTable, symbols: SymbolTable
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []
    for node in ast.walk(parsed.tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        if dotted is None:
            continue
        path = imports.resolve_dotted(dotted)
        kind: str | None = None
        text = ""
        if path.startswith("torchmetrics.functional."):
            name = path.rsplit(".", 1)[-1]
            if name in _FUNCTIONAL and len(path.split(".")) in {3, 4}:
                kind = "metric_computation"
                text = f"The TorchMetrics functional operation {name} is invoked."
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            constructor = symbols.resolve_constructor(node.func.value.id)
            if constructor and is_metric_constructor(constructor):
                if node.func.attr == "update":
                    kind = "metric_update"
                    text = "A configured TorchMetrics metric accumulator is updated."
                elif node.func.attr == "compute":
                    kind = "metric_computation"
                    text = "A configured TorchMetrics metric result is requested."
                elif node.func.attr == "reset":
                    kind = "metric_reset"
                    text = "A configured TorchMetrics metric accumulator is reset."
        if kind is None:
            continue
        ev = Evidence(
            level=EvidenceLevel.E3, rule=f"torchmetrics.{kind}",
            source=parsed.source_segment(node),
            line_start=getattr(node, "lineno", None),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
            cell=parsed.cell,
        )
        operations.append(
            Operation(kind, "torchmetrics", subject=dotted, attributes=_keyword_values(node, symbols), evidence=ev)
        )
        claims.append(Claim(text, ev))
    return SemanticOutput(operations, claims)
