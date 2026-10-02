from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from astscribe.sir import AnalysisResult, Claim, EvidenceLevel

_ALLOWED = {EvidenceLevel.E1, EvidenceLevel.E2, EvidenceLevel.E3}


@dataclass(frozen=True)
class MethodologySection:
    title: str
    claims: tuple[Claim, ...]


@dataclass(frozen=True)
class MethodologyReport:
    sections: tuple[MethodologySection, ...]

    def render(self, *, include_evidence: bool = False) -> str:
        lines: list[str] = ["# Methods"]
        evidence: list[Claim] = []

        for section in self.sections:
            if not section.claims:
                continue
            lines.extend(["", f"## {section.title}", ""])
            lines.append(" ".join(claim.text for claim in section.claims))
            evidence.extend(section.claims)

        if include_evidence and evidence:
            lines.extend(["", "## Evidence", ""])
            for claim in evidence:
                location = _format_location(claim)
                source = claim.evidence.source.strip().replace("\n", " ")
                lines.append(
                    f"- [{claim.evidence_level}] {location} — `{source}` "
                    f"(`{claim.rule}`)"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_methodology_report(results: tuple[AnalysisResult, ...]) -> MethodologyReport:
    buckets: dict[str, list[Claim]] = {
        "Reproducibility": [],
        "Dataset": [],
        "Preprocessing and augmentation": [],
        "Data loading": [],
        "Execution environment": [],
        "Model architecture": [],
        "Model and objective": [],
        "Optimization": [],
        "Numerical precision": [],
        "Training procedure": [],
        "Evaluation and inference": [],
        "Metrics": [],
        "Checkpointing": [],
        "Other methodology": [],
    }
    seen: set[tuple[str, str, int | None, int | None]] = set()

    for result in results:
        for claim in result.claims:
            if claim.evidence.level not in _ALLOWED:
                continue
            if claim.rule == "context.optimizer_binding":
                continue

            key = (
                claim.text,
                claim.rule,
                claim.evidence.cell,
                claim.evidence.line_start,
            )
            if key in seen:
                continue
            seen.add(key)

            section = _section_for_claim(claim, result)
            buckets[section].append(claim)

    sections = tuple(
        MethodologySection(title=title, claims=tuple(claims))
        for title, claims in buckets.items()
        if claims
    )
    return MethodologyReport(sections=sections)


def _section_for_claim(claim: Claim, result: AnalysisResult) -> str:
    rule = claim.rule

    if rule in {
        "pytorch.manual_seed",
        "pytorch.cuda_manual_seed_all",
        "pytorch.use_deterministic_algorithms",
        "pytorch.cudnn_deterministic",
        "pytorch.cudnn_benchmark",
    }:
        return "Reproducibility"
    if rule in {"pytorch.dataset_configuration", "pytorch.dataset_split"}:
        return "Dataset"
    if rule == "pytorch.preprocessing_transform":
        return "Preprocessing and augmentation"
    if rule == "pytorch.dataloader":
        return "Data loading"
    if rule in {"pytorch.device_configuration", "pytorch.device_transfer"}:
        return "Execution environment"
    if rule in {
        "pytorch.model_configuration",
        "pytorch.model_head_replacement",
        "pytorch.parameter_freeze",
    }:
        return "Model architecture"
    if rule == "pytorch.loss_configuration":
        return "Model and objective"
    if rule in {
        "pytorch.optimizer_configuration",
        "pytorch.scheduler_configuration",
        "pytorch.scheduler_step",
        "pytorch.gradient_clipping",
    }:
        return "Optimization"
    if rule in {
        "pytorch.autocast",
        "pytorch.grad_scaler",
        "pytorch.grad_scaler_scale",
        "pytorch.grad_scaler_step",
        "pytorch.grad_scaler_update",
    }:
        return "Numerical precision"
    if rule in {
        "pytorch.metric_configuration",
        "pytorch.prediction_argmax",
        "pytorch.softmax",
    }:
        return "Metrics"
    if rule in {
        "pytorch.checkpoint_save",
        "pytorch.checkpoint_load",
        "pytorch.load_state_dict",
    }:
        return "Checkpointing"
    if rule in {
        "pytorch.model_eval",
        "pytorch.no_grad",
        "pytorch.inference_mode",
    }:
        return "Evaluation and inference"
    if rule == "pytorch.forward_pass" and result.inference is not None:
        return "Evaluation and inference"
    if rule in {
        "pytorch.model_train",
        "pytorch.optimizer_zero_grad",
        "pytorch.backward",
        "pytorch.optimizer_step",
        "pytorch.loss_computation",
        "pytorch.epoch_loop",
    }:
        return "Training procedure"
    if rule == "pytorch.forward_pass" and result.training_step is not None:
        return "Training procedure"
    return "Other methodology"


def _format_location(claim: Claim) -> str:
    parts: list[str] = []
    if claim.evidence.cell is not None:
        parts.append(f"cell {claim.evidence.cell}")
    if claim.evidence.line_start is not None:
        if claim.evidence.line_end not in {None, claim.evidence.line_start}:
            parts.append(f"lines {claim.evidence.line_start}-{claim.evidence.line_end}")
        else:
            parts.append(f"line {claim.evidence.line_start}")
    return ", ".join(parts) if parts else "source location unavailable"
