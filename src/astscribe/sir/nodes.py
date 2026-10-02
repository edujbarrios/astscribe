from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class EvidenceLevel(str, Enum):
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"


@dataclass(frozen=True)
class Evidence:
    level: EvidenceLevel
    rule: str
    source: str
    line_start: int | None
    line_end: int | None
    cell: int | None = None


@dataclass(frozen=True)
class Claim:
    text: str
    evidence: Evidence

    @property
    def evidence_level(self) -> str:
        return self.evidence.level.value

    @property
    def line_start(self) -> int | None:
        return self.evidence.line_start

    @property
    def line_end(self) -> int | None:
        return self.evidence.line_end

    @property
    def rule(self) -> str:
        return self.evidence.rule


@dataclass(frozen=True)
class Operation:
    kind: str
    framework: str
    subject: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence: Evidence | None = None


@dataclass(frozen=True)
class TrainingStep:
    framework: Literal["pytorch"] = "pytorch"
    gradient_reset: bool = False
    forward_pass: bool = False
    loss_computation: bool = False
    backward_pass: bool = False
    parameter_update: bool = False


@dataclass(frozen=True)
class InferenceProcedure:
    framework: Literal["pytorch"] = "pytorch"
    evaluation_mode: bool = False
    gradient_tracking_disabled: bool = False
    inference_mode: bool = False
    forward_pass: bool = False


@dataclass
class AnalysisResult:
    source: str
    frameworks: list[str] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    training_step: TrainingStep | None = None
    inference: InferenceProcedure | None = None

    def render(self, style: str = "scientific") -> str:
        from astscribe.renderers import render

        return render(self, style=style)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
