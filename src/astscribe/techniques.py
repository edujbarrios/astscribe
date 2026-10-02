from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from astscribe.sir import AnalysisResult, Evidence, Operation


@dataclass(frozen=True)
class TechniqueFinding:
    key: str
    title: str
    summary: str
    evidence: tuple[Evidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [f"## {self.title}", "", self.summary]
        if self.evidence:
            lines.extend(["", "Evidence:"])
            for item in self.evidence:
                location: list[str] = []
                if item.cell is not None:
                    location.append(f"cell {item.cell}")
                if item.line_start is not None:
                    location.append(f"line {item.line_start}")
                where = ", ".join(location) if location else "source location unavailable"
                lines.append(f"- {where}: `{item.rule}`")
        return "\n".join(lines)


def _first_operation(
    operations: tuple[Operation, ...],
    kind: str,
    *,
    predicate: object | None = None,
) -> Operation | None:
    for operation in operations:
        if operation.kind != kind:
            continue
        if predicate is None:
            return operation
        if callable(predicate) and predicate(operation):
            return operation
    return None


def detect_techniques(results: tuple[AnalysisResult, ...]) -> tuple[TechniqueFinding, ...]:
    operations = tuple(operation for result in results for operation in result.operations)
    findings: list[TechniqueFinding] = []

    quantized_4bit = _first_operation(
        operations,
        "quantized_model_load",
        predicate=lambda operation: operation.attributes.get("bits") == 4,
    )
    lora_application = _first_operation(
        operations,
        "adapter_application",
        predicate=lambda operation: operation.attributes.get("adapter_config") == "LoraConfig",
    )
    if lora_application is None:
        lora_config = _first_operation(
            operations,
            "adapter_configuration",
            predicate=lambda operation: operation.subject == "LoRA",
        )
        generic_application = _first_operation(operations, "adapter_application")
        if lora_config is not None and generic_application is not None:
            lora_application = generic_application

    training = _first_operation(operations, "trainer_train")
    if training is None:
        training = _first_operation(operations, "parameter_update")

    if quantized_4bit is not None and lora_application is not None and training is not None:
        evidence: list[Evidence] = []
        for operation in (quantized_4bit, lora_application):
            if operation.evidence is not None:
                evidence.append(operation.evidence)
        kbit_prep = _first_operation(operations, "kbit_training_preparation")
        if kbit_prep is not None and kbit_prep.evidence is not None:
            evidence.append(kbit_prep.evidence)
        if training.evidence is not None:
            evidence.append(training.evidence)

        findings.append(
            TechniqueFinding(
                key="qlora",
                title="QLoRA",
                summary=(
                    "The notebook provides source-level evidence for QLoRA-style fine-tuning: "
                    "a model is loaded with explicit 4-bit bitsandbytes quantization, LoRA is "
                    "applied through PEFT, and a training procedure is invoked."
                ),
                evidence=tuple(evidence),
            )
        )

    return tuple(findings)


def render_techniques(findings: tuple[TechniqueFinding, ...]) -> str:
    if not findings:
        return "No composite training technique could be established from the available evidence."
    blocks = ["# Detected techniques"]
    for finding in findings:
        blocks.extend(["", finding.render()])
    return "\n".join(blocks)
