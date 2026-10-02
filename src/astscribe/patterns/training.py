from __future__ import annotations

from astscribe.sir import Operation, TrainingStep


def detect_training_step(operations: list[Operation]) -> TrainingStep | None:
    kinds = {operation.kind for operation in operations}
    signals = {
        "gradient_reset",
        "forward_pass",
        "loss_computation",
        "backward_pass",
        "parameter_update",
    }
    if len(kinds & signals) < 3 or not ({"backward_pass", "parameter_update"} <= kinds):
        return None
    return TrainingStep(
        gradient_reset="gradient_reset" in kinds,
        forward_pass="forward_pass" in kinds,
        loss_computation="loss_computation" in kinds,
        backward_pass="backward_pass" in kinds,
        parameter_update="parameter_update" in kinds,
    )
