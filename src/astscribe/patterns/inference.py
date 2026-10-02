from __future__ import annotations

from astscribe.sir import InferenceProcedure, Operation


def detect_inference(operations: list[Operation]) -> InferenceProcedure | None:
    kinds = {operation.kind for operation in operations}
    forward = "forward_pass" in kinds
    evaluation = "evaluation_mode" in kinds
    no_grad = "gradient_tracking_disabled" in kinds
    inference_mode = "inference_mode" in kinds
    if not forward or not (evaluation or no_grad or inference_mode):
        return None
    return InferenceProcedure(
        evaluation_mode=evaluation,
        gradient_tracking_disabled=no_grad or inference_mode,
        inference_mode=inference_mode,
        forward_pass=forward,
    )
