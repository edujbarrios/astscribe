from __future__ import annotations

from astscribe.sir import AnalysisResult, EvidenceLevel

_ALLOWED = {EvidenceLevel.E1, EvidenceLevel.E2, EvidenceLevel.E3}


def render_scientific(result: AnalysisResult) -> str:
    claims = [claim for claim in result.claims if claim.evidence.level in _ALLOWED]
    if not claims:
        return "No evidence-backed PyTorch methodology could be inferred from the analyzed source."

    if result.training_step:
        heading = "Training procedure"
    elif result.inference:
        heading = "Inference procedure"
    else:
        heading = "Methodological description"

    paragraphs: list[str] = [heading, ""]
    if result.training_step:
        step = result.training_step
        statements: list[str] = []
        if step.gradient_reset:
            statements.append("previously accumulated gradients are reset")
        if step.forward_pass:
            statements.append("a forward pass is executed")
        if step.loss_computation:
            statements.append("the optimization objective is evaluated")
        if step.backward_pass:
            statements.append("gradients are computed through backward propagation")
        if step.parameter_update:
            statements.append("the optimizer updates model parameters")
        if statements:
            paragraphs.append(
                "The analyzed code implements a PyTorch optimization procedure in which "
                + ", ".join(statements[:-1])
                + (f", and {statements[-1]}." if len(statements) > 1 else f"{statements[-1]}.")
            )
            paragraphs.append("")

    paragraphs.extend(claim.text for claim in claims)
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph != "")
