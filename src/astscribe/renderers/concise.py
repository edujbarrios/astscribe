from __future__ import annotations

from astscribe.sir import AnalysisResult


def render_concise(result: AnalysisResult) -> str:
    if result.training_step:
        return "Performs a PyTorch gradient-based training step with backward propagation and a parameter update."
    if result.inference:
        return "Performs PyTorch inference using evaluation-oriented execution semantics."
    if result.claims:
        return " ".join(claim.text for claim in result.claims[:2])
    return "No supported PyTorch semantics were identified in the analyzed source."
