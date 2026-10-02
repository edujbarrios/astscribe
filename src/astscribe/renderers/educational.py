from __future__ import annotations

from astscribe.sir import AnalysisResult


def render_educational(result: AnalysisResult) -> str:
    if not result.claims:
        return "ASTScribe could not identify any supported PyTorch operations in this code."
    lines = ["Step-by-step explanation", ""]
    lines.extend(f"{index}. {claim.text}" for index, claim in enumerate(result.claims, start=1))
    return "\n".join(lines)
