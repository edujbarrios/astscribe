from __future__ import annotations

from astscribe.sir import AnalysisResult

from .concise import render_concise
from .educational import render_educational
from .scientific import render_scientific


def render(result: AnalysisResult, style: str = "scientific") -> str:
    if style == "concise":
        return render_concise(result)
    if style == "educational":
        return render_educational(result)
    if style == "scientific":
        return render_scientific(result)
    raise ValueError(f"Unsupported rendering style: {style!r}")


__all__ = ["render"]
