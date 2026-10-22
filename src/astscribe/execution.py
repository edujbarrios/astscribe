"""Static audit of persisted notebook execution counters.

These counters are notebook metadata, not verified kernel execution traces.
The audit never executes code or infers that saved outputs are trustworthy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Severity = Literal["info", "warning"]


@dataclass(frozen=True)
class ExecutionCell:
    cell: int
    execution_count: int | None


@dataclass(frozen=True)
class ExecutionFinding:
    code: str
    severity: Severity
    cell: int
    message: str
    related_cell: int | None = None


@dataclass(frozen=True)
class NotebookExecutionReport:
    cells: tuple[ExecutionCell, ...]
    findings: tuple[ExecutionFinding, ...]

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "warning" for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "notebook_execution_metadata",
            "schema_version": 1,
            "verification": "saved_metadata_only",
            "cells": [asdict(item) for item in self.cells],
            "findings": [asdict(item) for item in self.findings],
            "warning_count": self.warning_count,
        }

    def render(self) -> str:
        lines = [
            "# Notebook execution metadata",
            "",
            "Saved execution counters are not a verified kernel execution trace.",
            f"Analyzed code-cell metadata: {len(self.cells)}; warnings: {self.warning_count}.",
        ]
        if not self.findings:
            lines.append("No inconsistencies were detected in saved execution counters.")
        for item in self.findings:
            lines.append(
                f"- [{item.severity.upper()}] cell {item.cell}: "
                f"{item.message} (`{item.code}`)"
            )
        return "\n".join(lines)


def audit_execution_counts(counts: dict[int, int | None]) -> NotebookExecutionReport:
    """Check counters in source order without pretending they prove actual run order."""
    cells = tuple(
        ExecutionCell(cell=cell, execution_count=count)
        for cell, count in sorted(counts.items())
    )
    findings: list[ExecutionFinding] = []
    seen: dict[int, int] = {}
    previous: ExecutionCell | None = None

    for cell in cells:
        count = cell.execution_count
        if count is None:
            findings.append(
                ExecutionFinding(
                    code="execution.no_saved_count",
                    severity="info",
                    cell=cell.cell,
                    message="No positive integer execution count is saved for this code cell.",
                )
            )
            continue
        if count in seen:
            findings.append(
                ExecutionFinding(
                    code="execution.duplicate_count",
                    severity="warning",
                    cell=cell.cell,
                    related_cell=seen[count],
                    message=(
                        f"Execution count {count} is also recorded for cell {seen[count]}; "
                        "saved execution metadata is ambiguous."
                    ),
                )
            )
        else:
            seen[count] = cell.cell

        if previous is not None and previous.execution_count is not None:
            if count < previous.execution_count:
                findings.append(
                    ExecutionFinding(
                        code="execution.nonmonotonic_count",
                        severity="warning",
                        cell=cell.cell,
                        related_cell=previous.cell,
                        message=(
                            f"Saved count {count} follows count {previous.execution_count} "
                            f"in earlier source cell {previous.cell}; saved order differs "
                            "from source order."
                        ),
                    )
                )
        previous = cell

    return NotebookExecutionReport(cells=cells, findings=tuple(findings))
