"""Conservative, evidence-backed notebook CI preflight.

Combines distinct reports without implying a notebook was safely executed,
the listed dependencies were installed, or saved execution metadata is truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from astscribe.diagnostics import NotebookDiagnostics
from astscribe.execution import NotebookExecutionReport
from astscribe.import_inventory import NotebookImportReport

Severity = Literal["info", "warning"]


@dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: Severity
    cell: int
    message: str
    related_cell: int | None = None


@dataclass(frozen=True)
class NotebookAuditReport:
    analyzed_cells: int
    skipped_cells: int
    findings: tuple[AuditFinding, ...]
    non_stdlib_roots: tuple[str, ...]

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "warning" for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "notebook_static_audit",
            "schema_version": 1,
            "verification": "source_and_saved_metadata_only",
            "analyzed_cells": self.analyzed_cells,
            "skipped_cells": self.skipped_cells,
            "warning_count": self.warning_count,
            "findings": [asdict(item) for item in self.findings],
            "non_stdlib_roots": list(self.non_stdlib_roots),
        }

    def render(self) -> str:
        lines = [
            "# Notebook static audit",
            "",
            f"Analyzed code cells: {self.analyzed_cells}. "
            f"Skipped cells: {self.skipped_cells}. Warnings: {self.warning_count}.",
            "Results describe static source and saved metadata, not verified runtime behavior.",
        ]
        if not self.findings:
            lines.append("No issues detected by the currently supported audit checks.")
        for finding in self.findings:
            lines.append(
                f"- [{finding.severity.upper()}] cell {finding.cell}: "
                f"{finding.message} (`{finding.code}`)"
            )
        if self.non_stdlib_roots:
            lines.append(
                "Non-standard-library import roots (not verified installations): "
                + ", ".join(self.non_stdlib_roots)
            )
        return "\n".join(lines)


def build_notebook_audit(
    *,
    analyzed_cells: int,
    skipped_cells: tuple[tuple[int, str], ...],
    diagnostics: NotebookDiagnostics,
    execution: NotebookExecutionReport,
    imports: NotebookImportReport,
) -> NotebookAuditReport:
    findings: list[AuditFinding] = []

    for skipped_cell, reason in skipped_cells:
        findings.append(
            AuditFinding(
                code="audit.skipped_code",
                severity="warning",
                cell=skipped_cell,
                message=(
                    "Code cell was not analyzed; its effects on notebook state "
                    f"are unknown ({reason})."
                ),
            )
        )

    for item in diagnostics.items:
        findings.append(
            AuditFinding(
                code=item.code,
                severity=item.severity,
                cell=item.cell,
                message=item.message,
                related_cell=item.related_cell,
            )
        )

    for item in execution.findings:
        findings.append(
            AuditFinding(
                code=item.code,
                severity=item.severity,
                cell=item.cell,
                message=item.message,
                related_cell=item.related_cell,
            )
        )

    findings.sort(
        key=lambda item: (
            item.cell,
            item.code,
            item.related_cell if item.related_cell is not None else -1,
        )
    )
    return NotebookAuditReport(
        analyzed_cells=analyzed_cells,
        skipped_cells=len(skipped_cells),
        findings=tuple(findings),
        non_stdlib_roots=tuple(imports.to_dict()["non_stdlib_roots"]),
    )
