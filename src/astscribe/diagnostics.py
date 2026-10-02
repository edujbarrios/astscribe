from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from astscribe.dependency import NotebookDependencyGraph

DiagnosticSeverity = Literal["info", "warning"]


@dataclass(frozen=True)
class NotebookDiagnostic:
    code: str
    severity: DiagnosticSeverity
    message: str
    cell: int
    symbol: str
    related_cell: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NotebookDiagnostics:
    items: tuple[NotebookDiagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}

    def by_code(self, code: str) -> tuple[NotebookDiagnostic, ...]:
        return tuple(item for item in self.items if item.code == code)

    def render(self) -> str:
        if not self.items:
            return "No supported notebook dependency diagnostics were detected."

        lines = ["# Notebook diagnostics", ""]
        for item in self.items:
            label = item.severity.upper()
            lines.append(f"- [{label}] cell {item.cell}: {item.message} (`{item.code}`)")
        return "\n".join(lines)


def _definitions_by_symbol(graph: NotebookDependencyGraph) -> dict[str, tuple[int, ...]]:
    cells: dict[str, list[int]] = {}
    for node in graph.nodes:
        for symbol in node.defines:
            cells.setdefault(symbol, []).append(node.cell)
    return {symbol: tuple(values) for symbol, values in cells.items()}


def _next_definition(
    definitions: dict[str, tuple[int, ...]],
    symbol: str,
    cell: int,
) -> int | None:
    return next(
        (definition_cell for definition_cell in definitions.get(symbol, ()) if definition_cell > cell),
        None,
    )


def _definition_consumed_before_redefinition(
    graph: NotebookDependencyGraph,
    symbol: str,
    producer_cell: int,
    redefinition_cell: int,
) -> bool:
    return any(
        edge.producer_cell == producer_cell
        and edge.consumer_cell <= redefinition_cell
        and symbol in edge.symbols
        for edge in graph.edges
    )


def build_notebook_diagnostics(graph: NotebookDependencyGraph) -> NotebookDiagnostics:
    definitions = _definitions_by_symbol(graph)
    diagnostics: list[NotebookDiagnostic] = []

    for node in graph.nodes:
        for symbol in node.unresolved_reads:
            later_cell = _next_definition(definitions, symbol, node.cell)
            if later_cell is not None:
                diagnostics.append(
                    NotebookDiagnostic(
                        code="dependency.forward_reference",
                        severity="warning",
                        message=(
                            f"`{symbol}` is read before its next supported definition in "
                            f"cell {later_cell}; source-order analysis cannot resolve this read."
                        ),
                        cell=node.cell,
                        symbol=symbol,
                        related_cell=later_cell,
                    )
                )
            else:
                diagnostics.append(
                    NotebookDiagnostic(
                        code="dependency.unresolved_symbol",
                        severity="warning",
                        message=(
                            f"`{symbol}` is read without a supported prior notebook definition; "
                            "it may come from external or hidden kernel state."
                        ),
                        cell=node.cell,
                        symbol=symbol,
                    )
                )

    for redefinition in graph.redefinitions:
        diagnostics.append(
            NotebookDiagnostic(
                code="dependency.symbol_redefinition",
                severity="info",
                message=(
                    f"`{redefinition.symbol}` replaces the definition from cell "
                    f"{redefinition.previous_cell}."
                ),
                cell=redefinition.new_cell,
                symbol=redefinition.symbol,
                related_cell=redefinition.previous_cell,
            )
        )

        if not _definition_consumed_before_redefinition(
            graph,
            redefinition.symbol,
            redefinition.previous_cell,
            redefinition.new_cell,
        ):
            diagnostics.append(
                NotebookDiagnostic(
                    code="dependency.overwritten_before_cross_cell_use",
                    severity="info",
                    message=(
                        f"The definition of `{redefinition.symbol}` in cell "
                        f"{redefinition.previous_cell} is replaced in cell "
                        f"{redefinition.new_cell} before any later analyzed cell consumes it."
                    ),
                    cell=redefinition.new_cell,
                    symbol=redefinition.symbol,
                    related_cell=redefinition.previous_cell,
                )
            )

    diagnostics.sort(
        key=lambda item: (
            item.cell,
            item.code,
            item.symbol,
            item.related_cell if item.related_cell is not None else -1,
        )
    )
    return NotebookDiagnostics(items=tuple(diagnostics))
