from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astscribe.api import _analyze_parsed
from astscribe.dependency import NotebookDependencyGraph, build_dependency_graph
from astscribe.diagnostics import NotebookDiagnostics, build_notebook_diagnostics
from astscribe.impact import (
    CellImpactSummary,
    NotebookImpactReport,
    build_impact_report,
    rank_cells_by_impact,
)
from astscribe.methodology import MethodologyReport, build_methodology_report
from astscribe.parser import (
    ImportTable,
    SymbolTable,
    build_import_table,
    build_symbol_table,
    parse_source,
)
from astscribe.pipeline import ExperimentPipeline, build_experiment_pipeline
from astscribe.sir import AnalysisResult
from astscribe.techniques import TechniqueFinding, detect_techniques, render_techniques


@dataclass(frozen=True)
class SkippedCell:
    """A notebook cell intentionally omitted from static Python analysis."""

    index: int
    reason: str


@dataclass
class NotebookAnalyzer:
    """Incrementally analyze notebook cells without executing them.

    Context flows only forward: each cell can use imports, literal values, and constructor
    bindings established by previous cells. Later cells never affect earlier analyses.
    """

    _imports: ImportTable = field(default_factory=ImportTable)
    _symbols: SymbolTable = field(default_factory=SymbolTable)
    _cells: list[str] = field(default_factory=list)
    _cell_indices: list[int] = field(default_factory=list)
    _results: list[AnalysisResult] = field(default_factory=list)
    _skipped_cells: list[SkippedCell] = field(default_factory=list)

    @classmethod
    def from_cells(cls, cells: list[str]) -> NotebookAnalyzer:
        analyzer = cls()
        for cell in cells:
            analyzer.add_cell(cell)
        return analyzer

    @classmethod
    def from_ipynb(
        cls,
        path: str | Path,
        *,
        skip_invalid_python: bool = True,
    ) -> NotebookAnalyzer:
        """Load code cells from a Jupyter ``.ipynb`` file using only the standard library."""

        notebook_path = Path(path)
        with notebook_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_ipynb_data(data, skip_invalid_python=skip_invalid_python)

    @classmethod
    def from_ipynb_data(
        cls,
        data: object,
        *,
        skip_invalid_python: bool = True,
    ) -> NotebookAnalyzer:
        """Build an analyzer from an already-decoded Jupyter notebook document."""

        if not isinstance(data, dict):
            raise ValueError("Invalid notebook: expected a top-level object.")

        cells = data.get("cells")
        if not isinstance(cells, list):
            raise ValueError("Invalid notebook: expected a top-level 'cells' list.")

        analyzer = cls()
        for notebook_index, cell in enumerate(cells):
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue

            raw_source = cell.get("source", "")
            if isinstance(raw_source, list):
                source = "".join(str(part) for part in raw_source)
            elif isinstance(raw_source, str):
                source = raw_source
            else:
                analyzer._skipped_cells.append(
                    SkippedCell(notebook_index, "code cell has an unsupported source representation")
                )
                continue

            if not source.strip():
                continue

            try:
                analyzer.add_cell(source, cell_index=notebook_index)
            except SyntaxError as exc:
                if not skip_invalid_python:
                    raise
                reason = f"not valid Python for static AST analysis: {exc.msg}"
                analyzer._skipped_cells.append(SkippedCell(notebook_index, reason))

        return analyzer

    def add_cell(self, source: str, *, cell_index: int | None = None) -> AnalysisResult:
        context_index = len(self._cells) if cell_index is None else cell_index
        parsed = parse_source(source, cell=context_index)

        local_imports = build_import_table(parsed.tree)
        self._imports.aliases.update(local_imports.aliases)
        self._symbols = build_symbol_table(
            parsed.tree,
            self._imports,
            base=self._symbols,
            source=source,
            cell=context_index,
        )

        result = _analyze_parsed(parsed, self._imports, self._symbols)
        self._cells.append(source)
        self._cell_indices.append(context_index)
        self._results.append(result)
        return result

    def analyze_cell(self, index: int) -> AnalysisResult:
        return self._results[index]

    def explain_cell(self, index: int, style: str = "scientific") -> str:
        return self.analyze_cell(index).render(style)

    def methodology(self) -> MethodologyReport:
        """Build a notebook-level, evidence-backed scientific Methods report."""

        return build_methodology_report(self.results)

    def render_methodology(self, *, include_evidence: bool = False) -> str:
        """Render the notebook methodology as deterministic Markdown."""

        return self.methodology().render(include_evidence=include_evidence)

    def pipeline(self) -> ExperimentPipeline:
        """Build a structured experiment pipeline from evidence-backed operations."""

        return build_experiment_pipeline(self.results)

    def render_pipeline(self) -> str:
        """Render the reconstructed experiment pipeline as a compact text diagram."""

        return self.pipeline().render()

    def techniques(self) -> tuple[TechniqueFinding, ...]:
        """Detect composite notebook techniques that require evidence across cells."""

        return detect_techniques(self.results)

    def render_techniques(self) -> str:
        """Render composite technique findings with their supporting source evidence."""

        return render_techniques(self.techniques())

    def dependency_graph(self) -> NotebookDependencyGraph:
        """Build a static symbol-flow graph between analyzed notebook cells."""

        cells = tuple(zip(self._cell_indices, self._cells, strict=True))
        return build_dependency_graph(cells)

    def render_dependency_graph(self) -> str:
        """Render cross-cell symbol dependencies as deterministic text."""

        return self.dependency_graph().render()

    def dependency_dot(self) -> str:
        """Export the dependency graph as Graphviz DOT without requiring Graphviz."""

        return self.dependency_graph().to_dot()

    def diagnostics(self) -> NotebookDiagnostics:
        """Build conservative notebook diagnostics from the dependency graph."""

        return build_notebook_diagnostics(self.dependency_graph())

    def render_diagnostics(self) -> str:
        """Render dependency-aware notebook diagnostics as deterministic text."""

        return self.diagnostics().render()

    def impact(self, cell: int) -> NotebookImpactReport:
        """Compute direct and transitive static impact for one notebook cell."""

        return build_impact_report(self.dependency_graph(), cell)

    def render_impact(self, cell: int) -> str:
        """Render the static blast radius and propagation paths for one cell."""

        return self.impact(cell).render()

    def impact_ranking(self) -> tuple[CellImpactSummary, ...]:
        """Rank analyzed cells by downstream static blast radius."""

        return rank_cells_by_impact(self.dependency_graph())

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def cell_indices(self) -> tuple[int, ...]:
        """Notebook cell indices corresponding to analyzed code cells."""

        return tuple(self._cell_indices)

    @property
    def skipped_cells(self) -> tuple[SkippedCell, ...]:
        """Cells skipped because they could not be represented as static Python AST."""

        return tuple(self._skipped_cells)

    @property
    def results(self) -> tuple[AnalysisResult, ...]:
        return tuple(self._results)
