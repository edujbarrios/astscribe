from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from astscribe.api import _analyze_parsed
from astscribe.dependency import NotebookDependencyGraph, build_dependency_graph
from astscribe.diagnostics import NotebookDiagnostics, build_notebook_diagnostics
from astscribe.impact import (
    CellImpactSummary,
    NotebookImpactReport,
    build_impact_report,
    rank_cells_by_impact,
    render_impact_ranking,
)
from astscribe.methodology import MethodologyReport, build_methodology_report
from astscribe.parser import (
    ImportTable,
    SymbolTable,
    build_symbol_table,
    parse_source,
    update_import_table,
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
    _next_cell_index: int = 0

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
        with notebook_path.open("r", encoding="utf-8-sig") as handle:
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
            if not isinstance(cell, dict):
                reason = "cell is not an object"
            elif cell.get("cell_type") in {"markdown", "raw"}:
                continue
            elif cell.get("cell_type") != "code":
                reason = f"unsupported cell_type {cell.get('cell_type')!r}"
            elif "source" not in cell:
                reason = "code cell is missing a source field"
            else:
                reason = ""

            if reason:
                if not skip_invalid_python:
                    raise ValueError(f"Invalid notebook cell {notebook_index}: {reason}.")
                analyzer._skipped_cells.append(SkippedCell(notebook_index, reason))
                continue

            raw_source = cell["source"]
            if isinstance(raw_source, list) and all(
                isinstance(part, str) for part in raw_source
            ):
                source = "".join(raw_source)
            elif isinstance(raw_source, str):
                source = raw_source
            else:
                reason = "code cell has an unsupported source representation"
                if not skip_invalid_python:
                    raise ValueError(f"Invalid notebook code cell {notebook_index}: {reason}.")
                analyzer._skipped_cells.append(SkippedCell(notebook_index, reason))
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

        # Keep the full notebook coordinate space, including trailing markdown,
        # empty code, and syntax-skipped cells that have no AnalysisResult.
        analyzer._next_cell_index = len(cells)
        return analyzer

    def add_cell(self, source: str, *, cell_index: int | None = None) -> AnalysisResult:
        if cell_index is None:
            context_index = self._next_cell_index
        else:
            if type(cell_index) is not int or cell_index < 0:
                raise ValueError("cell_index must be a non-negative integer.")
            context_index = cell_index

        if context_index < self._next_cell_index:
            raise ValueError("cell_index must not reuse an existing notebook cell index.")

        parsed = parse_source(source, cell=context_index)

        imports = update_import_table(parsed.tree, base=self._imports)
        symbols = build_symbol_table(
            parsed.tree,
            imports,
            base=self._symbols,
            source=source,
            cell=context_index,
        )

        result = _analyze_parsed(parsed, imports, symbols)
        self._imports = imports
        self._symbols = symbols
        self._cells.append(source)
        self._cell_indices.append(context_index)
        self._results.append(result)
        self._next_cell_index = context_index + 1
        return result

    def analyze_cell(self, index: int) -> AnalysisResult:
        return self._results[index]

    def explain_cell(self, index: int, style: str = "scientific") -> str:
        """Explain an analyzed cell by its position among analyzed Python cells."""
        return self.analyze_cell(index).render(style)

    def analyze_notebook_cell(self, cell_index: int) -> AnalysisResult:
        """Analyze by the original .ipynb cell index, including Markdown gaps."""
        if type(cell_index) is not int or cell_index < 0:
            raise ValueError("cell_index must be a non-negative integer.")
        try:
            position = self._cell_indices.index(cell_index)
        except ValueError as exc:
            if any(item.index == cell_index for item in self._skipped_cells):
                raise ValueError(
                    f"Notebook cell {cell_index} was skipped during static analysis."
                ) from exc
            raise ValueError(
                f"Notebook cell {cell_index} has no analyzed Python code."
            ) from exc
        return self._results[position]

    def explain_notebook_cell(self, cell_index: int, style: str = "scientific") -> str:
        """Explain a cell using its original index in a loaded notebook."""
        return self.analyze_notebook_cell(cell_index).render(style)

    def methodology(self) -> MethodologyReport:
        """Build a notebook-level, evidence-backed scientific Methods report."""

        return build_methodology_report(self.results)

    def render_methodology(self, *, include_evidence: bool = False) -> str:
        """Render the notebook methodology as deterministic Markdown."""

        return self.methodology().render(include_evidence=include_evidence)

    def render_overview(self, *, include_evidence: bool = False) -> str:
        """Explain an entire notebook with methodology, pipeline and diagnostics.

        The source is analyzed statically; no notebook cells are executed.
        """
        methods = self.render_methodology(include_evidence=include_evidence)
        methods = methods.removeprefix("# Methods").strip()
        methods = methods.replace("## ", "### ") if methods else (
            "No supported ML methodology was identified."
        )
        diagnostics = self.render_diagnostics()
        diagnostics = diagnostics.removeprefix("# Notebook diagnostics").strip()

        sections = [
            "# Notebook overview",
            "",
            f"- Analyzed Python cells: {self.cell_count}",
            f"- Skipped cells: {len(self.skipped_cells)}",
            "",
            "## Experiment pipeline",
            "",
            self.render_pipeline(),
            "",
            "## Methodology",
            "",
            methods,
            "",
            "## Dependency diagnostics",
            "",
            diagnostics,
        ]
        if self.skipped_cells:
            sections.extend(["", "## Skipped cells", ""])
            sections.extend(
                f"- Cell {item.index}: {item.reason}" for item in self.skipped_cells
            )
        return "\n".join(sections)

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

    def render_impact_ranking(self) -> str:
        """Render cells ordered by downstream static blast radius."""

        return render_impact_ranking(self.impact_ranking())

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def cell_indices(self) -> tuple[int, ...]:
        """Notebook cell indices corresponding to analyzed code cells."""

        return tuple(self._cell_indices)

    @property
    def skipped_cells(self) -> tuple[SkippedCell, ...]:
        """Notebook cells skipped because their shape or Python source is unsupported."""

        return tuple(self._skipped_cells)

    @property
    def results(self) -> tuple[AnalysisResult, ...]:
        return tuple(self._results)
