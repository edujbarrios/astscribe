from __future__ import annotations

from dataclasses import dataclass, field

from astscribe.api import _analyze_parsed
from astscribe.parser import ImportTable, SymbolTable, build_import_table, build_symbol_table, parse_source
from astscribe.sir import AnalysisResult


@dataclass
class NotebookAnalyzer:
    """Incrementally analyze notebook cells without executing them.

    Context flows only forward: each cell can use imports, literal values, and constructor
    bindings established by previous cells. Later cells never affect earlier analyses.
    """

    _imports: ImportTable = field(default_factory=ImportTable)
    _symbols: SymbolTable = field(default_factory=SymbolTable)
    _cells: list[str] = field(default_factory=list)
    _results: list[AnalysisResult] = field(default_factory=list)

    @classmethod
    def from_cells(cls, cells: list[str]) -> NotebookAnalyzer:
        analyzer = cls()
        for cell in cells:
            analyzer.add_cell(cell)
        return analyzer

    def add_cell(self, source: str) -> AnalysisResult:
        cell_index = len(self._cells)
        parsed = parse_source(source, cell=cell_index)

        local_imports = build_import_table(parsed.tree)
        self._imports.aliases.update(local_imports.aliases)
        self._symbols = build_symbol_table(
            parsed.tree,
            self._imports,
            base=self._symbols,
            source=source,
            cell=cell_index,
        )

        result = _analyze_parsed(parsed, self._imports, self._symbols)
        self._cells.append(source)
        self._results.append(result)
        return result

    def analyze_cell(self, index: int) -> AnalysisResult:
        return self._results[index]

    def explain_cell(self, index: int, style: str = "scientific") -> str:
        return self.analyze_cell(index).render(style)

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def results(self) -> tuple[AnalysisResult, ...]:
        return tuple(self._results)
