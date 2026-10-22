"""Evidence-backed import inventory for notebook reproducibility review.

This module does not import analyzed packages, query environments, or infer
installed versions. Unknown import roots may be local modules, not PyPI packages.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

ImportKind = Literal["stdlib", "non_stdlib", "relative"]

# Hints only, not verified installation requirements or version constraints.
_DISTRIBUTION_HINTS = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


@dataclass(frozen=True)
class ImportEvidence:
    cell: int
    line: int
    module: str
    root: str
    kind: ImportKind
    distribution_hint: str | None
    inside_scope: bool


@dataclass(frozen=True)
class NotebookImportReport:
    imports: tuple[ImportEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "notebook_import_inventory",
            "schema_version": 1,
            "verification": "source_only",
            "imports": [asdict(item) for item in self.imports],
            "non_stdlib_roots": sorted({
                item.root for item in self.imports if item.kind == "non_stdlib"
            }),
        }

    def render(self) -> str:
        lines = [
            "# Notebook import inventory",
            "",
            "Static import statements only: package installation and versions are not verified.",
        ]
        if not self.imports:
            lines.append("No import statements were detected in analyzed Python cells.")
        for item in self.imports:
            hint = f" (distribution hint: {item.distribution_hint})" if item.distribution_hint else ""
            scope = " [nested/conditional]" if item.inside_scope else ""
            lines.append(
                f"- Cell {item.cell}, line {item.line}: `{item.module}` "
                f"[{item.kind}]{hint}{scope}"
            )
        return "\n".join(lines)


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, cell: int) -> None:
        self.cell = cell
        self.depth = 0
        self.imports: list[ImportEvidence] = []

    def _record(self, module: str, line: int) -> None:
        root = module.split(".", 1)[0] if not module.startswith(".") else ""
        kind: ImportKind
        if module.startswith("."):
            kind = "relative"
        elif root in sys.stdlib_module_names or root == "__future__":
            kind = "stdlib"
        else:
            kind = "non_stdlib"
        self.imports.append(
            ImportEvidence(
                cell=self.cell,
                line=line,
                module=module,
                root=root,
                kind=kind,
                distribution_hint=_DISTRIBUTION_HINTS.get(root) if kind == "non_stdlib" else None,
                inside_scope=self.depth > 0,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        self._record(module, node.lineno)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.Try,
                ast.With,
                ast.AsyncWith,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
            ),
        ):
            self.depth += 1
            super().generic_visit(node)
            self.depth -= 1
        else:
            super().generic_visit(node)


def build_import_report(cells: tuple[tuple[int, str], ...]) -> NotebookImportReport:
    imports: list[ImportEvidence] = []
    for cell_index, source in cells:
        tree = ast.parse(source)
        collector = _ImportCollector(cell_index)
        collector.visit(tree)
        imports.extend(collector.imports)
    return NotebookImportReport(
        imports=tuple(sorted(imports, key=lambda item: (item.cell, item.line, item.module)))
    )
