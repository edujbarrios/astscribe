from __future__ import annotations

import json
from pathlib import Path

import pytest

from astscribe import NotebookAnalyzer
from astscribe.cli import main


def _notebook() -> dict[str, object]:
    return {
        "cells": [
            {"cell_type": "code", "source": "import torch\nx = 1", "execution_count": 5},
            {"cell_type": "code", "source": "%pip install torch", "execution_count": 2},
            {"cell_type": "markdown", "source": "# Experiment"},
            {"cell_type": "code", "source": "y = x + missing", "execution_count": 2},
            {"cell_type": "code", "source": "z = 3", "execution_count": None},
        ]
    }


def test_notebook_static_audit_unifies_findings_and_preserves_cell_indices() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data(_notebook())
    report = analyzer.audit_report()
    assert report.analyzed_cells == 3
    assert report.skipped_cells == 1
    assert report.non_stdlib_roots == ("torch",)
    assert report.warning_count >= 4
    assert report.to_dict()["verification"] == "source_and_saved_metadata_only"
    findings = report.findings
    assert [(item.cell, item.code) for item in findings] == sorted(
        (item.cell, item.code) for item in findings
    )
    assert any(item.code == "audit.skipped_code" and item.cell == 1 for item in findings)
    assert any(item.code == "execution.nonmonotonic_count" and item.cell == 1 for item in findings)
    assert any(item.code == "execution.duplicate_count" and item.cell == 3 for item in findings)
    assert any(
        item.code == "dependency.unresolved_symbol" and item.cell == 3 for item in findings
    )
    assert "saved metadata" in report.render().lower()


def test_clean_notebook_has_no_audit_warnings() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data(
        {"cells": [
            {"cell_type": "code", "source": "import torch\na = 1", "execution_count": 1},
            {"cell_type": "code", "source": "b = a + 1", "execution_count": 2},
        ]}
    )
    report = analyzer.audit_report()
    assert report.warning_count == 0
    assert report.non_stdlib_roots == ("torch",)
    assert report == analyzer.audit_report()


def test_audit_cli_can_fail_ci_for_skipped_cells_without_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "audit.ipynb"
    path.write_text(json.dumps(_notebook()), encoding="utf-8")

    assert main([str(path), "--report", "audit", "--json", "--fail-on-warning"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "notebook_static_audit"
    assert payload["warning_count"] >= 4
    assert payload["skipped_cells"] == 1

    assert main([str(path), "--report", "audit", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["warning_count"] >= 4

    assert main([str(path), "--report", "audit", "--fail-on-warning"]) == 1
    assert "audit.skipped_code" in capsys.readouterr().out


def test_audit_cli_preserves_old_fail_on_warning_semantics_for_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "skipped.ipynb"
    path.write_text(
        json.dumps({"cells": [
            {"cell_type": "code", "source": "x = 1", "execution_count": 3},
            {"cell_type": "code", "source": "%timeit x", "execution_count": 1},
        ]}),
        encoding="utf-8",
    )

    # Dependency-only reports continue gating on dependency warnings.
    assert main([str(path), "--report", "diagnostics", "--fail-on-warning"]) == 0
    capsys.readouterr()

    # The aggregate gate also includes skipped-cell and metadata warnings.
    assert main([str(path), "--report", "audit", "--fail-on-warning"]) == 1
    assert "audit.skipped_code" in capsys.readouterr().out


def test_python_source_rejects_notebook_audit_mode(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    path.write_text("x = 1", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main([str(path), "--report", "audit"])
