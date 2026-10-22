from __future__ import annotations

import json
from pathlib import Path

from astscribe import NotebookAnalyzer
from astscribe.cli import main


def test_execution_audit_preserves_original_cell_indices_and_avoids_execution() -> None:
    notebook = NotebookAnalyzer.from_ipynb_data(
        {
            "cells": [
                {"cell_type": "markdown", "source": "# Audit"},
                {"cell_type": "code", "source": "x = 1", "execution_count": 8},
                {"cell_type": "code", "source": "%timeit x", "execution_count": 2},
                {"cell_type": "raw", "source": "notes"},
                {"cell_type": "code", "source": "y = x + 1", "execution_count": 8},
                {"cell_type": "code", "source": "z = 3", "execution_count": None},
            ]
        }
    )

    report = notebook.execution_report()
    assert [item.cell for item in report.cells] == [1, 2, 4, 5]
    assert [item.execution_count for item in report.cells] == [8, 2, 8, None]
    assert report.warning_count == 2
    assert [item.code for item in report.findings] == [
        "execution.nonmonotonic_count",
        "execution.duplicate_count",
        "execution.no_saved_count",
    ]
    assert report.findings[1].related_cell == 1
    assert report.to_dict()["verification"] == "saved_metadata_only"
    assert "not a verified kernel execution trace" in report.render()


def test_execution_report_ignores_bool_and_invalid_counts() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data(
        {"cells": [
            {"cell_type": "code", "source": "a = 1", "execution_count": True},
            {"cell_type": "code", "source": "b = 2", "execution_count": "2"},
            {"cell_type": "code", "source": "c = 3", "execution_count": 0},
        ]}
    )
    report = analyzer.execution_report()
    assert report.warning_count == 0
    assert len(report.findings) == 3
    assert all(item.code == "execution.no_saved_count" for item in report.findings)


def test_execution_order_is_monotonic_for_increasing_saved_counts() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data(
        {"cells": [
            {"cell_type": "code", "source": "a = 1", "execution_count": 1},
            {"cell_type": "code", "source": "b = 2", "execution_count": 5},
        ]}
    )
    assert analyzer.execution_report().findings == ()


def test_execution_cli_json_and_text(tmp_path: Path, capsys: object) -> None:
    notebook = tmp_path / "order.ipynb"
    notebook.write_text(
        json.dumps({"cells": [
            {"cell_type": "code", "source": "a = 1", "execution_count": 3},
            {"cell_type": "code", "source": "b = a", "execution_count": 1},
        ]}),
        encoding="utf-8",
    )

    assert main([str(notebook), "--report", "execution", "--json"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload["kind"] == "notebook_execution_metadata"
    assert payload["warning_count"] == 1
    assert payload["findings"][0]["code"] == "execution.nonmonotonic_count"

    assert main([str(notebook), "--report", "execution"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "saved execution counters" in captured.out.lower()
    assert "execution.nonmonotonic_count" in captured.out
