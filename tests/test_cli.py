from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from astscribe.cli import main


def test_cli_explains_python_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "inference.py"
    source.write_text("model.eval()\n", encoding="utf-8")

    assert main([str(source), "--style", "concise"]) == 0
    assert "evaluation mode" in capsys.readouterr().out


def test_cli_reads_python_from_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("model.eval()\n"))

    assert main(["-", "--style", "concise"]) == 0
    assert "evaluation mode" in capsys.readouterr().out


def test_cli_renders_notebook_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notebook = tmp_path / "experiment.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": "result = future + 1"},
                    {"cell_type": "code", "source": "future = 41"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main([str(notebook), "--report", "diagnostics"]) == 0
    assert "dependency.forward_reference" in capsys.readouterr().out


def test_cli_reports_invalid_notebook_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notebook = tmp_path / "invalid.ipynb"
    notebook.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(notebook)])

    assert "expected a top-level object" in capsys.readouterr().err


def test_cli_requires_cell_for_impact_report(tmp_path: Path) -> None:
    notebook = tmp_path / "experiment.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(notebook), "--report", "impact"])
