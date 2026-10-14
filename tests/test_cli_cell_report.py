import json

import pytest

from astscribe.cli import main


def _notebook(tmp_path):
    path = tmp_path / "example.ipynb"
    path.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": "import torch"},
        {"cell_type": "markdown", "source": "# Explanation"},
        {"cell_type": "code", "source": "torch.manual_seed(42)"},
    ]}), encoding="utf-8")
    return path


def test_cell_report_respects_original_indices(tmp_path, capsys) -> None:
    path = _notebook(tmp_path)
    assert main([str(path), "--report", "cell", "--cell", "2"]) == 0
    assert "seed" in capsys.readouterr().out.lower()


def test_cell_report_emits_structured_json(tmp_path, capsys) -> None:
    path = _notebook(tmp_path)
    assert main([str(path), "--report", "cell", "--cell", "2", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert any(claim["evidence"]["cell"] == 2 for claim in data["claims"])


def test_cell_report_rejects_missing_index(tmp_path) -> None:
    path = _notebook(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        main([str(path), "--report", "cell"])
