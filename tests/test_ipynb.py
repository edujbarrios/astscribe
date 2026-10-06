import json

import pytest

from astscribe import NotebookAnalyzer


def test_from_ipynb_data_preserves_notebook_cell_indices() -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Experiment"]},
            {"cell_type": "code", "source": ["import torch\n", "torch.manual_seed(7)\n"]},
            {"cell_type": "markdown", "source": ["Notes"]},
            {"cell_type": "code", "source": "model.eval()"},
        ]
    }

    analyzer = NotebookAnalyzer.from_ipynb_data(notebook)

    assert analyzer.cell_count == 2
    assert analyzer.cell_indices == (1, 3)
    seed_claim = next(
        claim for claim in analyzer.results[0].claims if claim.rule == "pytorch.manual_seed"
    )
    assert seed_claim.evidence.cell == 1


def test_from_ipynb_skips_non_python_cells_and_records_reason(tmp_path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "code", "source": "%matplotlib inline"},
            {"cell_type": "code", "source": "import torch\nmodel.eval()"},
        ]
    }
    path = tmp_path / "demo.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")

    analyzer = NotebookAnalyzer.from_ipynb(path)

    assert analyzer.cell_count == 1
    assert analyzer.cell_indices == (1,)
    assert len(analyzer.skipped_cells) == 1
    assert analyzer.skipped_cells[0].index == 0
    assert "not valid Python" in analyzer.skipped_cells[0].reason


def test_from_ipynb_can_fail_strictly_on_invalid_python() -> None:
    notebook = {"cells": [{"cell_type": "code", "source": "!pip install torch"}]}

    with pytest.raises(SyntaxError):
        NotebookAnalyzer.from_ipynb_data(notebook, skip_invalid_python=False)


def test_from_ipynb_rejects_invalid_notebook_shape() -> None:
    with pytest.raises(ValueError):
        NotebookAnalyzer.from_ipynb_data({"metadata": {}})


def test_from_ipynb_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="top-level object"):
        NotebookAnalyzer.from_ipynb_data([])
