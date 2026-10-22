from __future__ import annotations

import json
from pathlib import Path

from astscribe import NotebookAnalyzer
from astscribe.cli import main


def test_import_manifest_exposes_source_provenance_and_distribution_hints() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data(
        {
            "cells": [
                {"cell_type": "markdown", "source": "analysis"},
                {"cell_type": "code", "source": "import os\nimport numpy as np\n"},
                {"cell_type": "code", "source": "from PIL import Image\nfrom . import helpers\n"},
                {"cell_type": "code", "source": "if use_gpu:\n    import torch\n"},
                {"cell_type": "code", "source": "%pip install numpy"},
            ]
        }
    )
    report = analyzer.import_report()
    evidence = report.imports
    assert [(item.cell, item.line, item.module) for item in evidence] == [
        (1, 1, "os"),
        (1, 2, "numpy"),
        (2, 1, "PIL"),
        (2, 2, "."),
        (3, 2, "torch"),
    ]
    assert evidence[0].kind == "stdlib"
    assert evidence[1].kind == "non_stdlib"
    assert evidence[2].distribution_hint == "Pillow"
    assert evidence[3].kind == "relative"
    assert evidence[4].inside_scope is True
    assert report.to_dict()["non_stdlib_roots"] == ["PIL", "numpy", "torch"]
    assert report.to_dict()["verification"] == "source_only"


def test_notebook_import_inventory_is_deterministic_and_has_no_network_calls() -> None:
    analyzer = NotebookAnalyzer.from_cells(
        [
            "import sklearn, sys, cv2\nfrom yaml import safe_load",
            "from requests.sessions import Session\nimport astscribe",
        ]
    )
    report = analyzer.import_report()
    assert report == analyzer.import_report()
    assert report.to_dict()["non_stdlib_roots"] == [
        "astscribe", "cv2", "requests", "sklearn", "yaml",
    ]
    hints = {item.root: item.distribution_hint for item in report.imports}
    assert hints["sklearn"] == "scikit-learn"
    assert hints["cv2"] == "opencv-python"
    assert hints["yaml"] == "PyYAML"
    assert "not verified" in report.render()


def test_notebook_import_manifest_includes_nested_and_relative_imports() -> None:
    analyzer = NotebookAnalyzer.from_cells(
        [
            "from __future__ import annotations\n"
            "def train():\n"
            "    import torch.optim as optim\n"
            "    from ..models import trainer\n"
        ]
    )
    evidence = analyzer.import_report().imports
    assert [item.module for item in evidence] == [
        "__future__", "torch.optim", "..models",
    ]
    assert [item.inside_scope for item in evidence] == [False, True, True]
    assert [item.kind for item in evidence] == ["stdlib", "non_stdlib", "relative"]


def test_cli_imports_report_text_and_json(
    tmp_path: Path, capsys: object
) -> None:
    path = tmp_path / "model.ipynb"
    path.write_text(
        json.dumps({"cells": [
            {"cell_type": "code", "source": "import os\nimport transformers as tf"},
            {"cell_type": "markdown", "source": "Notes"},
            {"cell_type": "code", "source": "from sklearn.metrics import accuracy_score"},
        ]}),
        encoding="utf-8",
    )
    assert main([str(path), "--report", "imports", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["imports"][2]["cell"] == 2
    assert result["imports"][2]["line"] == 1
    assert result["imports"][2]["distribution_hint"] == "scikit-learn"

    assert main([str(path), "--report", "imports"]) == 0
    result_text = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Cell 2" in result_text
    assert "Static import statements" in result_text
