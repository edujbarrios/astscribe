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


def test_cli_strict_notebook_rejects_invalid_python(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notebook = tmp_path / "invalid-python.ipynb"
    notebook.write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": "!pip install torch"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        main([str(notebook), "--strict"])

    assert "invalid syntax" in capsys.readouterr().err


def test_cli_requires_cell_for_impact_report(tmp_path: Path) -> None:
    notebook = tmp_path / "experiment.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(notebook), "--report", "impact"])


def test_cli_exports_dependency_graph_as_dot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notebook = tmp_path / "dependencies.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": "x = 1"},
                    {"cell_type": "code", "source": "y = x + 1"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main([str(notebook), "--report", "dependencies", "--dot"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("digraph ASTScribeNotebook")
    assert 'c0 -> c1 [label="x"]' in output


def test_cli_rejects_dot_for_non_dependency_report(tmp_path: Path) -> None:
    notebook = tmp_path / "experiment.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(notebook), "--report", "diagnostics", "--dot"])


def test_cli_renders_impact_ranking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notebook = tmp_path / "ranking.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": "config = make_config()"},
                    {"cell_type": "code", "source": "model = build_model(config)"},
                    {"cell_type": "code", "source": "result = evaluate(model)"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main([str(notebook), "--report", "ranking"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("# Notebook impact ranking")
    assert "Cell 0: 2 affected cell(s)" in output


def test_cli_emits_json_for_python_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "inference.py"
    source.write_text(
        "import torch\n"
        "model.eval()\n"
        "with torch.no_grad():\n"
        "    outputs = model(inputs)\n",
        encoding="utf-8",
    )

    assert main([str(source), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["inference"]["evaluation_mode"] is True
    assert payload["claims"]


def test_cli_emits_json_for_notebook_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notebook = tmp_path / "diagnostics.ipynb"
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

    assert main([str(notebook), "--report", "diagnostics", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["items"][0]["code"] == "dependency.forward_reference"
    assert payload["items"][0]["related_cell"] == 1


def test_cli_emits_json_for_impact_ranking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notebook = tmp_path / "ranking.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": "x = source()"},
                    {"cell_type": "code", "source": "y = transform(x)"},
                    {"cell_type": "code", "source": "z = consume(y)"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main([str(notebook), "--report", "ranking", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["items"][0] == {
        "affected_cells": 2,
        "cell": 0,
        "direct_dependents": 1,
    }


def test_cli_rejects_json_and_dot_together(tmp_path: Path) -> None:
    notebook = tmp_path / "experiment.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(notebook), "--report", "dependencies", "--json", "--dot"])
