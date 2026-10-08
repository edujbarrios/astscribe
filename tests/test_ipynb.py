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


def test_from_ipynb_accepts_utf8_bom(tmp_path) -> None:
    notebook = {"cells": [{"cell_type": "code", "source": "x = 1"}]}
    path = tmp_path / "bom.ipynb"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(notebook).encode("utf-8"))

    analyzer = NotebookAnalyzer.from_ipynb(path)

    assert analyzer.cell_count == 1
    assert analyzer.cell_indices == (0,)


def test_from_ipynb_does_not_stringify_invalid_source_list_items() -> None:
    notebook = {"cells": [{"cell_type": "code", "source": ["x = ", 1]}]}

    analyzer = NotebookAnalyzer.from_ipynb_data(notebook)

    assert analyzer.cell_count == 0
    assert analyzer.skipped_cells[0].index == 0
    assert "unsupported source representation" in analyzer.skipped_cells[0].reason



def test_strict_ipynb_rejects_unsupported_code_source_representation() -> None:
    notebook = {"cells": [{"cell_type": "code", "source": ["x = ", 1]}]}

    with pytest.raises(ValueError, match="unsupported source representation"):
        NotebookAnalyzer.from_ipynb_data(notebook, skip_invalid_python=False)

def test_append_after_ipynb_uses_original_coordinate_space() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data({
        "cells": [
            {"cell_type": "markdown", "source": "# Experiment"},
            {"cell_type": "code", "source": "x = 1"},
            {"cell_type": "code", "source": "%timeit x"},
            {"cell_type": "code", "source": "y = x + 1"},
        ]
    })

    analyzer.add_cell("z = y + 1")

    assert analyzer.cell_indices == (1, 3, 4)
    assert [(e.producer_cell, e.consumer_cell) for e in analyzer.dependency_graph().edges] == [
        (1, 3),
        (3, 4),
    ]
    assert analyzer.impact(1).affected_cells == (3, 4)


def test_explicit_cell_indices_must_be_unique_and_increasing() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("x = 1", cell_index=5)

    for invalid in (-1, True, 3, 5):
        with pytest.raises(ValueError, match="cell_index"):
            analyzer.add_cell("y = 2", cell_index=invalid)

    assert analyzer.cell_indices == (5,)
    analyzer.add_cell("y = x + 1")
    assert analyzer.cell_indices == (5, 6)


def test_rejected_cell_does_not_mutate_notebook_context() -> None:
    analyzer = NotebookAnalyzer.from_cells(["x = 1"])

    with pytest.raises(SyntaxError):
        analyzer.add_cell("def broken(")

    assert analyzer.cell_indices == (0,)
    assert analyzer.add_cell("y = x + 1").source == "y = x + 1"
    assert analyzer.cell_indices == (0, 1)

def test_append_after_trailing_markdown_and_skipped_cells() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data(
        {
            "cells": [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "markdown", "source": "analysis"},
                {"cell_type": "code", "source": "%timeit x"},
                {"cell_type": "code", "source": ""},
                {"cell_type": "markdown", "source": "last note"},
            ]
        }
    )

    result = analyzer.add_cell("y = x + 1")

    assert analyzer.cell_indices == (0, 5)
    assert analyzer.dependency_graph().edges[0].consumer_cell == 5
    assert result.source == "y = x + 1"


def test_append_to_notebook_with_no_analyzable_code_uses_full_index() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data(
        {"cells": [
            {"cell_type": "markdown", "source": "# Heading"},
            {"cell_type": "code", "source": "!echo skipped"},
            {"cell_type": "code", "source": "  "},
        ]}
    )

    analyzer.add_cell("x = 1")

    assert analyzer.cell_indices == (3,)


def test_explicit_index_cannot_reuse_skipped_notebook_position() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data(
        {"cells": [
            {"cell_type": "code", "source": "x = 1"},
            {"cell_type": "markdown", "source": "some notes"},
        ]}
    )

    with pytest.raises(ValueError, match="existing notebook cell index"):
        analyzer.add_cell("y = 2", cell_index=1)

    assert analyzer.cell_indices == (0,)


def test_original_notebook_cell_lookup_ignores_markdown_gaps() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data({
        "cells": [
            {"cell_type": "markdown", "source": "# Model"},
            {"cell_type": "code", "source": "import torch\nmodel.eval()"},
            {"cell_type": "raw", "source": "Notes"},
            {"cell_type": "code", "source": "with torch.no_grad():\n    prediction = model(inputs)"},
        ]
    })
    assert analyzer.cell_indices == (1, 3)
    assert analyzer.analyze_notebook_cell(3) is analyzer.results[1]
    assert "Gradient tracking is disabled" in analyzer.explain_notebook_cell(3)

    with pytest.raises(ValueError, match="no analyzed Python code"):
        analyzer.explain_notebook_cell(2)
    with pytest.raises(ValueError, match="non-negative integer"):
        analyzer.explain_notebook_cell(True)


def test_invalid_notebook_shapes_are_reported_without_losing_indices() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data({
        "cells": [
            {"cell_type": "code", "source": "x = 1"},
            None,
            {"cell_type": "python", "source": "x = 2"},
            {"cell_type": "code"},
            {"cell_type": "markdown", "source": "# title"},
            {"cell_type": "code", "source": "y = x + 1"},
        ]
    })
    assert analyzer.cell_indices == (0, 5)
    assert [item.index for item in analyzer.skipped_cells] == [1, 2, 3]
    assert "not an object" in analyzer.skipped_cells[0].reason
    assert "unsupported cell_type" in analyzer.skipped_cells[1].reason
    assert "missing a source field" in analyzer.skipped_cells[2].reason
    with pytest.raises(ValueError, match="was skipped"):
        analyzer.analyze_notebook_cell(2)


@pytest.mark.parametrize("cell", [
    None,
    {"cell_type": "python", "source": "x = 1"},
    {"cell_type": "code"},
])
def test_strict_loader_rejects_malformed_notebook_cells(cell: object) -> None:
    with pytest.raises(ValueError, match="Invalid notebook cell 0"):
        NotebookAnalyzer.from_ipynb_data(
            {"cells": [cell]}, skip_invalid_python=False
        )


def test_whole_notebook_overview_covers_pipeline_methods_and_skipped_cells() -> None:
    analyzer = NotebookAnalyzer.from_ipynb_data({
        "cells": [
            {"cell_type": "markdown", "source": "# Experiment"},
            {"cell_type": "code", "source": "import torch\nmodel.eval()"},
            {"cell_type": "code", "source": "%time train()"},
            {"cell_type": "code", "source": "with torch.no_grad():\n    output = model(inputs)"},
        ]
    })
    overview = analyzer.render_overview(include_evidence=True)
    assert overview.startswith("# Notebook overview")
    assert "Analyzed Python cells: 2" in overview
    assert "Skipped cells: 1" in overview
    assert "## Experiment pipeline" in overview
    assert "## Methodology" in overview
    assert "## Dependency diagnostics" in overview
    assert "## Skipped cells" in overview
    assert "Cell 2:" in overview
    assert "pytorch.no_grad" in overview
    assert "### Evaluation and inference" in overview


def test_empty_notebook_overview_explains_missing_evidence() -> None:
    overview = NotebookAnalyzer.from_cells([]).render_overview()
    assert "No supported ML methodology was identified." in overview
    assert "No evidence-backed experiment pipeline could be reconstructed." in overview
