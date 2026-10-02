from astscribe import NotebookAnalyzer


def test_forward_reference_is_distinguished_from_external_unresolved_symbol() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "output = model(1)",
            "model = object()",
        ]
    )

    diagnostics = notebook.diagnostics()
    forward = diagnostics.by_code("dependency.forward_reference")

    assert len(forward) == 1
    assert forward[0].cell == 0
    assert forward[0].symbol == "model"
    assert forward[0].related_cell == 1
    assert diagnostics.by_code("dependency.unresolved_symbol") == ()


def test_external_or_hidden_state_is_reported_as_unresolved_symbol() -> None:
    notebook = NotebookAnalyzer.from_cells(["result = injected_value + 1"])

    unresolved = notebook.diagnostics().by_code("dependency.unresolved_symbol")

    assert len(unresolved) == 1
    assert unresolved[0].symbol == "injected_value"
    assert unresolved[0].related_cell is None
    assert "hidden kernel state" in unresolved[0].message


def test_symbol_redefinition_is_reported() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "x = 2"])

    diagnostics = notebook.diagnostics()
    redefinitions = diagnostics.by_code("dependency.symbol_redefinition")

    assert len(redefinitions) == 1
    assert redefinitions[0].cell == 1
    assert redefinitions[0].related_cell == 0
    assert redefinitions[0].symbol == "x"


def test_overwrite_before_cross_cell_use_is_reported_conservatively() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "x = 2", "result = x"])

    overwritten = notebook.diagnostics().by_code(
        "dependency.overwritten_before_cross_cell_use"
    )

    assert len(overwritten) == 1
    assert overwritten[0].cell == 1
    assert overwritten[0].related_cell == 0
    assert "before any later analyzed cell consumes it" in overwritten[0].message


def test_consumed_definition_is_not_reported_as_overwritten_before_use() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "first = x", "x = 2"])

    diagnostics = notebook.diagnostics()

    assert diagnostics.by_code("dependency.overwritten_before_cross_cell_use") == ()
    assert len(diagnostics.by_code("dependency.symbol_redefinition")) == 1


def test_read_before_redefinition_counts_as_consumption() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "x = x + 1"])

    diagnostics = notebook.diagnostics()

    assert diagnostics.by_code("dependency.overwritten_before_cross_cell_use") == ()
    assert len(diagnostics.by_code("dependency.symbol_redefinition")) == 1


def test_clean_dependency_chain_has_no_diagnostics() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "y = x + 1", "z = y + 1"])

    diagnostics = notebook.diagnostics()

    assert diagnostics.items == ()
    assert notebook.render_diagnostics() == (
        "No supported notebook dependency diagnostics were detected."
    )


def test_diagnostic_rendering_is_deterministic() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "value = missing + later",
            "later = 3",
        ]
    )

    rendered = notebook.render_diagnostics()

    assert rendered.startswith("# Notebook diagnostics")
    assert "dependency.forward_reference" in rendered
    assert "dependency.unresolved_symbol" in rendered
    assert rendered.index("dependency.forward_reference") < rendered.index(
        "dependency.unresolved_symbol"
    )


def test_diagnostics_preserve_original_notebook_cell_indices() -> None:
    notebook = NotebookAnalyzer.from_ipynb_data(
        {
            "cells": [
                {"cell_type": "code", "source": "output = model(1)"},
                {"cell_type": "markdown", "source": "# model setup"},
                {"cell_type": "code", "source": "model = object()"},
            ]
        }
    )

    diagnostic = notebook.diagnostics().by_code("dependency.forward_reference")[0]

    assert diagnostic.cell == 0
    assert diagnostic.related_cell == 2


def test_diagnostics_to_dict_is_structured() -> None:
    notebook = NotebookAnalyzer.from_cells(["result = external_value"])

    data = notebook.diagnostics().to_dict()

    assert data["items"][0]["code"] == "dependency.unresolved_symbol"
    assert data["items"][0]["severity"] == "warning"
    assert data["items"][0]["symbol"] == "external_value"
