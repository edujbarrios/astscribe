import pytest

from astscribe import NotebookAnalyzer


def test_impact_follows_transitive_dependency_chain() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "dataset = load_data()",
            "features = preprocess(dataset)",
            "model = train(features)",
            "metrics = evaluate(model)",
        ]
    )

    impact = notebook.impact(0)

    assert impact.direct_dependents == (1,)
    assert impact.affected_cells == (1, 2, 3)
    assert impact.blast_radius == 3
    assert [path.target_cell for path in impact.paths] == [1, 2, 3]
    assert [path.distance for path in impact.paths] == [1, 2, 3]
    assert impact.paths[-1].cells == (0, 1, 2, 3)


def test_impact_handles_branching_dependencies() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "x = source()",
            "left = f(x)",
            "right = g(x)",
            "joined = combine(left, right)",
        ]
    )

    impact = notebook.impact(0)

    assert impact.direct_dependents == (1, 2)
    assert impact.affected_cells == (1, 2, 3)
    path_to_join = next(path for path in impact.paths if path.target_cell == 3)
    assert path_to_join.distance == 2
    assert path_to_join.cells in {(0, 1, 3), (0, 2, 3)}


def test_impact_reports_transitive_prerequisites() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "raw = load()",
            "clean = clean_data(raw)",
            "features = featurize(clean)",
            "result = fit(features)",
        ]
    )

    impact = notebook.impact(3)

    assert impact.direct_dependents == ()
    assert impact.affected_cells == ()
    assert impact.required_ancestors == (0, 1, 2)


def test_impact_path_keeps_symbols_for_each_hop() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "dataset = load()",
            "features = transform(dataset)",
            "score = evaluate(features)",
        ]
    )

    path = next(path for path in notebook.impact(0).paths if path.target_cell == 2)

    assert path.cells == (0, 1, 2)
    assert path.hops[0].symbols == ("dataset",)
    assert path.hops[1].symbols == ("features",)


def test_impact_uses_original_ipynb_cell_indices() -> None:
    notebook = NotebookAnalyzer.from_ipynb_data(
        {
            "cells": [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "markdown", "source": "# notes"},
                {"cell_type": "code", "source": "y = x + 1"},
                {"cell_type": "code", "source": "z = y + 1"},
            ]
        }
    )

    impact = notebook.impact(0)

    assert impact.affected_cells == (2, 3)
    assert impact.paths[-1].cells == (0, 2, 3)


def test_unknown_cell_is_rejected_explicitly() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1"])

    with pytest.raises(ValueError, match="Cell 99 is not present"):
        notebook.impact(99)


def test_impact_ranking_surfaces_high_fanout_state() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "config = make_config()",
            "model = build_model(config)",
            "dataset = load_dataset(config)",
            "trainer = make_trainer(model, dataset, config)",
            "result = trainer.train()",
        ]
    )

    ranking = notebook.impact_ranking()

    assert ranking[0].cell == 0
    assert ranking[0].affected_cells == 4
    assert ranking[0].direct_dependents == 3


def test_impact_rendering_contains_blast_radius_and_paths() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "y = x + 1", "z = y + 1"])

    rendered = notebook.render_impact(0)

    assert rendered.startswith("# Impact analysis for Cell 0")
    assert "Blast radius: 2 downstream cell(s)." in rendered
    assert "0 -> 1 -> 2" in rendered


def test_impact_to_dict_includes_derived_path_fields() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "y = x + 1"])

    data = notebook.impact(0).to_dict()

    assert data["blast_radius"] == 1
    assert data["paths"][0]["target_cell"] == 1
    assert data["paths"][0]["distance"] == 1
    assert data["paths"][0]["cells"] == (0, 1)


def test_isolated_cell_has_zero_blast_radius() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "y = 2"])

    impact = notebook.impact(0)

    assert impact.blast_radius == 0
    assert impact.paths == ()
    assert impact.render().startswith("# Impact analysis for Cell 0")


def test_impact_ranking_rendering_is_deterministic() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "config = make_config()",
            "model = build_model(config)",
            "dataset = load_dataset(config)",
            "trainer = make_trainer(model, dataset, config)",
        ]
    )

    rendered = notebook.render_impact_ranking()

    assert rendered.startswith("# Notebook impact ranking")
    assert rendered.index("Cell 0") < rendered.index("Cell 1")
    assert "3 affected cell(s)" in rendered
