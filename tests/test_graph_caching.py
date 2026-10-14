from astscribe import NotebookAnalyzer


def test_dependency_graph_cache_is_reused_and_invalidated() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "y = x"])
    original = notebook.dependency_graph()
    assert notebook.dependency_graph() is original
    notebook.add_cell("z = y")
    updated = notebook.dependency_graph()
    assert updated is not original
    assert len(updated.edges) == 2


def test_optimized_ranking_agrees_with_individual_impact_reports() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "x = source()",
        "a = left(x)",
        "b = right(x)",
        "c = merge(a, b)",
        "d = score(c)",
    ])
    for ranked in notebook.impact_ranking():
        report = notebook.impact(ranked.cell)
        assert ranked.affected_cells == report.blast_radius
        assert ranked.direct_dependents == len(report.direct_dependents)


def test_ranking_large_chain() -> None:
    cells = ["x0 = 0"] + [f"x{i} = x{i - 1} + 1" for i in range(1, 150)]
    notebook = NotebookAnalyzer.from_cells(cells)
    ranking = notebook.impact_ranking()
    assert ranking[0].cell == 0
    assert ranking[0].affected_cells == 149
