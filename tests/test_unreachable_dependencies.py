from astscribe import NotebookAnalyzer


def test_if_false_does_not_produce_dependency() -> None:
    graph = NotebookAnalyzer.from_cells(["if False:\n    model = load_model()", "result = model(inputs)"]).dependency_graph()
    assert graph.edges == ()
    assert graph.nodes[1].unresolved_reads == ("inputs", "model")


def test_if_true_does_not_produce_from_else_branch() -> None:
    graph = NotebookAnalyzer.from_cells(["if True:\n    x = 1\nelse:\n    ghost = 2", "y = ghost"]).dependency_graph()
    assert graph.edges == ()
    assert graph.nodes[1].unresolved_reads == ("ghost",)


def test_while_false_does_not_produce_dependency() -> None:
    graph = NotebookAnalyzer.from_cells(["while False:\n    x = 1", "y = x"]).dependency_graph()
    assert graph.edges == ()
    assert graph.nodes[1].unresolved_reads == ("x",)


def test_unknown_condition_still_remains_conservative() -> None:
    graph = NotebookAnalyzer.from_cells(["if unknown:\n    x = 1", "y = x"]).dependency_graph()
    assert [(edge.producer_cell, edge.consumer_cell) for edge in graph.edges] == [(0, 1)]
