from astscribe import NotebookAnalyzer
from astscribe.dependency import collect_symbol_events


def test_dependency_graph_tracks_cross_cell_symbol_flow() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("dataset = load_data()")
    notebook.add_cell("tokenized = preprocess(dataset)")
    notebook.add_cell("batch = collate(tokenized)\nresult = evaluate(batch)")

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("dataset",)),
        (1, 2, ("tokenized",)),
    ]
    assert graph.parents(2) == (1,)
    assert graph.children(0) == (1,)


def test_import_then_use_in_same_cell_does_not_create_external_dependency() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("import torch\nx = torch.tensor([1, 2, 3])")

    graph = notebook.dependency_graph()
    node = graph.nodes[0]

    assert graph.edges == ()
    assert "torch" in node.defines
    assert "torch" in node.reads
    assert "torch" not in node.unresolved_reads


def test_read_before_reassignment_depends_on_previous_producer() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("x = 1")
    notebook.add_cell("x = x + 1")

    graph = notebook.dependency_graph()

    assert len(graph.edges) == 1
    assert graph.edges[0].producer_cell == 0
    assert graph.edges[0].consumer_cell == 1
    assert graph.edges[0].symbols == ("x",)
    assert graph.redefinitions[0].symbol == "x"
    assert graph.redefinitions[0].previous_cell == 0
    assert graph.redefinitions[0].new_cell == 1


def test_latest_redefinition_becomes_future_producer() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("x = 1")
    notebook.add_cell("before = x\nx = 2")
    notebook.add_cell("after = x")

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("x",)),
        (1, 2, ("x",)),
    ]


def test_comprehension_target_is_local_to_comprehension() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("values = [1, 2, 3]")
    notebook.add_cell("doubled = [item * 2 for item in values]")

    graph = notebook.dependency_graph()
    node = graph.nodes[1]

    assert graph.edges[0].symbols == ("values",)
    assert "item" not in node.defines
    assert "item" not in node.unresolved_reads


def test_function_body_is_not_treated_as_execution_dependency() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("scale = 2")
    notebook.add_cell("def transform(value):\n    return value * scale")

    graph = notebook.dependency_graph()

    assert graph.edges == ()
    assert graph.nodes[1].defines == ("transform",)


def test_function_default_expression_is_dependency() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("scale = 2")
    notebook.add_cell("def transform(value, factor=scale):\n    return value * factor")

    graph = notebook.dependency_graph()

    assert graph.edges[0].producer_cell == 0
    assert graph.edges[0].consumer_cell == 1
    assert graph.edges[0].symbols == ("scale",)


def test_delete_removes_symbol_as_future_producer() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("x = 1")
    notebook.add_cell("del x")
    notebook.add_cell("y = x")

    graph = notebook.dependency_graph()

    assert graph.edges == ()
    assert graph.nodes[2].unresolved_reads == ("x",)


def test_dependency_graph_preserves_original_ipynb_cell_indices() -> None:
    notebook = NotebookAnalyzer.from_ipynb_data(
        {
            "cells": [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "markdown", "source": "# Notes"},
                {"cell_type": "code", "source": "y = x + 1"},
            ]
        }
    )

    graph = notebook.dependency_graph()

    assert [node.cell for node in graph.nodes] == [0, 2]
    assert graph.edges[0].producer_cell == 0
    assert graph.edges[0].consumer_cell == 2


def test_dot_export_requires_no_graphviz_runtime() -> None:
    notebook = NotebookAnalyzer.from_cells(["x = 1", "y = x + 1"])

    dot = notebook.dependency_dot()

    assert dot.startswith("digraph ASTScribeNotebook {")
    assert 'c0 -> c1 [label="x"]' in dot


def test_unresolved_reads_exclude_python_builtins() -> None:
    notebook = NotebookAnalyzer.from_cells(["size = len(items)\nprint(size)"])

    node = notebook.dependency_graph().nodes[0]

    assert "len" not in node.unresolved_reads
    assert "print" not in node.unresolved_reads
    assert "items" in node.unresolved_reads


def test_symbol_events_keep_read_before_write_order() -> None:
    events = collect_symbol_events("x = x + 1")

    assert [(event.kind, event.symbol) for event in events] == [
        ("read", "x"),
        ("write", "x"),
    ]


def test_class_body_reads_create_cross_cell_dependencies() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "default_batch_size = 32",
            "class Config:\n    batch_size = default_batch_size\n    doubled = batch_size * 2",
        ]
    )

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("default_batch_size",)),
    ]
    assert graph.nodes[1].defines == ("Config",)
    assert "batch_size" not in graph.nodes[1].unresolved_reads


def test_class_body_imports_remain_class_local() -> None:
    notebook = NotebookAnalyzer.from_cells(
        ["class Constants:\n    import math\n    tau = math.tau", "value = math.pi"]
    )

    graph = notebook.dependency_graph()

    assert graph.nodes[0].defines == ("Constants",)
    assert "math" not in graph.nodes[0].unresolved_reads
    assert graph.nodes[1].unresolved_reads == ("math",)


def test_annotation_only_name_does_not_create_runtime_producer() -> None:
    notebook = NotebookAnalyzer.from_cells(["x: int", "y = x"])

    graph = notebook.dependency_graph()

    assert "x" not in graph.nodes[0].defines
    assert graph.edges == ()
    assert graph.nodes[1].unresolved_reads == ("x",)


def test_class_comprehension_does_not_capture_class_local_name() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "x = 10",
            "class Config:\n    x = 1\n    values = [x for _ in range(1)]",
        ]
    )

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("x",)),
    ]


def test_class_delete_exposes_outer_notebook_producer_again() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "x = 10",
            "class Config:\n    x = 1\n    del x\n    fallback = x",
        ]
    )

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("x",)),
    ]


def test_class_comprehension_first_iterable_uses_class_scope() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "size = 99",
            "class Config:\n    size = 3\n    values = [item for item in range(size)]",
        ]
    )

    graph = notebook.dependency_graph()

    assert graph.edges == ()
    assert "size" not in graph.nodes[1].unresolved_reads


def test_class_comprehension_body_still_skips_class_scope() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "value = 10",
            "class Config:\n    value = 3\n    values = [value for _ in range(1)]",
        ]
    )

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("value",)),
    ]


def test_comprehension_assignment_expression_binds_containing_scope() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "values = [1, 2, 3]",
            "doubled = [last := item * 2 for item in values]",
            "result = last",
        ]
    )

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("values",)),
        (1, 2, ("last",)),
    ]
    assert "last" in graph.nodes[1].defines
    assert "item" not in graph.nodes[1].defines


def test_match_capture_is_available_to_guard_body_and_later_cells() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            'payload = {"value": 3}',
            'match payload:\n    case {"value": value} if value > 0:\n        result = value',
            "after = value",
        ]
    )

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("payload",)),
        (1, 2, ("value",)),
    ]
    assert set(graph.nodes[1].defines) == {"result", "value"}
    assert "value" not in graph.nodes[1].unresolved_reads


def test_match_value_pattern_reads_outer_symbol() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "Color = make_color_type()\nitem = get_item()",
            'match item:\n    case Color.RED:\n        result = "red"',
        ]
    )

    graph = notebook.dependency_graph()

    assert [(edge.producer_cell, edge.consumer_cell, edge.symbols) for edge in graph.edges] == [
        (0, 1, ("Color", "item")),
    ]


def test_exception_alias_is_scoped_to_handler_and_cleared_afterwards() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "try:\n    risky()\nexcept Exception as exc:\n    message = str(exc)",
            "after = exc",
        ]
    )

    graph = notebook.dependency_graph()

    assert "exc" not in graph.nodes[0].unresolved_reads
    assert graph.edges == ()
    assert graph.nodes[1].unresolved_reads == ("exc",)
