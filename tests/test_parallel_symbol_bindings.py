from astscribe import NotebookAnalyzer
from astscribe.parser import build_import_table, build_symbol_table, parse_source


def test_chained_assignment_propagates_constant_to_both_names() -> None:
    parsed = parse_source("epochs = max_epochs = 3")
    table = build_symbol_table(parsed.tree, build_import_table(parsed.tree))
    assert table.constants["epochs"] == 3
    assert table.constants["max_epochs"] == 3


def test_parallel_unpack_uses_previous_values() -> None:
    parsed = parse_source("a = 1\nb = 2\na, b = b, a")
    table = build_symbol_table(parsed.tree, build_import_table(parsed.tree))
    assert table.constants["a"] == 2
    assert table.constants["b"] == 1


def test_nested_literal_unpack_propagates_values() -> None:
    parsed = parse_source("batch, (epochs, seed) = 16, (4, 7)")
    table = build_symbol_table(parsed.tree, build_import_table(parsed.tree))
    assert table.constants == {"batch": 16, "epochs": 4, "seed": 7}


def test_unknown_unpack_clears_previous_bindings() -> None:
    parsed = parse_source("epochs = 5\nepochs, seed = get_config()")
    table = build_symbol_table(parsed.tree, build_import_table(parsed.tree))
    assert "epochs" not in table.constants
    assert "seed" not in table.constants


def test_chained_constructor_binding_preserves_provenance() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from torch import nn\nloss_a = loss_b = nn.CrossEntropyLoss()",
        "loss = loss_b(outputs, targets)",
    ])
    assert any(op.kind == "loss_computation" for op in notebook.results[1].operations)
