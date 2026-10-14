from astscribe import NotebookAnalyzer, analyze
from astscribe.parser import build_import_table, parse_source


def test_shadowing_torch_root_does_not_imply_tensor_creation() -> None:
    result = analyze("import torch\ntorch = backend\nx = torch.zeros(3)")
    assert not any(op.kind == "tensor_creation" for op in result.operations)


def test_shadowing_framework_root_persists_across_cells() -> None:
    notebook = NotebookAnalyzer.from_cells(["import torch", "torch = backend", "x = torch.ones(2)"])
    assert not any(op.kind == "tensor_creation" for op in notebook.results[2].operations)


def test_importing_again_restores_known_framework() -> None:
    notebook = NotebookAnalyzer.from_cells(["import torch", "torch = backend", "import torch", "x = torch.ones(2)"])
    assert any(op.kind == "tensor_creation" for op in notebook.results[3].operations)


def test_shadowed_alias_does_not_resolve_as_canonical_framework() -> None:
    table = build_import_table(parse_source("import torch as t\nt = backend").tree)
    assert table.resolve_dotted("t.zeros") == ""
