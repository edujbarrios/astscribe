"""Proven unreachable branches must not contribute semantic evidence."""

from astscribe import NotebookAnalyzer, analyze


def test_if_false_does_not_create_semantic_optimizer_claims() -> None:
    result = analyze(
        "import torch\nif False:\n    optimizer.step()\n    loss.backward()"
    )
    assert not any(op.kind in {"parameter_update", "backward_pass"} for op in result.operations)
    assert not result.training_step


def test_if_true_ignores_else_semantics() -> None:
    result = analyze(
        "import torch\nif True:\n    model.eval()\nelse:\n    model.train()"
    )
    assert any(op.kind == "evaluation_mode" for op in result.operations)
    assert not any(op.kind == "training_mode" for op in result.operations)


def test_while_false_keeps_else_but_not_body() -> None:
    result = analyze(
        "import torch\nwhile False:\n    model.train()\nelse:\n    model.eval()"
    )
    assert any(op.kind == "evaluation_mode" for op in result.operations)
    assert not any(op.kind == "training_mode" for op in result.operations)


def test_unreachable_constructor_binding_does_not_leak_across_cells() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from transformers import AutoModel\n"
        "if False:\n    model = AutoModel.from_pretrained('bert-base-uncased')",
        "model.generate(input_ids)",
    ])
    assert not any(c.rule == "transformers.generate" for c in notebook.results[1].claims)


def test_reachable_if_true_import_has_original_evidence_lines() -> None:
    result = analyze(
        "if True:\n    from torch.optim import AdamW\n"
        "opt = AdamW(params, lr=0.01)"
    )
    operations = [op for op in result.operations if op.kind == "optimizer_configuration"]
    assert operations
    assert operations[0].evidence is not None
    assert operations[0].evidence.line_start == 3
