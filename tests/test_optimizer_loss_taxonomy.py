from astscribe import NotebookAnalyzer, analyze


def test_extended_optimizers_are_detected_with_source_evidence() -> None:
    for optimizer in ("RAdam", "RMSprop", "NAdam", "Adagrad", "LBFGS", "SparseAdam"):
        result = analyze(
            "import torch\n"
            f"optimizer = torch.optim.{optimizer}(model.parameters(), lr=0.001)\n"
            "optimizer.step()\n"
        )
        configs = [op for op in result.operations if op.kind == "optimizer_configuration"]
        assert len(configs) == 1
        assert configs[0].subject == optimizer
        assert configs[0].attributes["lr"] == 0.001
        assert any(op.kind == "parameter_update" for op in result.operations)


def test_loss_modules_are_objectives_not_forward_passes() -> None:
    result = analyze(
        "from torch import nn\n"
        "objective = nn.L1Loss()\n"
        "loss = objective(predictions, targets)\n"
    )
    assert any(op.kind == "loss_configuration" and op.subject == "L1Loss"
               for op in result.operations)
    assert len([op for op in result.operations if op.kind == "loss_computation"]) == 1
    assert not any(op.kind == "forward_pass" for op in result.operations)


def test_criterion_loss_in_cross_cell_context_is_not_a_model() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from torch.nn import KLDivLoss\ncriterion = KLDivLoss()",
        "loss = criterion(log_probs, target_probs)",
    ])
    kinds = [op.kind for op in notebook.results[1].operations]
    assert "loss_computation" in kinds
    assert "forward_pass" not in kinds
