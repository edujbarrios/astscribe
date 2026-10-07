from astscribe import NotebookAnalyzer


def test_cross_cell_optimizer_context_is_resolved() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell(
        "import torch\n"
        "from torch.optim import AdamW\n"
        "learning_rate = 2e-5\n"
    )
    analyzer.add_cell(
        "optimizer = AdamW(\n"
        "    model.parameters(),\n"
        "    lr=learning_rate,\n"
        "    weight_decay=0.01,\n"
        ")\n"
    )
    result = analyzer.add_cell(
        "optimizer.zero_grad()\n"
        "loss.backward()\n"
        "optimizer.step()\n"
    )

    contextual = [claim for claim in result.claims if claim.rule == "context.optimizer_binding"]
    assert len(contextual) == 1
    assert "AdamW" in contextual[0].text
    assert "2e-05" in contextual[0].text
    assert "0.01" in contextual[0].text
    assert contextual[0].evidence.cell == 1

    update = next(op for op in result.operations if op.kind == "parameter_update")
    assert update.attributes["optimizer"] == "AdamW"
    assert update.attributes["lr"] == 2e-5
    assert update.attributes["weight_decay"] == 0.01


def test_import_aliases_persist_between_cells() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("import torch as t")
    result = analyzer.add_cell(
        "model.eval()\n"
        "with t.no_grad():\n"
        "    outputs = model(inputs)\n"
    )

    assert result.inference is not None
    assert result.inference.evaluation_mode
    assert result.inference.gradient_tracking_disabled
    assert result.inference.forward_pass


def test_constructor_context_can_identify_loss_calls() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("from torch import nn\ncriterion = nn.CrossEntropyLoss()")
    result = analyzer.add_cell("loss = criterion(outputs, targets)")

    assert any(operation.kind == "loss_computation" for operation in result.operations)


def test_cell_provenance_is_attached_to_current_evidence() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("import torch")
    result = analyzer.add_cell("torch.manual_seed(42)")

    seed_claim = next(claim for claim in result.claims if claim.rule == "pytorch.manual_seed")
    assert seed_claim.evidence.cell == 1


def test_context_flows_forward_only() -> None:
    analyzer = NotebookAnalyzer()
    first = analyzer.add_cell("optimizer.step()")
    analyzer.add_cell("from torch.optim import AdamW\noptimizer = AdamW(model.parameters())")

    assert not any(claim.rule == "context.optimizer_binding" for claim in first.claims)



def test_shadowed_import_alias_does_not_leak_framework_semantics() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("import torch as t")
    analyzer.add_cell("t = custom_backend")
    result = analyzer.add_cell("with t.no_grad():\n    output = compute()")

    assert not any(operation.kind == "gradient_tracking_disabled" for operation in result.operations)


def test_import_inside_function_does_not_leak_to_later_cells() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("def helper():\n    import torch as t\n    return t.tensor([1])")
    result = analyzer.add_cell("with t.no_grad():\n    output = compute()")

    assert not any(operation.kind == "gradient_tracking_disabled" for operation in result.operations)


def test_reimported_alias_restores_framework_context() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("import torch as t")
    analyzer.add_cell("t = custom_backend")
    analyzer.add_cell("import torch as t")
    result = analyzer.add_cell("with t.no_grad():\n    output = compute()")

    assert any(operation.kind == "gradient_tracking_disabled" for operation in result.operations)


def test_deleted_constructor_context_is_not_reused() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("from torch.optim import AdamW\nopt = AdamW(model.parameters())")
    analyzer.add_cell("del opt")
    result = analyzer.add_cell("opt.step()")

    assert not any(operation.kind == "parameter_update" for operation in result.operations)


def test_function_redefinition_clears_constructor_context() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("from torch import nn\nloss_fn = nn.CrossEntropyLoss()")
    analyzer.add_cell("def loss_fn(outputs, targets):\n    return custom_loss(outputs, targets)")
    result = analyzer.add_cell("loss = loss_fn(outputs, targets)")

    assert not any(operation.kind == "loss_computation" for operation in result.operations)


def test_loop_binding_invalidates_stale_constant_context() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("import torch\nepochs = 5")
    analyzer.add_cell("for epochs in schedule:\n    pass")
    result = analyzer.add_cell("for epoch in range(epochs):\n    train_one_epoch()")

    assert not any(operation.kind == "epoch_loop" for operation in result.operations)
