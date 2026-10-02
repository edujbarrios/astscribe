from astscribe import analyze


def test_epoch_count_is_extracted_from_static_context() -> None:
    result = analyze(
        "import torch\n"
        "epochs = 4\n"
        "for epoch in range(epochs):\n"
        "    model.train()\n"
    )

    operation = next(op for op in result.operations if op.kind == "epoch_loop")
    assert operation.attributes["epochs"] == 4
    assert any(claim.rule == "pytorch.epoch_loop" for claim in result.claims)


def test_scheduler_step_is_not_misclassified_as_optimizer_step() -> None:
    result = analyze(
        "from torch.optim.lr_scheduler import StepLR\n"
        "scheduler = StepLR(optimizer, step_size=2, gamma=0.5)\n"
        "scheduler.step()\n"
    )

    kinds = [operation.kind for operation in result.operations]
    assert "scheduler_configuration" in kinds
    assert "scheduler_step" in kinds
    assert "parameter_update" not in kinds


def test_gradient_clipping_is_reported_with_static_threshold() -> None:
    result = analyze(
        "import torch\n"
        "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
    )

    operation = next(op for op in result.operations if op.kind == "gradient_clipping")
    assert operation.attributes["maximum_norm"] == 1.0
    claim = next(claim for claim in result.claims if claim.rule == "pytorch.gradient_clipping")
    assert "1.0" in claim.text


def test_autocast_and_grad_scaler_are_recognized() -> None:
    result = analyze(
        "import torch\n"
        "scaler = torch.amp.GradScaler()\n"
        "with torch.autocast('cuda'):\n"
        "    outputs = model(inputs)\n"
        "scaled_loss = scaler.scale(loss)\n"
        "scaler.step(optimizer)\n"
        "scaler.update()\n"
    )

    kinds = {operation.kind for operation in result.operations}
    assert "automatic_mixed_precision" in kinds
    assert "grad_scaler_configuration" in kinds
    assert "loss_scaling" in kinds
    assert "scaled_optimizer_step" in kinds
    assert "grad_scaler_update" in kinds


def test_device_configuration_and_transfer_are_reported() -> None:
    result = analyze(
        "import torch\n"
        "device = torch.device('cuda')\n"
        "model = model.to(device)\n"
    )

    kinds = {operation.kind for operation in result.operations}
    assert "device_configuration" in kinds
    assert "device_transfer" in kinds
    assert any(
        claim.rule == "pytorch.device_configuration" and "cuda" in claim.text
        for claim in result.claims
    )
