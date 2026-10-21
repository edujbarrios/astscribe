"""Receiver heuristics must not override known incompatible constructor types."""

from astscribe import analyze


def test_unrelated_optimizer_receiver_is_not_parameter_update() -> None:
    result = analyze("class Worker:\n    def step(self):\n        pass\n"
                     "optimizer = Worker()\noptimizer.step()")
    assert not any(op.kind == "parameter_update" for op in result.operations)


def test_unrelated_scheduler_and_scaler_receivers_are_not_pytorch() -> None:
    source = (
        "class Worker:\n    def step(self, *args):\n        pass\n"
        "scheduler = Worker()\nscheduler.step()\n"
        "scaler = Worker()\nscaler.step(other)"
    )
    result = analyze(source)
    assert not any(op.kind in {"scheduler_step", "scaled_optimizer_step"} for op in result.operations)


def test_unrelated_model_and_criterion_calls_are_not_forward_or_loss() -> None:
    result = analyze(
        "class Worker:\n    def __call__(self, *args):\n        pass\n"
        "model = Worker()\ncriterion = Worker()\n"
        "result = model(inputs)\nloss = criterion(result, targets)"
    )
    assert not any(op.kind in {"forward_pass", "loss_computation"} for op in result.operations)


def test_torch_optimizer_constructor_still_enables_parameter_update() -> None:
    result = analyze("import torch\noptimizer = torch.optim.AdamW(params)\noptimizer.step()")
    assert any(op.kind == "parameter_update" for op in result.operations)


def test_unknown_conventional_optimizer_name_keeps_legacy_support() -> None:
    result = analyze("optimizer.step()")
    assert any(op.kind == "parameter_update" for op in result.operations)
