from astscribe import analyze, explain


def test_optimizer_configuration_with_static_symbol() -> None:
    code = """
import torch
learning_rate = 2e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
"""
    result = analyze(code)
    optimizer = next(op for op in result.operations if op.kind == "optimizer_configuration")
    assert optimizer.subject == "AdamW"
    assert optimizer.attributes["lr"] == 2e-5
    assert optimizer.attributes["weight_decay"] == 0.01


def test_training_pattern() -> None:
    code = """
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
"""
    result = analyze(code)
    assert result.training_step is not None
    assert result.training_step.backward_pass
    assert result.training_step.parameter_update
    text = result.render("scientific")
    assert "optimization procedure" in text


def test_inference_pattern() -> None:
    code = """
model.eval()
with torch.inference_mode():
    outputs = model(inputs)
"""
    result = analyze(code)
    assert result.inference is not None
    assert result.inference.inference_mode
    assert result.inference.gradient_tracking_disabled


def test_manual_seed_does_not_claim_full_reproducibility() -> None:
    text = explain("import torch\ntorch.manual_seed(42)")
    assert "seeded with the value 42" in text
    assert "fully reproducible" not in text.lower()
