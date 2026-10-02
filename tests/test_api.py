from astscribe import explain


def test_explain_returns_scientific_text() -> None:
    code = """
model.eval()
with torch.no_grad():
    outputs = model(inputs)
"""
    text = explain(code)
    assert isinstance(text, str)
    assert "Inference procedure" in text
    assert "evaluation mode" in text


def test_return_result() -> None:
    result = explain("model.eval()", return_result=True)
    assert not isinstance(result, str)
    assert result.claims


def test_unknown_code_is_conservative() -> None:
    text = explain("value = custom_function(data)")
    assert "No evidence-backed" in text
