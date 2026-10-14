from astscribe import NotebookAnalyzer, analyze


def test_import_before_shadowed_alias_still_resolves_earlier_call() -> None:
    result = analyze("from torch import randn as sample\ntensor = sample(3, 3)\nsample = custom_function")
    assert any(op.kind == "tensor_creation" for op in result.operations)


def test_alias_resolution_does_not_use_future_import() -> None:
    result = analyze("tensor = sample(3)\nfrom torch import randn as sample")
    assert not any(op.kind == "tensor_creation" for op in result.operations)


def test_shadowing_mid_cell_drops_only_later_framework_calls() -> None:
    result = analyze("import torch\na = torch.zeros(2)\ntorch = backend\nb = torch.ones(2)")
    assert [op.subject for op in result.operations if op.kind == "tensor_creation"] == ["zeros"]


def test_constant_value_is_taken_at_call_position() -> None:
    result = analyze("import torch\nseed = 7\ntorch.cuda.manual_seed_all(seed)\nseed = 42")
    claims = [claim.text for claim in result.claims if claim.rule == "pytorch.cuda_manual_seed_all"]
    assert len(claims) == 1
    assert "value 7" in claims[0]
    assert "value 42" not in claims[0]


def test_cross_cell_rebind_does_not_change_prior_cell_explanation() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "import torch as t\nseed = 17",
        "t.cuda.manual_seed_all(seed)\nseed = 100",
        "t.cuda.manual_seed_all(seed)",
    ])
    claims_1 = [claim.text for claim in notebook.results[1].claims if claim.rule == "pytorch.cuda_manual_seed_all"]
    claims_2 = [claim.text for claim in notebook.results[2].claims if claim.rule == "pytorch.cuda_manual_seed_all"]
    assert "value 17" in claims_1[0]
    assert "value 100" in claims_2[0]


def test_reassigned_constructor_between_statements_is_not_reused() -> None:
    result = analyze("from torch import nn\ncriterion = nn.CrossEntropyLoss()\nloss1 = criterion(x, y)\ncriterion = custom\nloss2 = criterion(x, y)")
    assert sum(op.kind == "loss_computation" for op in result.operations) == 1
