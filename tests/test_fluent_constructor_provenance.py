"""Regression tests for constructor provenance after self-assignment method calls."""

from astscribe import NotebookAnalyzer


def test_state_dict_self_assignment_invalidates_model_provenance() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from transformers import AutoModel\n"
        "model = AutoModel.from_pretrained('bert-base-uncased')\n"
        "model = model.state_dict()",
        "model.generate(input_ids)",
    ])
    assert not any(op.kind == "text_generation" for op in notebook.results[1].operations)
    assert not any(claim.rule == "transformers.generate" for claim in notebook.results[1].claims)


def test_unknown_self_assignment_invalidates_optimizer_provenance() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "import torch\n"
        "opt = torch.optim.AdamW(params, lr=0.01)\n"
        "opt = opt.state_dict()",
        "opt.step()",
    ])
    assert not any(op.kind == "parameter_update" for op in notebook.results[1].operations)


def test_known_fluent_model_assignment_retains_provenance() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from transformers import AutoModelForCausalLM\n"
        "model = AutoModelForCausalLM.from_pretrained('gpt2')\n"
        "model = model.to('cpu')",
        "model.generate(input_ids)",
    ])
    assert any(claim.rule == "transformers.generate" for claim in notebook.results[1].claims)
