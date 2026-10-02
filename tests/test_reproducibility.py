from astscribe import NotebookAnalyzer, analyze


def test_cuda_seed_and_deterministic_algorithms_are_detected() -> None:
    result = analyze(
        "import torch\n"
        "seed = 123\n"
        "torch.cuda.manual_seed_all(seed)\n"
        "torch.use_deterministic_algorithms(True, warn_only=True)\n"
    )

    rules = {claim.rule for claim in result.claims}
    assert "pytorch.cuda_manual_seed_all" in rules
    assert "pytorch.use_deterministic_algorithms" in rules
    assert any("123" in claim.text for claim in result.claims)
    assert any("warning-only" in claim.text for claim in result.claims)


def test_cudnn_reproducibility_flags_are_reported() -> None:
    result = analyze(
        "import torch\n"
        "torch.backends.cudnn.deterministic = True\n"
        "torch.backends.cudnn.benchmark = False\n"
    )

    rules = {claim.rule for claim in result.claims}
    assert "pytorch.cudnn_deterministic" in rules
    assert "pytorch.cudnn_benchmark" in rules
    assert any("deterministic execution is explicitly enabled" in claim.text for claim in result.claims)
    assert any("benchmarking is explicitly disabled" in claim.text for claim in result.claims)


def test_reproducibility_controls_are_grouped_in_methods() -> None:
    notebook = NotebookAnalyzer.from_cells(
        [
            "import torch\n"
            "torch.cuda.manual_seed_all(7)\n"
            "torch.use_deterministic_algorithms(True)\n"
            "torch.backends.cudnn.deterministic = True\n"
        ]
    )

    report = notebook.render_methodology(include_evidence=True)
    assert "## Reproducibility" in report
    assert "pytorch.cuda_manual_seed_all" in report
    assert "pytorch.use_deterministic_algorithms" in report
    assert "pytorch.cudnn_deterministic" in report
