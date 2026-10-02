from astscribe import NotebookAnalyzer


def test_notebook_methods_report_groups_scientific_sections() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell(
        "import torch\n"
        "from torch.optim import AdamW\n"
        "from torch import nn\n"
        "torch.manual_seed(42)\n"
        "learning_rate = 2e-5\n"
        "criterion = nn.CrossEntropyLoss()\n"
        "optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)\n"
    )
    analyzer.add_cell(
        "model.train()\n"
        "optimizer.zero_grad()\n"
        "outputs = model(inputs)\n"
        "loss = criterion(outputs, targets)\n"
        "loss.backward()\n"
        "optimizer.step()\n"
    )

    report = analyzer.methodology()
    markdown = report.render()

    assert markdown.startswith("# Methods")
    assert "## Reproducibility" in markdown
    assert "## Model and objective" in markdown
    assert "## Optimization" in markdown
    assert "## Training procedure" in markdown
    assert "AdamW" in markdown
    assert "CrossEntropyLoss" in markdown


def test_methods_report_can_include_traceable_evidence() -> None:
    analyzer = NotebookAnalyzer()
    analyzer.add_cell("import torch\ntorch.manual_seed(7)")

    markdown = analyzer.render_methodology(include_evidence=True)

    assert "## Evidence" in markdown
    assert "[E1] cell 0, line 2" in markdown
    assert "`torch.manual_seed(7)`" in markdown
    assert "`pytorch.manual_seed`" in markdown
