from astscribe import NotebookAnalyzer


def test_pipeline_reconstructs_ordered_experiment_stages() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell(
        "from torchvision import datasets, transforms, models\n"
        "from torch import nn\n"
        "transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])\n"
        "dataset = datasets.ImageFolder(root='images', transform=transform)\n"
    )
    notebook.add_cell(
        "model = models.resnet50(weights='DEFAULT')\n"
        "model.fc = nn.Linear(model.fc.in_features, 3)\n"
    )
    notebook.add_cell(
        "criterion = nn.CrossEntropyLoss()\n"
        "optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)\n"
    )

    pipeline = notebook.pipeline()
    keys = [stage.key for stage in pipeline.stages]
    assert keys[:5] == ["dataset", "preprocessing", "model", "objective", "optimization"]

    rendered = notebook.render_pipeline()
    assert "Dataset" in rendered
    assert "Preprocessing and augmentation" in rendered
    assert "Model architecture" in rendered
    assert "Objective" in rendered
    assert "Optimization" in rendered


def test_pipeline_is_serializable() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from torchvision import datasets\ndataset = datasets.MNIST(root='data', train=True)",
    ])

    payload = notebook.pipeline().to_dict()
    assert payload["stages"][0]["key"] == "dataset"
    assert payload["stages"][0]["operations"][0]["kind"] == "dataset_configuration"
