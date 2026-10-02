from astscribe import NotebookAnalyzer, analyze


def test_torchvision_dataset_and_preprocessing_are_detected() -> None:
    result = analyze(
        "from torchvision import datasets, transforms\n"
        "transform = transforms.Compose([\n"
        "    transforms.Resize((224, 224)),\n"
        "    transforms.RandomHorizontalFlip(),\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),\n"
        "])\n"
        "train_dataset = datasets.ImageFolder(root='data/train', transform=transform)\n"
    )

    kinds = {operation.kind for operation in result.operations}
    rules = {claim.rule for claim in result.claims}
    assert "dataset_configuration" in kinds
    assert "preprocessing_pipeline" in kinds
    assert "preprocessing_transform" in kinds
    assert "pytorch.dataset_configuration" in rules
    assert "pytorch.preprocessing_transform" in rules
    assert any("stochastic data-augmentation" in claim.text for claim in result.claims)
    assert any("normalized" in claim.text for claim in result.claims)


def test_torchvision_model_and_replaced_head_are_detected() -> None:
    result = analyze(
        "from torchvision import models\n"
        "from torch import nn\n"
        "model = models.resnet50(weights='DEFAULT')\n"
        "num_classes = 4\n"
        "model.fc = nn.Linear(model.fc.in_features, num_classes)\n"
    )

    configuration = next(
        operation for operation in result.operations if operation.kind == "model_configuration"
    )
    replacement = next(
        operation for operation in result.operations if operation.kind == "model_head_replacement"
    )
    assert configuration.subject == "resnet50"
    assert configuration.attributes["weights"] == "DEFAULT"
    assert replacement.subject == "model.fc"
    assert replacement.attributes["out_features"] == 4


def test_random_split_is_reported_without_executing_it() -> None:
    result = analyze(
        "import torch\n"
        "train_dataset, val_dataset = torch.utils.data.random_split(dataset, [800, 200])\n"
    )

    split = next(operation for operation in result.operations if operation.kind == "dataset_split")
    assert split.attributes["lengths"] == [800, 200]
    assert "train_dataset, val_dataset" in split.attributes["targets"]


def test_metric_and_prediction_selection_are_detected() -> None:
    result = analyze(
        "from torchmetrics.classification import MulticlassAccuracy\n"
        "metric = MulticlassAccuracy(num_classes=10)\n"
        "predictions = outputs.argmax(dim=1)\n"
    )

    kinds = {operation.kind for operation in result.operations}
    assert "metric_configuration" in kinds
    assert "prediction_selection" in kinds
    prediction = next(
        operation for operation in result.operations if operation.kind == "prediction_selection"
    )
    assert prediction.attributes["dim"] == 1


def test_methodology_report_reconstructs_experiment_stages() -> None:
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
        "from torchmetrics.classification import MulticlassAccuracy\n"
        "metric = MulticlassAccuracy(num_classes=3)\n"
        "predictions = outputs.argmax(dim=1)\n"
    )

    report = notebook.render_methodology(include_evidence=True)
    assert "## Dataset" in report
    assert "## Preprocessing and augmentation" in report
    assert "## Model architecture" in report
    assert "## Metrics" in report
    assert "pytorch.dataset_configuration" in report
    assert "pytorch.model_configuration" in report


def test_parameter_freezing_is_conservative() -> None:
    result = analyze(
        "for parameter in model.parameters():\n"
        "    parameter.requires_grad = False\n"
    )

    claim = next(claim for claim in result.claims if claim.rule == "pytorch.parameter_freeze")
    assert claim.text == "Gradient computation is explicitly disabled for `parameter`."
    assert "fine-tun" not in claim.text.lower()
