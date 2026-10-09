from astscribe import NotebookAnalyzer, analyze


def test_v2_multitask_augments_are_recognized_with_pipeline_evidence() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from torchvision.transforms import v2",
        "augment = v2.Compose([\n"
        "    v2.ToImage(),\n"
        "    v2.RandomPhotometricDistort(p=0.7),\n"
        "    v2.SanitizeBoundingBoxes(min_size=2),\n"
        "    v2.CutMix(num_classes=10),\n"
        "])",
    ])
    result = notebook.results[1]
    ops = [op for op in result.operations if op.kind == "preprocessing_transform"]
    assert {op.subject for op in ops} >= {
        "ToImage", "RandomPhotometricDistort", "SanitizeBoundingBoxes", "CutMix",
    }
    pipeline = next(op for op in result.operations if op.kind == "preprocessing_pipeline")
    assert pipeline.attributes["transforms"] == [
        "ToImage", "RandomPhotometricDistort", "SanitizeBoundingBoxes", "CutMix",
    ]
    assert any(claim.rule == "pytorch.preprocessing_transform"
               and "stochastic" in claim.text
               for claim in result.claims)
    assert all(op.evidence is not None and op.evidence.cell == 1 for op in ops)


def test_functional_calls_not_misclassified_as_transform_constructors() -> None:
    result = analyze(
        "from torchvision.transforms.v2 import functional as VF\n"
        "image = VF.resize(image, size=[224, 224])\n"
    )
    assert any(op.kind == "vision_functional_transform" for op in result.operations)
    assert not any(op.kind == "preprocessing_transform" for op in result.operations)


def test_unknown_vision_symbols_and_nested_model_paths_are_not_invented() -> None:
    result = analyze(
        "import torchvision as tv\n"
        "a = tv.transforms.v2.NotARealTransform()\n"
        "b = tv.models.custom.resnet18()\n"
        "c = tv.transforms.custom.Resize(224)\n"
        "d = tv.models.not_a_real_model()\n"
    )
    assert not any(op.kind in {
        "preprocessing_transform", "model_configuration", "preprocessing_pipeline"
    } for op in result.operations)
    assert not any(op.kind == "vision_model_configuration" for op in result.operations)


def test_known_classification_model_still_is_recognized() -> None:
    result = analyze("from torchvision import models\nmodel = models.resnet50(weights=None)")
    assert any(op.kind == "model_configuration" and op.subject == "resnet50"
               for op in result.operations)
