from astscribe import NotebookAnalyzer, analyze


def test_dataset_detection_and_segmentation_architecture() -> None:
    result = analyze(
        "import torchvision as tv\n"
        "data = tv.datasets.CocoDetection('/data', annFile='labels.json')\n"
        "model = tv.models.detection.fasterrcnn_resnet50_fpn(weights=None)\n"
    )
    kinds = {op.kind for op in result.operations}
    assert "vision_dataset_configuration" in kinds
    assert "vision_model_configuration" in kinds
    dataset = next(op for op in result.operations if op.kind == "vision_dataset_configuration")
    assert dataset.attributes["annFile"] == "labels.json"
    assert dataset.evidence is not None and dataset.evidence.line_start == 2
    assert any(op.subject.endswith("fasterrcnn_resnet50_fpn") for op in result.operations)


def test_vision_ops_functional_v2_and_typed_tensors() -> None:
    result = analyze(
        "from torchvision import ops\n"
        "from torchvision.transforms.v2 import functional as VF\n"
        "from torchvision.tv_tensors import BoundingBoxes\n"
        "kept = ops.nms(boxes, scores, 0.4)\n"
        "out = VF.resize(image, size=[224, 224])\n"
        "coords = BoundingBoxes(raw, format='XYXY', canvas_size=(400, 400))\n"
    )
    kinds = {op.kind for op in result.operations}
    assert {"vision_operator", "vision_functional_transform", "vision_typed_tensor"} <= kinds
    assert any(op.attributes.get("format") == "XYXY" for op in result.operations)


def test_io_and_notebook_context() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "import torchvision.io as io",
        "image = io.decode_jpeg(buf)\nio.write_png(image, 'out.png')",
    ])
    second = notebook.results[1]
    assert len([op for op in second.operations if op.kind == "vision_image_io"]) == 2
    assert "preprocessing" in {stage.key for stage in notebook.pipeline().stages}


def test_known_legacy_transforms_and_modern_v2_are_kept() -> None:
    result = analyze(
        "from torchvision.transforms import v2\n"
        "aug = v2.RandomIoUCrop()\n"
        "pipe = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32)])\n"
    )
    assert any(op.kind == "preprocessing_transform" for op in result.operations)


def test_same_names_from_unrelated_packages_do_not_count() -> None:
    result = analyze(
        "from customvision.ops import nms\n"
        "from customvision.datasets import CocoDetection\n"
        "nms(boxes, scores, 0.5)\n"
        "data = CocoDetection('/data')\n"
    )
    assert not any(op.framework == "torchvision" for op in result.operations)
