from astscribe import NotebookAnalyzer, analyze


def test_combined_torch_vision_audio_notebook_has_grounded_operations() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "import torch\nimport torchvision\nimport torchaudio",
        "vision_data = torchvision.datasets.Food101('/data')\n"
        "vision_model = torchvision.models.resnet18(weights=None)\n"
        "kept = torchvision.ops.nms(boxes, scores, 0.5)",
        "waveform, sample_rate = torchaudio.load('sound.wav')\n"
        "mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)\n"
        "model = torch.compile(vision_model, dynamic=True)",
    ])
    result = notebook.results
    frameworks = {op.framework for cell in result for op in cell.operations}
    assert {"pytorch", "torchvision", "torchaudio"} <= frameworks
    kinds = {op.kind for cell in result for op in cell.operations}
    assert {"vision_dataset_configuration", "vision_model_configuration",
            "vision_operator", "audio_loading", "audio_transform_configuration",
            "graph_compilation"} <= kinds
    assert any(op.attributes.get("dynamic") is True for op in result[2].operations)
    assert all(op.evidence is not None and op.evidence.cell is not None
               for cell in result for op in cell.operations)


def test_shadowed_framework_import_does_not_create_tensor_claims() -> None:
    result = analyze(
        "import torch as t\n"
        "t = some_library\n"
        "image = t.randn(3, 224, 224)\n"
    )
    assert not any(op.kind == "tensor_creation" for op in result.operations)


def test_audio_and_vision_unrelated_names_do_not_masquerade_as_torch() -> None:
    result = analyze(
        "from fake_audio.transforms import MelSpectrogram\n"
        "from fake_vision.ops import nms\n"
        "spec = MelSpectrogram(sample_rate=8000)\n"
        "kept = nms(boxes, scores, 0.3)\n"
    )
    assert not any(op.framework in {"torchvision", "torchaudio"} for op in result.operations)
