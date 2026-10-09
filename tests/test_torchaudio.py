from astscribe import NotebookAnalyzer, analyze


def test_torchaudio_load_resample_and_mel_features_with_aliases() -> None:
    result = analyze(
        "import torchaudio as ta\n"
        "from torchaudio.transforms import MelSpectrogram as Mel\n"
        "waveform, sr = ta.load('voice.wav')\n"
        "resampled = ta.functional.resample(waveform, sr, 16000)\n"
        "mel = Mel(sample_rate=16000, n_mels=80)\n"
    )
    kinds = {op.kind for op in result.operations}
    assert {"audio_loading", "audio_signal_operation", "audio_transform_configuration"} <= kinds
    transform = next(op for op in result.operations if op.kind == "audio_transform_configuration")
    assert transform.attributes["sample_rate"] == 16000
    assert transform.attributes["n_mels"] == 80
    assert transform.evidence is not None and transform.evidence.line_start == 5


def test_audio_datasets_model_and_pipeline_bundle() -> None:
    result = analyze(
        "from torchaudio.datasets import LIBRISPEECH\n"
        "from torchaudio.models import Conformer\n"
        "from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H as bundle\n"
        "dataset = LIBRISPEECH('/data')\n"
        "encoder = Conformer(input_dim=80, num_heads=4, ffn_dim=128, num_layers=2, depthwise_conv_kernel_size=31)\n"
        "asr = bundle.get_model()\n"
    )
    kinds = {op.kind for op in result.operations}
    assert {"audio_dataset_configuration", "audio_model_configuration", "audio_pretrained_bundle"} <= kinds
    assert any(op.attributes.get("input_dim") == 80 for op in result.operations)
    assert any("WAV2VEC2_ASR_BASE_960H" in (op.subject or "") for op in result.operations)


def test_audio_cross_cell_context_and_pipeline() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "import torchaudio.transforms as T\nfrom torchaudio import datasets",
        "audio = datasets.SPEECHCOMMANDS('/tmp', download=False)\n"
        "transform = T.Resample(48000, 16000)",
    ])
    assert {op.kind for op in notebook.results[1].operations} >= {
        "audio_dataset_configuration", "audio_transform_configuration"
    }
    transform = next(op for op in notebook.results[1].operations
                     if op.kind == "audio_transform_configuration")
    assert transform.attributes["orig_freq"] == 48000
    assert transform.attributes["new_freq"] == 16000
    assert {"dataset", "preprocessing"} <= {stage.key for stage in notebook.pipeline().stages}


def test_torchaudio_kaldi_features_and_save_via_new_api() -> None:
    result = analyze(
        "import torchaudio\n"
        "from torchaudio.compliance.kaldi import fbank\n"
        "features = fbank(waveform, sample_frequency=16000)\n"
        "torchaudio.save_with_torchcodec('processed.wav', waveform, 16000)\n"
    )
    kinds = {op.kind for op in result.operations}
    assert {"audio_feature_extraction", "audio_export"} <= kinds


def test_unrelated_audio_package_does_not_trigger_false_claims() -> None:
    result = analyze(
        "from audiokit.transforms import Resample\n"
        "from audiokit.datasets import LIBRISPEECH\n"
        "Resample(48000, 16000)\n"
        "LIBRISPEECH('/data')\n"
    )
    assert not any(op.framework == "torchaudio" for op in result.operations)
