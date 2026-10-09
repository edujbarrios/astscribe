"""Source-level semantics for public TorchAudio datasets, DSP, models and bundles.

TorchAudio 2.9+ delegates load/save to TorchCodec. Static analysis records
calls but does not assume a codec backend, downloaded data or successful IO.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from astscribe.parser import ImportTable, ParsedSource, SymbolTable
from astscribe.sir import Claim, Evidence, EvidenceLevel, Operation


@dataclass(frozen=True)
class SemanticOutput:
    operations: list[Operation]
    claims: list[Claim]


_DATASETS = {
    "COMMONVOICE", "CMUARCTIC", "CMUDict", "DR_VCTK", "FluentSpeechCommands",
    "GTZAN", "IEMOCAP", "LibriMix", "LIBRISPEECH", "LibriLightLimited",
    "LIBRITTS", "LJSPEECH", "MUSDB_HQ", "QUESST14", "Snips",
    "SPEECHCOMMANDS", "TEDLIUM", "VCTK_092", "VoxCeleb1Identification",
    "VoxCeleb1Verification", "YESNO", "VoxPopuli",
}
_TRANSFORMS = {
    "Spectrogram", "InverseSpectrogram", "GriffinLim", "MelScale",
    "InverseMelScale", "MelSpectrogram", "MFCC", "LFCC", "ComputeDeltas",
    "PitchShift", "Resample", "AmplitudeToDB", "DBToAmplitude",
    "MuLawEncoding", "MuLawDecoding", "Fade", "Vol", "Loudness",
    "AddNoise", "Convolve", "FrequencyMasking", "TimeMasking",
    "TimeStretch", "SlidingWindowCmn", "SpectralCentroid", "Vad",
    "RNNTLoss", "PSD", "MVDR", "RTFMVDR", "SoudenMVDR",
}
_FUNCTIONS = {
    "resample", "spectrogram", "inverse_spectrogram", "griffinlim",
    "melscale_fbanks", "linear_fbanks", "create_dct", "compute_deltas",
    "amplitude_to_DB", "DB_to_amplitude", "vad", "pitch_shift",
    "detect_pitch_frequency", "functional", "highpass_biquad",
    "lowpass_biquad", "bandpass_biquad", "bandreject_biquad",
    "allpass_biquad", "equalizer_biquad", "bass_biquad", "treble_biquad",
    "deemph_biquad", "riaa_biquad", "lfilter", "filtfilt", "biquad",
    "mu_law_encoding", "mu_law_decoding", "add_noise", "convolve",
    "fftconvolve", "apply_codec", "speed", "phase_vocoder",
    "mask_along_axis", "mask_along_axis_iid", "sliding_window_cmn",
    "forced_align", "rnnt_loss", "psd", "mvdr_weights_souden",
    "mvdr_weights_rtf", "rtf_evd", "rtf_power",
}
_MODELS = {
    "Conformer", "ConvTasNet", "DeepSpeech", "Emformer",
    "HDemucs", "HuBERTPretrainModel", "RNNT", "RNNTBeamSearch",
    "SquimObjective", "SquimSubjective", "Tacotron2", "Wav2Letter",
    "Wav2Vec2Model", "WaveRNN",
    "wav2vec2_model", "wav2vec2_base", "wav2vec2_large",
    "wav2vec2_large_lv60k", "hubert_base", "hubert_large",
    "hubert_xlarge", "hifigan_vocoder", "tacotron2",
    "emformer_rnnt_base", "conformer_rnnt_model",
    "conformer_rnnt_base", "conformer_rnnt_tiny",
    "squim_objective_base", "squim_subjective_base",
    "rnnt_model", "hdemucs_high", "hdemucs_low",
}
_KALDI = {"fbank", "mfcc", "spectrogram", "pitch", "resample_waveform"}
_PIPELINE_PREFIXES = (
    "WAV2VEC2_", "HUBERT_", "WAVLM_", "MMS_", "SQUIM_", "SUPERB_",
    "TACOTRON2_", "VOXPOPULI_", "EMFORMER_", "HDEMUCS_",
)
_BUNDLE_METHODS = {"get_model", "get_labels", "get_tokenizer", "get_vocoder", "get_text_processor"}
_POSITIONAL = {
    "Resample": ("orig_freq", "new_freq"),
    "Spectrogram": ("n_fft",),
    "MelSpectrogram": ("sample_rate", "n_fft"),
    "MFCC": ("sample_rate", "n_mfcc"),
}


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _attrs(node: ast.Call, symbols: SymbolTable, subject: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is not None:
            value = symbols.resolve_constant(keyword.value)
            if value is not None:
                attrs[keyword.arg] = value
    for name, arg in zip(_POSITIONAL.get(subject, ()), node.args, strict=False):
        value = symbols.resolve_constant(arg)
        if value is not None:
            attrs.setdefault(name, value)
    return attrs


def _classify(path: str) -> tuple[str, str] | None:
    if path in {"torchaudio.load", "torchaudio.load_with_torchcodec"}:
        return ("audio_loading", "An audio-loading API is invoked")
    if path in {"torchaudio.save", "torchaudio.save_with_torchcodec"}:
        return ("audio_export", "An audio-writing API is invoked")
    name = path.rsplit(".", 1)[-1]
    if path.startswith("torchaudio.datasets.") and path.count(".") == 2:
        if name in _DATASETS:
            return ("audio_dataset_configuration", "A TorchAudio dataset constructor is invoked")
    if path.startswith("torchaudio.transforms.") and path.count(".") == 2:
        if name in _TRANSFORMS:
            if name == "RNNTLoss":
                return ("audio_objective_configuration", "An audio sequence loss module is constructed")
            return ("audio_transform_configuration", "A TorchAudio DSP transform is constructed")
    if path.startswith("torchaudio.functional.") and path.count(".") == 2:
        if name in _FUNCTIONS:
            return ("audio_signal_operation", "A TorchAudio signal-processing function is invoked")
    if path.startswith("torchaudio.models.") and path.count(".") == 2:
        if name in _MODELS:
            return ("audio_model_configuration", "A TorchAudio model API is invoked")
    if path.startswith("torchaudio.compliance.kaldi.") and path.count(".") == 3:
        if name in _KALDI:
            return ("audio_feature_extraction", "A Kaldi-compatible audio feature function is invoked")
    if path.startswith("torchaudio.pipelines.") and path.count(".") == 3:
        bundle = path.split(".")[2]
        if bundle.startswith(_PIPELINE_PREFIXES) and name in _BUNDLE_METHODS:
            return ("audio_pretrained_bundle", "A pretrained audio pipeline bundle method is invoked")
    return None


def analyze_torchaudio(
    parsed: ParsedSource, imports: ImportTable, symbols: SymbolTable
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []
    for node in ast.walk(parsed.tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        if dotted is None:
            continue
        path = imports.resolve_dotted(dotted)
        classification = _classify(path)
        if classification is None:
            continue
        kind, description = classification
        attributes = _attrs(node, symbols, path.rsplit(".", 1)[-1])
        attributes["api"] = path
        ev = Evidence(
            level=EvidenceLevel.E2 if kind.endswith("configuration") else EvidenceLevel.E3,
            rule=f"torchaudio.{kind}",
            source=parsed.source_segment(node),
            line_start=getattr(node, "lineno", None),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
            cell=parsed.cell,
        )
        operations.append(Operation(kind, "torchaudio", subject=path, attributes=attributes, evidence=ev))
        claims.append(Claim(f"{description}: {path}.", ev))
    return SemanticOutput(operations=operations, claims=claims)
