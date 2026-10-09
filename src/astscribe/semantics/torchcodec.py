"""Recognize import-resolved TorchCodec multimedia workflows without decoding media."""
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


_DECODERS = {"AudioDecoder", "VideoDecoder", "WavDecoder"}
_ENCODERS = {"AudioEncoder", "VideoEncoder", "Encoder"}
_IMAGE_DECODERS = {
    "decode_image", "decode_jpeg", "decode_png", "decode_webp",
    "decode_gif", "decode_avif", "decode_heic",
}
_SAMPLERS = {
    "clips_at_regular_indices", "clips_at_random_indices",
    "clips_at_regular_timestamps", "clips_at_random_timestamps",
}
_DECODE_METHODS = {
    "get_all_samples", "get_samples_played_in_range", "get_frames_at",
    "get_frames_in_range", "get_frames_played_at", "get_frames_played_in_range",
}
_ENCODE_METHODS = {"to_file", "to_tensor"}
_STREAM_METHODS = {"add_audio", "add_video", "open_file", "open_file_like"}
_POSITIONAL = {
    "AudioDecoder": ("source",),
    "VideoDecoder": ("source",),
    "WavDecoder": ("source",),
}


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _literal_args(
    node: ast.Call, symbols: SymbolTable, method: str
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, arg in zip(_POSITIONAL.get(method, ()), node.args, strict=False):
        value = symbols.resolve_constant(arg)
        if value is not None:
            attrs[key] = value
    for keyword in node.keywords:
        if keyword.arg is None:
            continue
        value = symbols.resolve_constant(keyword.value)
        if value is not None:
            attrs[keyword.arg] = value
    return attrs


def _classify(
    node: ast.Call, resolved_path: str, symbols: SymbolTable
) -> tuple[str, str] | None:
    if resolved_path.startswith("torchcodec.decoders.") and resolved_path.count(".") == 2:
        name = resolved_path.rsplit(".", 1)[-1]
        if name in _DECODERS:
            return "media_decoder_configuration", f"A TorchCodec {name} is constructed"
        if name in _IMAGE_DECODERS:
            return "image_decoding", f"TorchCodec image decoding through {name} is invoked"
    if resolved_path.startswith("torchcodec.encoders.") and resolved_path.count(".") == 2:
        name = resolved_path.rsplit(".", 1)[-1]
        if name in _ENCODERS:
            return "media_encoder_configuration", f"A TorchCodec {name} is constructed"
    if resolved_path.startswith("torchcodec.samplers.") and resolved_path.count(".") == 2:
        name = resolved_path.rsplit(".", 1)[-1]
        if name in _SAMPLERS:
            return "video_clip_sampling", f"Video clip sampling via {name} is invoked"

    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        name = node.func.value.id
        constructor = symbols.resolve_constructor(name)
        if constructor is None:
            return None
        method = node.func.attr
        if constructor in {f"torchcodec.decoders.{name}" for name in _DECODERS}:
            if method in _DECODE_METHODS:
                return "media_frame_access", f"Frames or samples are requested from {name}"
        if constructor in {f"torchcodec.encoders.{name}" for name in _ENCODERS}:
            if method in _ENCODE_METHODS | _STREAM_METHODS:
                return "media_encoding_operation", f"TorchCodec encoder operation {method} is invoked"
    return None


def analyze_torchcodec(
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
        match = _classify(node, path, symbols)
        if match is None:
            continue
        kind, description = match
        ev = Evidence(
            level=EvidenceLevel.E2 if kind.endswith("configuration") else EvidenceLevel.E3,
            rule=f"torchcodec.{kind}",
            source=parsed.source_segment(node),
            line_start=getattr(node, "lineno", None),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
            cell=parsed.cell,
        )
        attrs = _literal_args(node, symbols, path.rsplit(".", 1)[-1])
        attrs["api"] = path
        operations.append(Operation(kind, "torchcodec", subject=dotted, attributes=attrs, evidence=ev))
        claims.append(Claim(description + ".", ev))
    return SemanticOutput(operations, claims)
