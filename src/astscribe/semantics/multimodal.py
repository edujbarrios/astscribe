"""Conservative, evidence-backed rules for image and vision-language workflows.

No image is opened, model loaded or multimodal generation executed by ASTScribe.
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


_VISION_LANGUAGE_CLASSES = {
    "AutoModelForVision2Seq",
    "AutoModelForImageTextToText",
    "AutoModelForVisualQuestionAnswering",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "Idefics2ForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "BlipForConditionalGeneration",
    "Blip2ForConditionalGeneration",
    "InstructBlipForConditionalGeneration",
    "Pix2StructForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "VisionEncoderDecoderModel",
    "CLIPModel",
    "SiglipModel",
    "Siglip2Model",
}
_NON_GENERATIVE = {"CLIPModel", "SiglipModel", "Siglip2Model"}
_VISION_TASKS = {
    "image-to-text", "image-text-to-text",
    "visual-question-answering", "zero-shot-image-classification",
}
_IMAGE_APIS = {
    "PIL.Image.open": "image_input",
    "PIL.Image.fromarray": "image_input",
    "torchvision.io.read_image": "image_input",
    "torchvision.io.decode_image": "image_input",
    "torchvision.io.read_video": "video_input",
    "cv2.imread": "image_input",
}
_MODALITY_KWARGS = {"images", "image", "videos", "video", "text", "input_ids", "pixel_values"}


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _class_from_pretrained(path: str | None) -> str | None:
    if path and path.startswith("transformers.") and path.endswith(".from_pretrained"):
        return path.removesuffix(".from_pretrained").rsplit(".", 1)[-1]
    return None


def _processor_constructor(path: str | None) -> bool:
    name = _class_from_pretrained(path)
    return bool(name and (
        name.endswith("Processor")
        or name.endswith("ImageProcessor")
        or name.endswith("FeatureExtractor")
    ))


def _vlm_constructor(path: str | None) -> str | None:
    name = _class_from_pretrained(path)
    return name if name in _VISION_LANGUAGE_CLASSES else None


def _keywords(call: ast.Call, symbols: SymbolTable) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = symbols.resolve_constant(keyword.value)
        if value is not None:
            values[keyword.arg] = value
    return values


def _pretrained_id(call: ast.Call, symbols: SymbolTable) -> str | None:
    if call.args:
        value = symbols.resolve_constant(call.args[0])
        if isinstance(value, str):
            return value
    for keyword in call.keywords:
        if keyword.arg in {"pretrained_model_name_or_path", "model_name_or_path"}:
            value = symbols.resolve_constant(keyword.value)
            if isinstance(value, str):
                return value
    return None


def _bound_constructor(node: ast.AST, symbols: SymbolTable) -> str | None:
    if isinstance(node, ast.Name):
        return symbols.resolve_constructor(node.id)
    return None


def analyze_multimodal(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []

    def record(
        node: ast.Call,
        kind: str,
        subject: str,
        message: str,
        attributes: dict[str, Any] | None = None,
        level: EvidenceLevel = EvidenceLevel.E3,
    ) -> None:
        ev = Evidence(
            level=level,
            rule=f"multimodal.{kind}",
            source=parsed.source_segment(node),
            line_start=getattr(node, "lineno", None),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
            cell=parsed.cell,
        )
        operations.append(
            Operation(
                kind,
                "multimodal",
                subject=subject,
                attributes=attributes if attributes is not None else {},
                evidence=ev,
            )
        )
        claims.append(Claim(message, ev))

    for node in ast.walk(parsed.tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        if dotted is None:
            continue
        path = imports.resolve_dotted(dotted)

        if path in _IMAGE_APIS:
            kind = _IMAGE_APIS[path]
            medium = "video" if kind == "video_input" else "image"
            record(node, kind, path, f"An explicit {medium} input-loading API is invoked.")
            continue

        model_name = _vlm_constructor(path)
        if model_name is not None:
            attributes = _keywords(node, symbols)
            identifier = _pretrained_id(node, symbols)
            if identifier is not None:
                attributes["pretrained_model_name_or_path"] = identifier
            record(
                node, "vision_language_model_configuration", model_name,
                f"A {model_name} vision-language architecture is selected using from_pretrained.",
                attributes, EvidenceLevel.E2,
            )
            continue

        if path == "transformers.pipeline" and node.args:
            task = symbols.resolve_constant(node.args[0])
            if isinstance(task, str) and task in _VISION_TASKS:
                record(
                    node, "vision_language_pipeline", task,
                    f"A Transformers pipeline is configured for the {task} task.",
                    {"task": task}, EvidenceLevel.E2,
                )
            continue

        if isinstance(node.func, ast.Name):
            constructor = _bound_constructor(node.func, symbols)
            if _processor_constructor(constructor):
                keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
                if keywords.intersection({"images", "image", "videos", "video"}):
                    record(
                        node, "multimodal_input_processing", node.func.id,
                        "A configured Transformers processor is called with explicitly named visual inputs.",
                        {"has_images": bool(keywords & {"images", "image"}),
                         "has_videos": bool(keywords & {"videos", "video"}),
                         "has_text": "text" in keywords,
                         **_keywords(node, symbols)},
                    )
                    continue

        if not isinstance(node.func, ast.Attribute):
            continue

        constructor = _bound_constructor(node.func.value, symbols)
        if node.func.attr == "apply_chat_template" and _processor_constructor(constructor):
            record(
                node, "multimodal_chat_template", dotted,
                "A configured processor formats a conversation with apply_chat_template; "
                "the actual message modalities are not statically established.",
                _keywords(node, symbols),
            )
            continue

        model_class = _vlm_constructor(constructor)
        if model_class is None:
            continue

        if node.func.attr == "generate" and model_class not in _NON_GENERATIVE:
            keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
            attributes = _keywords(node, symbols)
            attributes["visual_kwargs_explicit"] = bool(
                keywords & {"images", "image", "videos", "video", "pixel_values"}
            )
            attributes["unpacked_inputs"] = any(kw.arg is None for kw in node.keywords)
            record(
                node, "multimodal_generation", dotted,
                f"Generation is invoked on a configured {model_class} model. "
                "This does not establish the visual content or output quality.",
                attributes,
            )
        elif node.func.attr == "forward":
            record(
                node, "multimodal_forward_pass", dotted,
                f"The forward method of a configured {model_class} model is called.",
                {"explicit_modalities": sorted(
                    kw.arg for kw in node.keywords
                    if kw.arg is not None and kw.arg in _MODALITY_KWARGS
                )},
            )
    return SemanticOutput(operations=operations, claims=claims)
