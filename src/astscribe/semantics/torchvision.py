"""Source-based torchvision dataset, vision models, operators and image IO.

This is a deliberately curated public-API catalog. It is *not* evidence of
downloaded datasets, instantiated models or processed images at runtime.
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
    "CocoDetection", "CocoCaptions", "VOCDetection", "VOCSegmentation",
    "Cityscapes", "SBDataset", "ImageNet", "Places365", "Country211",
    "Caltech101", "Caltech256", "CelebA", "EMNIST", "KMNIST", "QMNIST",
    "USPS", "SVHN", "STL10", "LSUN", "LSUNClass", "FakeData",
    "FashionMNIST", "MNIST", "CIFAR10", "CIFAR100", "ImageFolder",
    "DatasetFolder", "Food101", "Flowers102", "OxfordIIITPet", "GTSRB",
    "EuroSAT", "DTD", "FGVCAircraft", "StanfordCars", "SUN397",
    "RenderedSST2", "INaturalist", "SBU", "LFWPeople", "LFWPairs",
    "WIDERFace", "Kitti", "Kinetics", "Kinetics400", "UCF101",
    "HMDB51", "MovingMNIST", "FER2013", "PCAM", "SEMEION",
}
_PREEXISTING_DATASETS = {"CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "ImageFolder"}
_MODEL_NAMES = {
    "alexnet", "vgg11", "vgg13", "vgg16", "vgg19",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2",
    "wide_resnet101_2", "densenet121", "densenet161", "densenet169",
    "densenet201", "mobilenet_v2", "mobilenet_v3_large",
    "mobilenet_v3_small", "efficientnet_b0", "efficientnet_b1",
    "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
    "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
    "squeezenet1_0", "squeezenet1_1", "shufflenet_v2_x1_0",
    "googlenet", "inception_v3", "mnasnet1_0", "regnet_y_8gf",
    "regnet_x_8gf", "convnext_tiny", "convnext_small", "convnext_base",
    "convnext_large", "vit_b_16", "vit_b_32", "vit_l_16",
    "swin_t", "swin_s", "swin_b", "swin_v2_t",
    "maxvit_t", "get_model",
    "fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_fpn",
    "retinanet_resnet50_fpn", "fcos_resnet50_fpn", "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
    "maskrcnn_resnet50_fpn", "keypointrcnn_resnet50_fpn",
    "deeplabv3_resnet50", "deeplabv3_resnet101",
    "fcn_resnet50", "fcn_resnet101", "lraspp_mobilenet_v3_large",
    "r3d_18", "mc3_18", "r2plus1d_18", "s3d", "mvit_v1_b", "mvit_v2_s",
    "swin3d_t", "swin3d_s", "raft_large", "raft_small",
}
_OPS = {
    "nms", "batched_nms", "box_area", "box_iou", "generalized_box_iou",
    "complete_box_iou", "distance_box_iou", "box_convert",
    "clip_boxes_to_image", "remove_small_boxes", "masks_to_boxes",
    "roi_align", "roi_pool", "ps_roi_align", "ps_roi_pool",
    "deform_conv2d", "stochastic_depth", "drop_block2d", "drop_block3d",
    "sigmoid_focal_loss", "generalized_box_iou_loss", "complete_box_iou_loss",
    "distance_box_iou_loss", "feature_pyramid_network",
    "MultiScaleRoIAlign", "RoIAlign", "RoIPool", "PSRoIAlign",
    "PSRoIPool", "DeformConv2d", "FrozenBatchNorm2d",
    "FeaturePyramidNetwork", "SqueezeExcitation", "StochasticDepth",
    "Conv2dNormActivation", "Conv3dNormActivation", "MLP",
    "Permute",
}
_FUNC_TRANSFORMS = {
    "resize", "crop", "center_crop", "resized_crop", "pad",
    "hflip", "vflip", "rotate", "affine", "perspective",
    "normalize", "to_tensor", "pil_to_tensor", "convert_image_dtype",
    "to_pil_image", "rgb_to_grayscale", "adjust_brightness",
    "adjust_contrast", "adjust_saturation", "adjust_hue", "adjust_gamma",
    "adjust_sharpness", "posterize", "solarize", "autocontrast",
    "equalize", "invert", "gaussian_blur", "elastic_transform",
    "erase", "five_crop", "ten_crop", "get_image_size",
    "get_dimensions", "get_num_channels", "to_dtype",
    "to_image", "clamp_bounding_boxes", "convert_bounding_box_format",
    "resize_bounding_boxes", "crop_bounding_boxes", "sanitize_bounding_boxes",
    "resize_mask", "resize_video",
}
_IMAGE_IO = {
    "decode_jpeg", "decode_png", "decode_webp", "decode_gif",
    "encode_jpeg", "encode_png", "write_jpeg", "write_png",
    "read_file", "write_file", "read_video_timestamps",
}
_TV_TENSORS = {"Image", "Video", "Mask", "BoundingBoxes", "wrap"}


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _static_kwargs(call: ast.Call, symbols: SymbolTable) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is not None:
            value = symbols.resolve_constant(keyword.value)
            if value is not None:
                result[keyword.arg] = value
    return result


def _classify(path: str) -> tuple[str, str] | None:
    if path.startswith("torchvision.datasets.") and path.count(".") == 2:
        name = path.rsplit(".", 1)[-1]
        if name in _DATASETS and name not in _PREEXISTING_DATASETS:
            return ("vision_dataset_configuration", "A torchvision dataset constructor is invoked")
    if path.startswith("torchvision.models."):
        name = path.rsplit(".", 1)[-1]
        if name in _MODEL_NAMES:
            return ("vision_model_configuration", "A torchvision model architecture is selected")
    if path.startswith("torchvision.ops.") and path.count(".") == 2:
        if path.rsplit(".", 1)[-1] in _OPS:
            return ("vision_operator", "A torchvision computer-vision operation is invoked")
    if path.startswith(("torchvision.transforms.functional.", "torchvision.transforms.v2.functional.")):
        if path.rsplit(".", 1)[-1] in _FUNC_TRANSFORMS:
            return ("vision_functional_transform", "A torchvision functional image or geometry transform is invoked")
    if path.startswith("torchvision.io.") and path.count(".") == 2:
        if path.rsplit(".", 1)[-1] in _IMAGE_IO:
            return ("vision_image_io", "A torchvision image or file I/O API is invoked")
    if path.startswith("torchvision.tv_tensors.") and path.count(".") == 2:
        if path.rsplit(".", 1)[-1] in _TV_TENSORS:
            return ("vision_typed_tensor", "A torchvision image/video/geometry tensor wrapper is invoked")
    return None


def analyze_torchvision(
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
        attributes = _static_kwargs(node, symbols)
        attributes["api"] = path
        ev = Evidence(
            level=EvidenceLevel.E2 if kind.endswith("configuration") else EvidenceLevel.E3,
            rule=f"torchvision.{kind}",
            source=parsed.source_segment(node),
            line_start=getattr(node, "lineno", None),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
            cell=parsed.cell,
        )
        operations.append(Operation(kind, "torchvision", subject=path, attributes=attributes, evidence=ev))
        claims.append(Claim(f"{description}: {path}.", ev))
    return SemanticOutput(operations=operations, claims=claims)
