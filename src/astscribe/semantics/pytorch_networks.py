"""Evidence-linked semantics for common neural-network and tensor primitives.

ASTScribe inspects syntax only: construction is not evidence of training,
model quality, tensor shapes at runtime, or successful execution.
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


_LAYERS = {
    "Linear", "Bilinear", "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "Embedding", "EmbeddingBag", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d",
    "InstanceNorm3d", "RMSNorm", "Dropout", "Dropout1d", "Dropout2d",
    "Dropout3d", "AlphaDropout", "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "Flatten", "Unflatten",
    "Identity", "Sequential", "ModuleList", "ModuleDict",
    "MultiheadAttention", "Transformer", "TransformerEncoder",
    "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer",
    "RNN", "GRU", "LSTM", "RNNCell", "GRUCell", "LSTMCell",
}
_ACTIVATIONS = {
    "ReLU", "ReLU6", "LeakyReLU", "PReLU", "ELU", "SELU",
    "GELU", "SiLU", "Mish", "Sigmoid", "Tanh", "Softmax", "LogSoftmax",
    "Softplus", "Hardswish", "Hardtanh",
}
_LOSSES = {
    "L1Loss", "SmoothL1Loss", "HuberLoss", "KLDivLoss", "NLLLoss",
    "BCELoss", "CTCLoss", "CosineEmbeddingLoss", "TripletMarginLoss",
    "MarginRankingLoss", "PoissonNLLLoss", "MultiMarginLoss",
}
_FUNCTIONS = {
    "relu", "relu6", "leaky_relu", "gelu", "silu", "mish", "sigmoid",
    "tanh", "softmax", "log_softmax", "dropout", "layer_norm",
    "batch_norm", "group_norm", "conv1d", "conv2d", "conv3d",
    "linear", "embedding", "max_pool2d", "avg_pool2d",
    "scaled_dot_product_attention", "interpolate", "pad",
}
_TENSOR_FACTORIES = {
    "tensor", "as_tensor", "from_numpy", "zeros", "ones", "empty",
    "full", "rand", "randn", "randint", "arange", "linspace",
    "eye", "zeros_like", "ones_like", "empty_like", "full_like",
    "rand_like", "randn_like",
}
_TENSOR_OPS = {"cat", "stack", "flatten", "reshape", "permute", "matmul", "bmm", "einsum"}
_TENSOR_METHODS = {"view", "reshape", "permute", "transpose", "flatten", "unsqueeze", "squeeze"}
_POSITIONAL = {
    "Linear": ("in_features", "out_features"),
    "Conv1d": ("in_channels", "out_channels", "kernel_size"),
    "Conv2d": ("in_channels", "out_channels", "kernel_size"),
    "Conv3d": ("in_channels", "out_channels", "kernel_size"),
    "Embedding": ("num_embeddings", "embedding_dim"),
    "MultiheadAttention": ("embed_dim", "num_heads"),
    "LSTM": ("input_size", "hidden_size", "num_layers"),
    "GRU": ("input_size", "hidden_size", "num_layers"),
}


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _values(call: ast.Call, symbols: SymbolTable, kind: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is not None:
            value = symbols.resolve_constant(keyword.value)
            if value is not None:
                values[keyword.arg] = value
    for name, arg in zip(_POSITIONAL.get(kind, ()), call.args, strict=False):
        value = symbols.resolve_constant(arg)
        if value is not None:
            values.setdefault(name, value)
    return values


def analyze_pytorch_networks(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
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
        kind: str | None = None
        subject = path.rsplit(".", 1)[-1]
        message: str | None = None
        attributes: dict[str, Any] = {}
        if path.startswith("torch.nn.") and path.count(".") == 2:
            if subject in _LAYERS:
                kind = "neural_layer_configuration"
                attributes = _values(node, symbols, subject)
                message = f"A PyTorch {subject} module is constructed."
            elif subject in _ACTIVATIONS:
                kind = "activation_configuration"
                attributes = _values(node, symbols, subject)
                message = f"A {subject} activation module is constructed."
            elif subject in _LOSSES:
                kind = "loss_configuration"
                attributes = _values(node, symbols, subject)
                message = f"A {subject} loss module is constructed."
        elif path.startswith("torch.nn.functional.") and path.count(".") == 3:
            if subject in _FUNCTIONS:
                kind = "attention_operation" if subject == "scaled_dot_product_attention" else (
                    "functional_neural_operation"
                )
                attributes = _values(node, symbols, subject)
                message = f"The PyTorch functional operation {subject} is invoked."
        elif path.startswith("torch.") and path.count(".") == 1:
            if subject in _TENSOR_FACTORIES:
                kind = "tensor_creation"
                attributes = _values(node, symbols, subject)
                message = f"torch.{subject} is called to construct or convert tensor data."
            elif subject in _TENSOR_OPS:
                kind = "tensor_operation"
                message = f"The PyTorch tensor operation {subject} is invoked."
        elif isinstance(node.func, ast.Attribute) and subject in _TENSOR_METHODS:
            # Only infer tensor semantics for a subject statically known to have
            # been assigned using a PyTorch tensor factory, not arbitrary objects.
            root = _dotted(node.func.value)
            constructor = symbols.resolve_constructor(root) if root else None
            if constructor in {f"torch.{name}" for name in _TENSOR_FACTORIES}:
                kind = "tensor_operation"
                message = f"The tensor method {subject} is invoked on a known tensor binding."
        if kind is None or message is None:
            continue
        evidence = Evidence(
            level=EvidenceLevel.E2 if kind.endswith("configuration") else EvidenceLevel.E3,
            rule=f"pytorch.{kind}",
            source=parsed.source_segment(node),
            line_start=getattr(node, "lineno", None),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
            cell=parsed.cell,
        )
        operations.append(
            Operation(kind, "pytorch", subject=subject, attributes=attributes, evidence=evidence)
        )
        claims.append(Claim(message, evidence))
    return SemanticOutput(operations=operations, claims=claims)
