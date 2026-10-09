"""Explicit, source-evidenced PyTorch API families beyond neural layers.

Only import-resolved canonical paths are considered. These rules describe the
*presence of calls*, not their success, tensor dimensions or runtime effects.
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


_DATA_CLASSES = {
    "TensorDataset", "ConcatDataset", "ChainDataset", "Subset",
    "RandomSampler", "SequentialSampler", "SubsetRandomSampler",
    "WeightedRandomSampler", "BatchSampler", "DistributedSampler",
}
_COMPILE = {
    "torch.compile", "torch.jit.script", "torch.jit.trace",
    "torch.jit.freeze", "torch.export.export",
    "torch.export.export_for_training", "torch.fx.symbolic_trace",
}
_AUTOGRAD = {
    "torch.autograd.grad", "torch.autograd.backward",
    "torch.autograd.functional.jacobian", "torch.autograd.functional.hessian",
    "torch.func.grad", "torch.func.grad_and_value", "torch.func.vmap",
    "torch.func.jacrev", "torch.func.jacfwd", "torch.func.hessian",
    "torch.func.functional_call",
}
_DISTRIBUTED = {
    "torch.distributed.init_process_group", "torch.distributed.destroy_process_group",
    "torch.distributed.barrier", "torch.distributed.broadcast",
    "torch.distributed.all_reduce", "torch.distributed.reduce",
    "torch.distributed.all_gather", "torch.distributed.gather",
    "torch.distributed.scatter", "torch.distributed.reduce_scatter",
    "torch.distributed.all_to_all", "torch.distributed.send",
    "torch.distributed.recv", "torch.distributed.broadcast_object_list",
}
_DISTRIBUTED_MODULES = {
    "torch.nn.parallel.DistributedDataParallel",
    "torch.distributed.fsdp.FullyShardedDataParallel",
}
_INIT = {
    "uniform_", "normal_", "constant_", "ones_", "zeros_",
    "eye_", "dirac_", "xavier_uniform_", "xavier_normal_",
    "kaiming_uniform_", "kaiming_normal_", "orthogonal_",
    "sparse_", "trunc_normal_", "calculate_gain",
}
_LINALG = {
    "norm", "vector_norm", "matrix_norm", "svd", "qr", "eig", "eigh",
    "solve", "lstsq", "inv", "pinv", "det", "slogdet", "cholesky",
    "matrix_rank", "cond", "cross", "multi_dot", "matrix_exp",
    "tensorinv", "tensorsolve",
}
_FFT = {"fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "rfft2", "irfft2",
        "fftn", "ifftn", "rfftn", "irfftn", "fftfreq", "rfftfreq", "fftshift", "ifftshift"}
_SPECIAL = {"erf", "erfc", "expit", "gammaln", "digamma", "logsumexp", "softmax",
            "log_softmax", "entr", "xlogy", "i0", "i0e", "sinc"}
_CHECKPOINTS = {
    "torch.utils.checkpoint.checkpoint", "torch.utils.checkpoint.checkpoint_sequential",
}
_CONFIGURATION = {
    "torch.set_default_dtype", "torch.set_default_device",
    "torch.set_float32_matmul_precision", "torch.use_deterministic_algorithms",
    "torch.set_num_threads", "torch.set_num_interop_threads",
}
_PROFILE = {"torch.profiler.profile", "torch.profiler.record_function"}
_OTHER = {
    "torch.hub.load": ("model_hub_loading", "A torch.hub.load invocation is present."),
    "torch.utils.data.default_collate": (
        "batch_collation", "The default PyTorch batching collator is invoked."
    ),
}


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _literal_keywords(call: ast.Call, symbols: SymbolTable) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is not None:
            value = symbols.resolve_constant(kw.value)
            if value is not None:
                attributes[kw.arg] = value
    return attributes


def _classification(path: str) -> tuple[str, str] | None:
    if path in _COMPILE:
        return ("graph_compilation", "A PyTorch graph compilation or tracing API is invoked.")
    if path in _AUTOGRAD:
        return ("autograd_operation", "A PyTorch automatic-differentiation API is invoked.")
    if path in _DISTRIBUTED:
        return ("distributed_communication", "A PyTorch distributed coordination API is invoked.")
    if path in _DISTRIBUTED_MODULES:
        return ("distributed_model_configuration", "A distributed model wrapper is constructed.")
    if path.startswith("torch.utils.data.") and path.rsplit(".", 1)[-1] in _DATA_CLASSES:
        return ("data_pipeline_configuration", "A PyTorch data pipeline component is constructed.")
    if path.startswith("torch.nn.init.") and path.rsplit(".", 1)[-1] in _INIT:
        return ("parameter_initialization", "A PyTorch parameter-initialization API is invoked.")
    if path.startswith("torch.linalg.") and path.rsplit(".", 1)[-1] in _LINALG:
        return ("linear_algebra_operation", "A torch.linalg operation is invoked.")
    if path.startswith("torch.fft.") and path.rsplit(".", 1)[-1] in _FFT:
        return ("frequency_operation", "A torch.fft operation is invoked.")
    if path.startswith("torch.special.") and path.rsplit(".", 1)[-1] in _SPECIAL:
        return ("special_function_operation", "A torch.special operation is invoked.")
    if path in _CHECKPOINTS:
        return ("activation_checkpointing", "A gradient-checkpointing API is invoked.")
    if path in _CONFIGURATION:
        return ("runtime_configuration", "A PyTorch runtime configuration API is invoked.")
    if path in _PROFILE:
        return ("profiling", "A PyTorch profiling API is invoked.")
    return _OTHER.get(path)


def analyze_pytorch_ecosystem(
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
        classified = _classification(path)
        if classified is None:
            continue
        kind, description = classified
        evidence = Evidence(
            level=EvidenceLevel.E3,
            rule=f"pytorch.{kind}",
            source=parsed.source_segment(node),
            line_start=getattr(node, "lineno", None),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
            cell=parsed.cell,
        )
        attributes = _literal_keywords(node, symbols)
        attributes["api"] = path
        operations.append(Operation(kind, "pytorch", subject=path, attributes=attributes, evidence=evidence))
        claims.append(Claim(f"{description} ({path}).", evidence))
    return SemanticOutput(operations=operations, claims=claims)
