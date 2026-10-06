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
    "torchvision.datasets.ImageFolder",
    "torchvision.datasets.CIFAR10",
    "torchvision.datasets.CIFAR100",
    "torchvision.datasets.MNIST",
    "torchvision.datasets.FashionMNIST",
}
_TRANSFORM_PREFIXES = ("torchvision.transforms.", "torchvision.transforms.v2.")
_STOCHASTIC_TRANSFORMS = {
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "RandomRotation",
    "RandomVerticalFlip",
    "ColorJitter",
    "RandomAffine",
}
_MODEL_PREFIXES = ("torchvision.models.",)
_TORCH_MODULE_PREFIX = "torch.nn."
_METRIC_PREFIXES = ("torchmetrics.",)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _call_path(call: ast.Call, imports: ImportTable) -> str | None:
    name = _dotted_name(call.func)
    return imports.resolve_dotted(name) if name else None


def _evidence(parsed: ParsedSource, node: ast.AST, level: EvidenceLevel, rule: str) -> Evidence:
    return Evidence(
        level=level,
        rule=rule,
        source=parsed.source_segment(node),
        line_start=getattr(node, "lineno", None),
        line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
        cell=parsed.cell,
    )


def _static_value(parsed: ParsedSource, symbols: SymbolTable, node: ast.AST) -> Any | None:
    value = symbols.resolve_constant(node)
    if value is not None:
        return value
    if isinstance(node, ast.Tuple | ast.List):
        values = [_static_value(parsed, symbols, item) for item in node.elts]
        if all(value is not None for value in values):
            return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Name | ast.Attribute):
        return _dotted_name(node)
    return None


def _keyword_values(
    parsed: ParsedSource,
    call: ast.Call,
    symbols: SymbolTable,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = _static_value(parsed, symbols, keyword.value)
        if value is not None:
            values[keyword.arg] = value
    return values


def _assignment_target(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted_name(node)
    if isinstance(node, ast.Tuple | ast.List):
        names = [_assignment_target(element) for element in node.elts]
        if all(name is not None for name in names):
            return ", ".join(name for name in names if name is not None)
    return None


def _assigned_calls(tree: ast.Module) -> dict[int, str]:
    targets: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.value, ast.Call):
                continue
            target = _assignment_target(node.targets[0])
            if target:
                targets[id(node.value)] = target
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            target = _assignment_target(node.target)
            if target:
                targets[id(node.value)] = target
    return targets


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return f"`{value}`"
    return repr(value)


def _transform_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _transform_sequence(call: ast.Call, imports: ImportTable) -> list[str]:
    if not call.args or not isinstance(call.args[0], ast.List | ast.Tuple):
        return []
    names: list[str] = []
    for element in call.args[0].elts:
        if not isinstance(element, ast.Call):
            continue
        path = _call_path(element, imports)
        if path and path.startswith(_TRANSFORM_PREFIXES):
            names.append(_transform_name(path))
    return names


def _model_claim(
    parsed: ParsedSource,
    call: ast.Call,
    path: str,
    target: str,
    symbols: SymbolTable,
) -> tuple[Operation, Claim]:
    name = path.rsplit(".", 1)[-1]
    params = _keyword_values(parsed, call, symbols)
    ev = _evidence(parsed, call, EvidenceLevel.E2, "pytorch.model_configuration")
    operation = Operation(
        "model_configuration",
        "pytorch",
        subject=name,
        attributes={"target": target, **params},
        evidence=ev,
    )
    details: list[str] = []
    if "weights" in params:
        details.append(f"weights={_format_value(params['weights'])}")
    if "num_classes" in params:
        details.append(f"num_classes={params['num_classes']}")
    suffix = f" with {', '.join(details)}" if details else ""
    claim = Claim(f"The model `{target}` is instantiated using {name}{suffix}.", ev)
    return operation, claim


def analyze_pytorch_experiment(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []
    assigned = _assigned_calls(parsed.tree)

    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
                target = _dotted_name(node.targets[0])
                if (
                    target
                    and target.startswith("model.")
                    and isinstance(node.value, ast.Call)
                ):
                    path = _call_path(node.value, imports)
                    if path and path.startswith(_TORCH_MODULE_PREFIX):
                        layer = path.rsplit(".", 1)[-1]
                        params = _keyword_values(parsed, node.value, symbols)
                        if layer == "Linear":
                            if len(node.value.args) >= 2:
                                out_features = _static_value(parsed, symbols, node.value.args[1])
                                if out_features is not None:
                                    params["out_features"] = out_features
                            elif "out_features" in params:
                                pass
                        ev = _evidence(
                            parsed,
                            node,
                            EvidenceLevel.E2,
                            "pytorch.model_head_replacement",
                        )
                        operations.append(
                            Operation(
                                "model_head_replacement",
                                "pytorch",
                                subject=target,
                                attributes={"layer": layer, **params},
                                evidence=ev,
                            )
                        )
                        suffix = ""
                        if "out_features" in params:
                            suffix = f" with {params['out_features']} output features"
                        claims.append(
                            Claim(
                                f"The `{target}` module is replaced by a {layer} layer{suffix}.",
                                ev,
                            )
                        )

            # Detect explicit parameter freezing without assuming why it is done.
            for target_node in node.targets:
                if (
                    isinstance(target_node, ast.Attribute)
                    and target_node.attr == "requires_grad"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is False
                ):
                    subject = _dotted_name(target_node.value) or "parameter"
                    ev = _evidence(parsed, node, EvidenceLevel.E1, "pytorch.parameter_freeze")
                    operations.append(
                        Operation("parameter_freeze", "pytorch", subject=subject, evidence=ev)
                    )
                    claims.append(
                        Claim(
                            f"Gradient computation is explicitly disabled for `{subject}`.",
                            ev,
                        )
                    )

        if not isinstance(node, ast.Call):
            continue

        path = _call_path(node, imports)
        if path is None:
            continue

        if path in _DATASETS:
            params = _keyword_values(parsed, node, symbols)
            dataset_name = path.rsplit(".", 1)[-1]
            target = assigned.get(id(node))
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.dataset_configuration")
            attributes = {"dataset": dataset_name, **params}
            if target:
                attributes["target"] = target
            operations.append(
                Operation("dataset_configuration", "pytorch", subject=target, attributes=attributes, evidence=ev)
            )
            details: list[str] = []
            if "root" in params:
                details.append(f"root={_format_value(params['root'])}")
            if "train" in params:
                details.append(f"train={params['train']}")
            suffix = f" with {', '.join(details)}" if details else ""
            label = f"`{target}`" if target else dataset_name
            claims.append(
                Claim(f"The dataset {label} is configured using {dataset_name}{suffix}.", ev)
            )
            continue

        if path == "torch.utils.data.random_split":
            target = assigned.get(id(node))
            lengths = None
            if len(node.args) >= 2:
                lengths = _static_value(parsed, symbols, node.args[1])
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.dataset_split")
            attrs: dict[str, Any] = {}
            if target:
                attrs["targets"] = target
            if lengths is not None:
                attrs["lengths"] = lengths
            operations.append(Operation("dataset_split", "pytorch", attributes=attrs, evidence=ev))
            suffix = f" using split sizes {lengths}" if lengths is not None else ""
            claims.append(Claim(f"The dataset is partitioned with `random_split`{suffix}.", ev))
            continue

        if path.startswith(_TRANSFORM_PREFIXES):
            name = _transform_name(path)
            params = _keyword_values(parsed, node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.preprocessing_transform")
            if name == "Compose":
                sequence = _transform_sequence(node, imports)
                operations.append(
                    Operation(
                        "preprocessing_pipeline",
                        "pytorch",
                        attributes={"transforms": sequence},
                        evidence=ev,
                    )
                )
                if sequence:
                    claims.append(
                        Claim(
                            "The preprocessing pipeline composes the operations "
                            + ", ".join(sequence)
                            + ".",
                            ev,
                        )
                    )
                else:
                    claims.append(Claim("A composed preprocessing pipeline is configured.", ev))
                continue

            attributes = {"transform": name, **params}
            if node.args:
                positional = _static_value(parsed, symbols, node.args[0])
                if positional is not None:
                    attributes["value"] = positional
            operations.append(
                Operation("preprocessing_transform", "pytorch", subject=name, attributes=attributes, evidence=ev)
            )
            if name in _STOCHASTIC_TRANSFORMS:
                claims.append(
                    Claim(f"{name} is included as a stochastic data-augmentation operation.", ev)
                )
            elif name == "Normalize":
                mean = params.get("mean")
                std = params.get("std")
                if mean is None and node.args:
                    mean = _static_value(parsed, symbols, node.args[0])
                if std is None and len(node.args) >= 2:
                    std = _static_value(parsed, symbols, node.args[1])
                details: list[str] = []
                if mean is not None:
                    details.append(f"mean={mean}")
                if std is not None:
                    details.append(f"std={std}")
                suffix = f" using {', '.join(details)}" if details else ""
                claims.append(Claim(f"Input tensors are normalized{suffix}.", ev))
            elif name == "Resize":
                size = attributes.get("value", params.get("size"))
                suffix = f" to {size}" if size is not None else ""
                claims.append(Claim(f"Inputs are resized{suffix} during preprocessing.", ev))
            elif name == "ToTensor":
                claims.append(Claim("Inputs are converted to tensor representation.", ev))
            continue

        target = assigned.get(id(node))
        if target and (
            path.startswith(_MODEL_PREFIXES)
            or (path.startswith(_TORCH_MODULE_PREFIX) and target == "model")
        ):
            operation, claim = _model_claim(parsed, node, path, target, symbols)
            operations.append(operation)
            claims.append(claim)
            continue

        if path.startswith(_METRIC_PREFIXES):
            metric_name = path.rsplit(".", 1)[-1]
            target = assigned.get(id(node))
            params = _keyword_values(parsed, node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.metric_configuration")
            operations.append(
                Operation(
                    "metric_configuration",
                    "pytorch",
                    subject=metric_name,
                    attributes={"target": target, **params},
                    evidence=ev,
                )
            )
            label = f" as `{target}`" if target else ""
            claims.append(Claim(f"The evaluation metric {metric_name} is configured{label}.", ev))
            continue

        if path == "torch.argmax" or path.endswith(".argmax"):
            dim = None
            if "dim" in _keyword_values(parsed, node, symbols):
                dim = _keyword_values(parsed, node, symbols)["dim"]
            elif len(node.args) >= 2 and path == "torch.argmax":
                dim = _static_value(parsed, symbols, node.args[1])
            elif node.args and path.endswith(".argmax"):
                dim = _static_value(parsed, symbols, node.args[0])
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.prediction_argmax")
            attrs = {"dim": dim} if dim is not None else {}
            operations.append(Operation("prediction_selection", "pytorch", attributes=attrs, evidence=ev))
            suffix = f" along dimension {dim}" if dim is not None else ""
            claims.append(
                Claim(f"Predicted indices are selected by taking an argmax{suffix}.", ev)
            )
            continue

        if path in {"torch.softmax", "torch.nn.functional.softmax"}:
            params = _keyword_values(parsed, node, symbols)
            dim = params.get("dim")
            if dim is None and len(node.args) >= 2:
                dim = _static_value(parsed, symbols, node.args[1])
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.softmax")
            attrs = {"dim": dim} if dim is not None else {}
            operations.append(Operation("softmax", "pytorch", attributes=attrs, evidence=ev))
            suffix = f" along dimension {dim}" if dim is not None else ""
            claims.append(Claim(f"Softmax normalization is applied{suffix}.", ev))
            continue

        if path.endswith(".load_state_dict") and isinstance(node.func, ast.Attribute):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.load_state_dict")
            load_subject = _dotted_name(node.func.value)
            operations.append(
                Operation("state_dict_load", "pytorch", subject=load_subject, evidence=ev)
            )
            claims.append(
                Claim(f"Serialized parameter state is loaded into `{load_subject or 'the model'}`.", ev)
            )

    return SemanticOutput(operations=operations, claims=claims)
