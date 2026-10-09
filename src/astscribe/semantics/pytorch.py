from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from astscribe.parser import ImportTable, ParsedSource, SymbolOrigin, SymbolTable
from astscribe.sir import Claim, Evidence, EvidenceLevel, Operation


@dataclass(frozen=True)
class SemanticOutput:
    operations: list[Operation]
    claims: list[Claim]


_OPTIMIZERS = {
    f"torch.optim.{name}"
    for name in (
        "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD",
        "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD", "SparseAdam",
    )
}
_LOSSES = {
    f"torch.nn.{name}"
    for name in (
        "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss",
        "L1Loss", "SmoothL1Loss", "HuberLoss", "KLDivLoss", "NLLLoss",
        "BCELoss", "CTCLoss", "CosineEmbeddingLoss", "TripletMarginLoss",
        "MarginRankingLoss", "PoissonNLLLoss", "MultiMarginLoss",
    )
}
_SCHEDULER_PREFIX = "torch.optim.lr_scheduler."
_AUTOCAST_PATHS = {
    "torch.autocast",
    "torch.amp.autocast",
    "torch.cuda.amp.autocast",
}
_GRAD_SCALERS = {"torch.amp.GradScaler", "torch.cuda.amp.GradScaler"}
_GRADIENT_CLIPPING = {
    "torch.nn.utils.clip_grad_norm_": "norm",
    "torch.nn.utils.clip_grad_value_": "value",
}


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _evidence(parsed: ParsedSource, node: ast.AST, level: EvidenceLevel, rule: str) -> Evidence:
    return Evidence(
        level=level,
        rule=rule,
        source=parsed.source_segment(node),
        line_start=getattr(node, "lineno", None),
        line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
        cell=parsed.cell,
    )


def _origin_evidence(origin: SymbolOrigin, rule: str) -> Evidence:
    return Evidence(
        level=EvidenceLevel.E2,
        rule=rule,
        source=origin.source,
        line_start=origin.line_start,
        line_end=origin.line_end,
        cell=origin.cell,
    )


def _literal(symbols: SymbolTable, node: ast.AST) -> Any | None:
    return symbols.resolve_constant(node)


def _call_path(call: ast.Call, imports: ImportTable) -> str | None:
    name = dotted_name(call.func)
    return imports.resolve_dotted(name) if name else None


def _keyword_values(call: ast.Call, symbols: SymbolTable) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = _literal(symbols, keyword.value)
        if value is not None:
            values[keyword.arg] = value
    return values


def _root_name(node: ast.AST) -> str | None:
    name = dotted_name(node)
    return name.split(".", 1)[0] if name else None


def _constructor_for_subject(node: ast.AST, symbols: SymbolTable) -> str | None:
    root = _root_name(node)
    return symbols.resolve_constructor(root) if root else None


def _describe_optimizer(algorithm: str, attributes: dict[str, Any]) -> str:
    details: list[str] = []
    if "lr" in attributes:
        details.append(f"a learning rate of {attributes['lr']}")
    if "weight_decay" in attributes:
        details.append(f"a weight-decay coefficient of {attributes['weight_decay']}")
    if "momentum" in attributes:
        details.append(f"momentum of {attributes['momentum']}")
    return f"{algorithm} with {', '.join(details)}" if details else algorithm


def _describe_scheduler(name: str, attributes: dict[str, Any]) -> str:
    details = [f"{key}={value}" for key, value in attributes.items()]
    return f"{name} with {', '.join(details)}" if details else name


def _optimizer_binding_claim(
    subject_node: ast.AST,
    symbols: SymbolTable,
) -> tuple[Claim, dict[str, Any]] | None:
    root = _root_name(subject_node)
    if root is None:
        return None
    context = symbols.constructor_context(root)
    if context is None:
        return None
    constructor, attributes, origin = context
    if constructor not in _OPTIMIZERS or origin is None:
        return None

    algorithm = constructor.rsplit(".", 1)[-1]
    description = _describe_optimizer(algorithm, attributes)
    evidence = _origin_evidence(origin, "context.optimizer_binding")
    claim = Claim(
        f"The optimizer referenced in this cell was previously configured as {description}.",
        evidence,
    )
    operation_attributes = {"optimizer": algorithm, **attributes}
    return claim, operation_attributes


def _known_model_call(node: ast.Call, symbols: SymbolTable) -> bool:
    if not isinstance(node.func, ast.Name):
        return False
    if node.func.id == "model":
        return True
    constructor = symbols.resolve_constructor(node.func.id)
    return bool(
        constructor
        and constructor.startswith("torch.nn.")
        and constructor not in _LOSSES
    )


def _known_loss_call(node: ast.Call, symbols: SymbolTable) -> bool:
    if not isinstance(node.func, ast.Name):
        return False
    if node.func.id == "criterion":
        return True
    constructor = symbols.resolve_constructor(node.func.id)
    return constructor in _LOSSES


def _is_optimizer_subject(node: ast.AST, symbols: SymbolTable) -> bool:
    root = _root_name(node)
    constructor = _constructor_for_subject(node, symbols)
    return root == "optimizer" or constructor in _OPTIMIZERS


def _is_scheduler_subject(node: ast.AST, symbols: SymbolTable) -> bool:
    root = _root_name(node)
    constructor = _constructor_for_subject(node, symbols)
    return root == "scheduler" or bool(constructor and constructor.startswith(_SCHEDULER_PREFIX))


def _is_scaler_subject(node: ast.AST, symbols: SymbolTable) -> bool:
    root = _root_name(node)
    constructor = _constructor_for_subject(node, symbols)
    return root == "scaler" or constructor in _GRAD_SCALERS


def _device_label(node: ast.AST, symbols: SymbolTable) -> str | None:
    value = _literal(symbols, node)
    if value is not None:
        return str(value)
    if isinstance(node, ast.Name):
        return node.id
    return dotted_name(node)


def _has_pytorch_context(imports: ImportTable, symbols: SymbolTable) -> bool:
    if any(value == "torch" or value.startswith("torch.") for value in imports.aliases.values()):
        return True
    return any(value.startswith("torch.") for value in symbols.constructors.values())


def _epoch_count(node: ast.For, symbols: SymbolTable) -> int | None:
    if not isinstance(node.target, ast.Name) or "epoch" not in node.target.id.lower():
        return None
    if not isinstance(node.iter, ast.Call):
        return None
    if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
        return None
    if len(node.iter.args) != 1:
        return None
    value = _literal(symbols, node.iter.args[0])
    if isinstance(value, int) and value >= 0:
        return value
    return None


def analyze_pytorch(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []
    contextual_claims: set[str] = set()
    epoch_candidates: list[tuple[ast.For, int]] = []

    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.For):
            epochs = _epoch_count(node, symbols)
            if epochs is not None:
                epoch_candidates.append((node, epochs))

        if isinstance(node, ast.With):
            for item in node.items:
                if not isinstance(item.context_expr, ast.Call):
                    continue
                path = _call_path(item.context_expr, imports)
                if path == "torch.no_grad":
                    ev = _evidence(parsed, item.context_expr, EvidenceLevel.E3, "pytorch.no_grad")
                    operations.append(
                        Operation("gradient_tracking_disabled", "pytorch", evidence=ev)
                    )
                    claims.append(
                        Claim(
                            "Gradient tracking is disabled for the enclosed operations, so no "
                            "autograd graph is constructed for computations executed within this "
                            "context.",
                            ev,
                        )
                    )
                elif path == "torch.inference_mode":
                    ev = _evidence(
                        parsed,
                        item.context_expr,
                        EvidenceLevel.E3,
                        "pytorch.inference_mode",
                    )
                    operations.append(Operation("inference_mode", "pytorch", evidence=ev))
                    claims.append(
                        Claim(
                            "PyTorch inference mode is enabled for the enclosed operations, "
                            "disabling gradient tracking and inference-irrelevant autograd "
                            "bookkeeping.",
                            ev,
                        )
                    )
                elif path in _AUTOCAST_PATHS:
                    device_type = None
                    if item.context_expr.args:
                        device_type = _literal(symbols, item.context_expr.args[0])
                    params = _keyword_values(item.context_expr, symbols)
                    device_type = params.get("device_type", device_type)
                    attributes = {"device_type": device_type} if device_type is not None else {}
                    ev = _evidence(parsed, item.context_expr, EvidenceLevel.E3, "pytorch.autocast")
                    operations.append(
                        Operation("automatic_mixed_precision", "pytorch", attributes=attributes, evidence=ev)
                    )
                    suffix = f" for the {device_type} device type" if device_type else ""
                    claims.append(
                        Claim(
                            "Automatic mixed-precision autocasting is enabled"
                            f"{suffix} for the enclosed operations.",
                            ev,
                        )
                    )

        if not isinstance(node, ast.Call):
            continue

        path = _call_path(node, imports)
        if path is None:
            continue

        if path.endswith(".train") and isinstance(node.func, ast.Attribute):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.model_train")
            operations.append(
                Operation(
                    "training_mode",
                    "pytorch",
                    subject=dotted_name(node.func.value),
                    evidence=ev,
                )
            )
            claims.append(Claim("The model is explicitly configured in training mode.", ev))
            continue

        if path.endswith(".eval") and isinstance(node.func, ast.Attribute):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.model_eval")
            operations.append(
                Operation(
                    "evaluation_mode",
                    "pytorch",
                    subject=dotted_name(node.func.value),
                    evidence=ev,
                )
            )
            claims.append(Claim("The model is explicitly configured in evaluation mode.", ev))
            continue

        if path == "backward" or path.endswith(".backward"):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.backward")
            operations.append(Operation("backward_pass", "pytorch", evidence=ev))
            claims.append(
                Claim(
                    "Backward propagation is invoked from the objective, causing gradients to be "
                    "computed for differentiable parameters that contribute to it.",
                    ev,
                )
            )
            continue

        if path.endswith(".zero_grad") and isinstance(node.func, ast.Attribute):
            if not _is_optimizer_subject(node.func.value, symbols):
                continue
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.optimizer_zero_grad")
            context = _optimizer_binding_claim(node.func.value, symbols)
            attributes: dict[str, Any] = {}
            if context is not None:
                context_claim, attributes = context
                root = _root_name(node.func.value)
                if root and root not in contextual_claims:
                    claims.append(context_claim)
                    contextual_claims.add(root)
            operations.append(
                Operation(
                    "gradient_reset",
                    "pytorch",
                    subject=dotted_name(node.func.value),
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(
                Claim(
                    "Previously accumulated optimizer gradients are reset before the next update.",
                    ev,
                )
            )
            continue

        if path.endswith(".step") and isinstance(node.func, ast.Attribute):
            subject = node.func.value
            if _is_scaler_subject(subject, symbols):
                ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.grad_scaler_step")
                operations.append(Operation("scaled_optimizer_step", "pytorch", evidence=ev))
                claims.append(
                    Claim(
                        "The gradient scaler delegates an optimizer step using the currently scaled gradients.",
                        ev,
                    )
                )
                continue
            if _is_scheduler_subject(subject, symbols):
                ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.scheduler_step")
                operations.append(Operation("scheduler_step", "pytorch", evidence=ev))
                claims.append(
                    Claim("The configured learning-rate scheduler advances by one step.", ev)
                )
                continue
            if not _is_optimizer_subject(subject, symbols):
                continue

            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.optimizer_step")
            context = _optimizer_binding_claim(subject, symbols)
            attributes = {}
            if context is not None:
                context_claim, attributes = context
                root = _root_name(subject)
                if root and root not in contextual_claims:
                    claims.append(context_claim)
                    contextual_claims.add(root)
            operations.append(
                Operation(
                    "parameter_update",
                    "pytorch",
                    subject=dotted_name(subject),
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(Claim("The configured optimizer performs a parameter-update step.", ev))
            continue

        if path.endswith(".update") and isinstance(node.func, ast.Attribute):
            if _is_scaler_subject(node.func.value, symbols):
                ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.grad_scaler_update")
                operations.append(Operation("grad_scaler_update", "pytorch", evidence=ev))
                claims.append(
                    Claim("The gradient-scaling factor is updated for subsequent iterations.", ev)
                )
                continue

        if path.endswith(".scale") and isinstance(node.func, ast.Attribute):
            if _is_scaler_subject(node.func.value, symbols):
                ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.grad_scaler_scale")
                operations.append(Operation("loss_scaling", "pytorch", evidence=ev))
                claims.append(
                    Claim("The objective is scaled before backward propagation for mixed-precision training.", ev)
                )
                continue

        if path == "torch.manual_seed":
            value = _literal(symbols, node.args[0]) if node.args else None
            ev = _evidence(parsed, node, EvidenceLevel.E1, "pytorch.manual_seed")
            operations.append(
                Operation(
                    "seed_configuration",
                    "pytorch",
                    attributes={"seed": value},
                    evidence=ev,
                )
            )
            text = "The PyTorch pseudorandom number generator is explicitly seeded"
            text += f" with the value {value}." if value is not None else "."
            claims.append(Claim(text, ev))
            continue

        if path in _OPTIMIZERS:
            params = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.optimizer_configuration")
            algorithm = path.rsplit(".", 1)[-1]
            operations.append(
                Operation(
                    "optimizer_configuration",
                    "pytorch",
                    subject=algorithm,
                    attributes=params,
                    evidence=ev,
                )
            )
            description = _describe_optimizer(algorithm, params)
            claims.append(Claim(f"The optimizer is configured as {description}.", ev))
            continue

        if path.startswith(_SCHEDULER_PREFIX):
            params = _keyword_values(node, symbols)
            scheduler_name = path.rsplit(".", 1)[-1]
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.scheduler_configuration")
            operations.append(
                Operation(
                    "scheduler_configuration",
                    "pytorch",
                    subject=scheduler_name,
                    attributes=params,
                    evidence=ev,
                )
            )
            description = _describe_scheduler(scheduler_name, params)
            claims.append(
                Claim(f"The learning-rate schedule is configured using {description}.", ev)
            )
            continue

        if path in _GRAD_SCALERS:
            params = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.grad_scaler")
            operations.append(
                Operation("grad_scaler_configuration", "pytorch", attributes=params, evidence=ev)
            )
            claims.append(
                Claim("A gradient scaler is configured for mixed-precision optimization.", ev)
            )
            continue

        if path in _LOSSES:
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.loss_configuration")
            loss_name = path.rsplit(".", 1)[-1]
            operations.append(
                Operation("loss_configuration", "pytorch", subject=loss_name, evidence=ev)
            )
            claims.append(Claim(f"The training objective is configured using {loss_name}.", ev))
            continue

        if path == "torch.device":
            device_type = _literal(symbols, node.args[0]) if node.args else None
            params = _keyword_values(node, symbols)
            device_type = params.get("type", device_type)
            attributes = {"device": device_type} if device_type is not None else {}
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.device_configuration")
            operations.append(
                Operation("device_configuration", "pytorch", attributes=attributes, evidence=ev)
            )
            suffix = f" for `{device_type}`" if device_type is not None else ""
            claims.append(Claim(f"A PyTorch execution device is configured{suffix}.", ev))
            continue

        if path in _GRADIENT_CLIPPING:
            mode = _GRADIENT_CLIPPING[path]
            params = _keyword_values(node, symbols)
            threshold = None
            if len(node.args) >= 2:
                threshold = _literal(symbols, node.args[1])
            if mode == "norm":
                threshold = params.get("max_norm", threshold)
                label = "maximum norm"
            else:
                threshold = params.get("clip_value", threshold)
                label = "clipping value"
            attributes = {label.replace(" ", "_"): threshold} if threshold is not None else {}
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.gradient_clipping")
            operations.append(
                Operation("gradient_clipping", "pytorch", attributes=attributes, evidence=ev)
            )
            suffix = f" using a {label} of {threshold}" if threshold is not None else ""
            claims.append(Claim(f"Gradient clipping is applied{suffix}.", ev))
            continue

        if path in {"torch.save", "torch.load"}:
            kind = "checkpoint_save" if path.endswith("save") else "checkpoint_load"
            ev = _evidence(parsed, node, EvidenceLevel.E3, f"pytorch.{kind}")
            operations.append(Operation(kind, "pytorch", evidence=ev))
            text = (
                "Model-related state is serialized to storage."
                if kind == "checkpoint_save"
                else "Serialized PyTorch state is loaded from storage."
            )
            claims.append(Claim(text, ev))
            continue

        if path.endswith("DataLoader"):
            params = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.dataloader")
            operations.append(
                Operation(
                    "dataloader_configuration",
                    "pytorch",
                    attributes=params,
                    evidence=ev,
                )
            )
            details: list[str] = []
            if "batch_size" in params:
                details.append(f"batch size {params['batch_size']}")
            if "shuffle" in params:
                details.append(f"shuffle={params['shuffle']}")
            if "num_workers" in params:
                details.append(f"num_workers={params['num_workers']}")
            suffix = f" ({', '.join(details)})" if details else ""
            claims.append(Claim(f"A PyTorch DataLoader is configured{suffix}.", ev))
            continue

        if path.endswith(".to") and isinstance(node.func, ast.Attribute) and node.args:
            device = _device_label(node.args[0], symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.device_transfer")
            attributes = {"device": device} if device else {}
            operations.append(
                Operation(
                    "device_transfer",
                    "pytorch",
                    subject=dotted_name(node.func.value),
                    attributes=attributes,
                    evidence=ev,
                )
            )
            suffix = f" `{device}`" if device else " the specified device"
            claims.append(Claim(f"The referenced object is moved to{suffix}.", ev))
            continue

        if path.endswith(".cuda") and isinstance(node.func, ast.Attribute):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.device_transfer")
            operations.append(Operation("device_transfer", "pytorch", attributes={"device": "cuda"}, evidence=ev))
            claims.append(Claim("The referenced object is moved to a CUDA device.", ev))
            continue

        if path.endswith(".cpu") and isinstance(node.func, ast.Attribute):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.device_transfer")
            operations.append(Operation("device_transfer", "pytorch", attributes={"device": "cpu"}, evidence=ev))
            claims.append(Claim("The referenced object is moved to CPU memory.", ev))
            continue

        if _known_model_call(node, symbols):
            model_subject = node.func.id if isinstance(node.func, ast.Name) else None
            ev = _evidence(parsed, node, EvidenceLevel.E1, "pytorch.forward_pass")
            operations.append(
                Operation("forward_pass", "pytorch", subject=model_subject, evidence=ev)
            )
            claims.append(
                Claim("A forward pass is performed by invoking the model on the supplied inputs.", ev)
            )
        elif _known_loss_call(node, symbols):
            loss_subject = node.func.id if isinstance(node.func, ast.Name) else None
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.loss_computation")
            operations.append(
                Operation("loss_computation", "pytorch", subject=loss_subject, evidence=ev)
            )
            claims.append(
                Claim(
                    "The configured objective function is evaluated using the supplied predictions "
                    "and targets.",
                    ev,
                )
            )

    if _has_pytorch_context(imports, symbols):
        for node, epochs in epoch_candidates:
            ev = _evidence(parsed, node.iter, EvidenceLevel.E2, "pytorch.epoch_loop")
            operations.append(
                Operation("epoch_loop", "pytorch", attributes={"epochs": epochs}, evidence=ev)
            )
            claims.append(
                Claim(f"The training procedure is configured to iterate for {epochs} epochs.", ev)
            )

    return SemanticOutput(operations=operations, claims=claims)
