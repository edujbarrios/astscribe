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


_OPTIMIZERS = {"torch.optim.Adam", "torch.optim.AdamW", "torch.optim.SGD"}
_LOSSES = {
    "torch.nn.CrossEntropyLoss",
    "torch.nn.MSELoss",
    "torch.nn.BCEWithLogitsLoss",
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


def _describe_optimizer(algorithm: str, attributes: dict[str, Any]) -> str:
    details: list[str] = []
    if "lr" in attributes:
        details.append(f"a learning rate of {attributes['lr']}")
    if "weight_decay" in attributes:
        details.append(f"a weight-decay coefficient of {attributes['weight_decay']}")
    if "momentum" in attributes:
        details.append(f"momentum of {attributes['momentum']}")
    return f"{algorithm} with {', '.join(details)}" if details else algorithm


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
    return bool(constructor and constructor.startswith("torch.nn."))


def _known_loss_call(node: ast.Call, symbols: SymbolTable) -> bool:
    if not isinstance(node.func, ast.Name):
        return False
    if node.func.id == "criterion":
        return True
    constructor = symbols.resolve_constructor(node.func.id)
    return constructor in _LOSSES


def analyze_pytorch(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []
    contextual_claims: set[str] = set()

    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if not isinstance(item.context_expr, ast.Call):
                    continue
                path = _call_path(item.context_expr, imports)
                if path == "torch.no_grad":
                    ev = _evidence(
                        parsed,
                        item.context_expr,
                        EvidenceLevel.E3,
                        "pytorch.no_grad",
                    )
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

        if path.endswith(".backward"):
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
                Claim("Previously accumulated optimizer gradients are reset before the next update.", ev)
            )
            continue

        if path.endswith(".step") and isinstance(node.func, ast.Attribute):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.optimizer_step")
            context = _optimizer_binding_claim(node.func.value, symbols)
            attributes = {}
            if context is not None:
                context_claim, attributes = context
                root = _root_name(node.func.value)
                if root and root not in contextual_claims:
                    claims.append(context_claim)
                    contextual_claims.add(root)
            operations.append(
                Operation(
                    "parameter_update",
                    "pytorch",
                    subject=dotted_name(node.func.value),
                    attributes=attributes,
                    evidence=ev,
                )
            )
            claims.append(Claim("The configured optimizer performs a parameter-update step.", ev))
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

        if path in _LOSSES:
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.loss_configuration")
            loss_name = path.rsplit(".", 1)[-1]
            operations.append(
                Operation("loss_configuration", "pytorch", subject=loss_name, evidence=ev)
            )
            claims.append(Claim(f"The training objective is configured using {loss_name}.", ev))
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
            suffix = f" ({', '.join(details)})" if details else ""
            claims.append(Claim(f"A PyTorch DataLoader is configured{suffix}.", ev))
            continue

        if _known_model_call(node, symbols):
            subject = node.func.id if isinstance(node.func, ast.Name) else None
            ev = _evidence(parsed, node, EvidenceLevel.E1, "pytorch.forward_pass")
            operations.append(Operation("forward_pass", "pytorch", subject=subject, evidence=ev))
            claims.append(
                Claim("A forward pass is performed by invoking the model on the supplied inputs.", ev)
            )
        elif _known_loss_call(node, symbols):
            subject = node.func.id if isinstance(node.func, ast.Name) else None
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.loss_computation")
            operations.append(
                Operation("loss_computation", "pytorch", subject=subject, evidence=ev)
            )
            claims.append(
                Claim(
                    "The configured objective function is evaluated using the supplied predictions "
                    "and targets.",
                    ev,
                )
            )

    return SemanticOutput(operations=operations, claims=claims)
