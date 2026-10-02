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


def analyze_pytorch(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
) -> SemanticOutput:
    operations: list[Operation] = []
    claims: list[Claim] = []

    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if not isinstance(item.context_expr, ast.Call):
                    continue
                path = _call_path(item.context_expr, imports)
                if path == "torch.no_grad":
                    ev = _evidence(parsed, item.context_expr, EvidenceLevel.E3, "pytorch.no_grad")
                    operations.append(Operation("gradient_tracking_disabled", "pytorch", evidence=ev))
                    claims.append(Claim(
                        "Gradient tracking is disabled for the enclosed operations, so no autograd graph is constructed for computations executed within this context.",
                        ev,
                    ))
                elif path == "torch.inference_mode":
                    ev = _evidence(parsed, item.context_expr, EvidenceLevel.E3, "pytorch.inference_mode")
                    operations.append(Operation("inference_mode", "pytorch", evidence=ev))
                    claims.append(Claim(
                        "PyTorch inference mode is enabled for the enclosed operations, disabling gradient tracking and inference-irrelevant autograd bookkeeping.",
                        ev,
                    ))

        if not isinstance(node, ast.Call):
            continue

        path = _call_path(node, imports)
        if path is None:
            continue

        if path.endswith(".train") and isinstance(node.func, ast.Attribute):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.model_train")
            operations.append(Operation("training_mode", "pytorch", subject=dotted_name(node.func.value), evidence=ev))
            claims.append(Claim("The model is explicitly configured in training mode.", ev))
            continue

        if path.endswith(".eval") and isinstance(node.func, ast.Attribute):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.model_eval")
            operations.append(Operation("evaluation_mode", "pytorch", subject=dotted_name(node.func.value), evidence=ev))
            claims.append(Claim("The model is explicitly configured in evaluation mode.", ev))
            continue

        if path.endswith(".backward"):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.backward")
            operations.append(Operation("backward_pass", "pytorch", evidence=ev))
            claims.append(Claim(
                "Backward propagation is invoked from the objective, causing gradients to be computed for differentiable parameters that contribute to it.",
                ev,
            ))
            continue

        if path.endswith(".zero_grad"):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.optimizer_zero_grad")
            operations.append(Operation("gradient_reset", "pytorch", evidence=ev))
            claims.append(Claim("Previously accumulated optimizer gradients are reset before the next update.", ev))
            continue

        if path.endswith(".step"):
            ev = _evidence(parsed, node, EvidenceLevel.E3, "pytorch.optimizer_step")
            operations.append(Operation("parameter_update", "pytorch", evidence=ev))
            claims.append(Claim("The configured optimizer performs a parameter-update step.", ev))
            continue

        if path == "torch.manual_seed":
            value = _literal(symbols, node.args[0]) if node.args else None
            ev = _evidence(parsed, node, EvidenceLevel.E1, "pytorch.manual_seed")
            operations.append(Operation("seed_configuration", "pytorch", attributes={"seed": value}, evidence=ev))
            text = "The PyTorch pseudorandom number generator is explicitly seeded"
            text += f" with the value {value}." if value is not None else "."
            claims.append(Claim(text, ev))
            continue

        if path in {"torch.optim.Adam", "torch.optim.AdamW", "torch.optim.SGD"}:
            params = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.optimizer_configuration")
            algorithm = path.rsplit(".", 1)[-1]
            operations.append(Operation(
                "optimizer_configuration",
                "pytorch",
                subject=algorithm,
                attributes=params,
                evidence=ev,
            ))
            details: list[str] = []
            if "lr" in params:
                details.append(f"a learning rate of {params['lr']}")
            if "weight_decay" in params:
                details.append(f"a weight-decay coefficient of {params['weight_decay']}")
            if "momentum" in params:
                details.append(f"momentum of {params['momentum']}")
            suffix = f" with {', '.join(details)}" if details else ""
            claims.append(Claim(f"The optimizer is configured as {algorithm}{suffix}.", ev))
            continue

        if path in {
            "torch.nn.CrossEntropyLoss",
            "torch.nn.MSELoss",
            "torch.nn.BCEWithLogitsLoss",
        }:
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.loss_configuration")
            loss_name = path.rsplit(".", 1)[-1]
            operations.append(Operation("loss_configuration", "pytorch", subject=loss_name, evidence=ev))
            claims.append(Claim(f"The training objective is configured using {loss_name}.", ev))
            continue

        if path in {"torch.save", "torch.load"}:
            kind = "checkpoint_save" if path.endswith("save") else "checkpoint_load"
            ev = _evidence(parsed, node, EvidenceLevel.E3, f"pytorch.{kind}")
            operations.append(Operation(kind, "pytorch", evidence=ev))
            claims.append(Claim(
                "Model-related state is serialized to storage." if kind == "checkpoint_save" else "Serialized PyTorch state is loaded from storage.",
                ev,
            ))
            continue

        if path.endswith("DataLoader"):
            params = _keyword_values(node, symbols)
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.dataloader")
            operations.append(Operation("dataloader_configuration", "pytorch", attributes=params, evidence=ev))
            details: list[str] = []
            if "batch_size" in params:
                details.append(f"batch size {params['batch_size']}")
            if "shuffle" in params:
                details.append(f"shuffle={params['shuffle']}")
            suffix = f" ({', '.join(details)})" if details else ""
            claims.append(Claim(f"A PyTorch DataLoader is configured{suffix}.", ev))
            continue

        if isinstance(node.func, ast.Name):
            constructor = symbols.constructors.get(node.func.id)
            if constructor and constructor.startswith("torch.optim."):
                continue

        if isinstance(node.func, ast.Name) and node.func.id == "model":
            ev = _evidence(parsed, node, EvidenceLevel.E1, "pytorch.forward_pass")
            operations.append(Operation("forward_pass", "pytorch", subject="model", evidence=ev))
            claims.append(Claim("A forward pass is performed by invoking the model on the supplied inputs.", ev))
        elif isinstance(node.func, ast.Name) and node.func.id == "criterion":
            ev = _evidence(parsed, node, EvidenceLevel.E2, "pytorch.loss_computation")
            operations.append(Operation("loss_computation", "pytorch", subject="criterion", evidence=ev))
            claims.append(Claim("The configured objective function is evaluated using the supplied predictions and targets.", ev))

    return SemanticOutput(operations=operations, claims=claims)
