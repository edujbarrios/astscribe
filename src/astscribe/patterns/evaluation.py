from __future__ import annotations

from astscribe.sir import Operation


def is_evaluation_context(operations: list[Operation]) -> bool:
    return any(operation.kind == "evaluation_mode" for operation in operations)
