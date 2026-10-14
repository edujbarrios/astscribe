from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .bindings import collect_binding_effects

_FRAMEWORK_ROOTS = frozenset({
    "torch", "torchvision", "torchaudio", "torchcodec", "torchmetrics",
    "transformers", "datasets", "peft",
})


@dataclass
class ImportTable:
    aliases: dict[str, str] = field(default_factory=dict)
    shadowed: set[str] = field(default_factory=set)

    def copy(self) -> ImportTable:
        return ImportTable(aliases=dict(self.aliases), shadowed=set(self.shadowed))

    def resolve_name(self, name: str) -> str:
        return "" if name in self.shadowed else self.aliases.get(name, name)

    def resolve_dotted(self, dotted: str) -> str:
        head, *tail = dotted.split(".")
        resolved = self.resolve_name(head)
        return ".".join([resolved, *tail]) if tail and resolved else resolved


def build_import_table(tree: ast.Module) -> ImportTable:
    """Resolve only module-scope imports, dropping aliases shadowed in source order.

    Imports inside functions, classes and conditional branches do not establish
    unconditional module bindings. Unknown control flow is treated conservatively.
    """
    return update_import_table(tree)


def update_import_table(
    tree: ast.Module,
    *,
    base: ImportTable | None = None,
) -> ImportTable:
    """Update persistent module-scope imports conservatively in source order."""

    table = base.copy() if base is not None else ImportTable()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                table.aliases[local] = alias.name if alias.asname else local
                table.shadowed.discard(local)
            continue

        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                table.aliases.clear()
                continue

            for alias in node.names:
                local = alias.asname or alias.name
                if node.module is None:
                    table.aliases.pop(local, None)
                    table.shadowed.add(local)
                else:
                    table.aliases[local] = f"{node.module}.{alias.name}"
                    table.shadowed.discard(local)
            continue

        effects = collect_binding_effects(node)
        if effects.clears_all:
            table.shadowed.update(table.aliases)
            table.aliases.clear()
        for name in effects.names:
            if name in table.aliases or name in _FRAMEWORK_ROOTS:
                table.shadowed.add(name)
            table.aliases.pop(name, None)

    return table
