from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .bindings import collect_binding_effects


@dataclass
class ImportTable:
    aliases: dict[str, str] = field(default_factory=dict)

    def copy(self) -> ImportTable:
        return ImportTable(aliases=dict(self.aliases))

    def resolve_name(self, name: str) -> str:
        return self.aliases.get(name, name)

    def resolve_dotted(self, dotted: str) -> str:
        head, *tail = dotted.split(".")
        resolved = self.resolve_name(head)
        return ".".join([resolved, *tail]) if tail else resolved


def build_import_table(tree: ast.Module) -> ImportTable:
    table = ImportTable()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                table.aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                table.aliases[local] = f"{node.module}.{alias.name}"
    return table



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
            continue

        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                table.aliases.clear()
                continue

            for alias in node.names:
                local = alias.asname or alias.name
                if node.module is None:
                    table.aliases.pop(local, None)
                else:
                    table.aliases[local] = f"{node.module}.{alias.name}"
            continue

        effects = collect_binding_effects(node)
        if effects.clears_all:
            table.aliases.clear()
        for name in effects.names:
            table.aliases.pop(name, None)

    return table
