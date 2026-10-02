from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ImportTable:
    aliases: dict[str, str] = field(default_factory=dict)

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
                table.aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                table.aliases[local] = f"{node.module}.{alias.name}"
    return table
