from __future__ import annotations

from typing import Any

from astscribe import explain


def load_ipython_extension(ipython: Any) -> None:
    try:
        from IPython.core.magic import register_cell_magic
        from IPython.display import Markdown, display
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install ASTScribe with the 'ipython' extra to use %%scribe.") from exc

    @register_cell_magic
    def scribe(line: str, cell: str) -> None:
        style = line.strip() or "scientific"
        display(Markdown(str(explain(cell, style=style))))
