from __future__ import annotations

import shlex
from collections.abc import Sequence
from typing import Any, cast

from astscribe.notebook import NotebookAnalyzer

_STYLES = frozenset({"scientific", "educational", "concise"})


def _validate_style(style: str) -> None:
    if style not in _STYLES:
        choices = ", ".join(sorted(_STYLES))
        raise ValueError(f"Unsupported rendering style {style!r}. Choose one of: {choices}.")


def _history_context(history: Sequence[str], *, before: int) -> NotebookAnalyzer:
    analyzer = NotebookAnalyzer()
    stop = min(before, len(history))

    for cell_index in range(1, stop):
        source = history[cell_index]
        if not source.strip():
            continue
        try:
            analyzer.add_cell(source, cell_index=cell_index)
        except SyntaxError:
            # IPython-only syntax may have arbitrary runtime side effects. Treat it as a
            # conservative context barrier instead of letting stale bindings flow past it.
            analyzer = NotebookAnalyzer()

    return analyzer


def explain_history_cell(
    history: Sequence[str],
    cell: int,
    style: str = "scientific",
) -> str:
    """Explain one ``In[n]`` entry using earlier Python inputs as static context."""

    _validate_style(style)
    if cell < 1 or cell >= len(history):
        raise ValueError(f"Cell In[{cell}] is not available in the current IPython history.")

    source = history[cell]
    if not source.strip():
        raise ValueError(f"Cell In[{cell}] is empty.")

    analyzer = _history_context(history, before=cell)
    try:
        result = analyzer.add_cell(source, cell_index=cell)
    except SyntaxError as exc:
        raise ValueError(
            f"Cell In[{cell}] is not valid Python for ASTScribe static analysis: {exc.msg}."
        ) from exc
    return result.render(style)


def _explain_source_with_history(
    history: Sequence[str],
    source: str,
    *,
    cell_index: int,
    style: str,
) -> str:
    _validate_style(style)
    analyzer = _history_context(history, before=cell_index)
    try:
        result = analyzer.add_cell(source, cell_index=cell_index)
    except SyntaxError as exc:
        raise ValueError(
            f"Current cell is not valid Python for ASTScribe static analysis: {exc.msg}."
        ) from exc
    return result.render(style)


def _parse_scribe_line(line: str, *, default_cell: int) -> tuple[int, str]:
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        raise ValueError(
            "Usage: %scribe [CELL] [scientific|educational|concise] "
            "or %scribe [CELL] --style STYLE"
        ) from exc

    target: int | None = None
    style = "scientific"
    style_seen = False
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token == "--style":
            if style_seen or index + 1 >= len(tokens):
                raise ValueError(
                    "Usage: %scribe [CELL] [scientific|educational|concise] "
                    "or %scribe [CELL] --style STYLE"
                )
            style = tokens[index + 1]
            style_seen = True
            index += 2
            continue

        if token.startswith("--style="):
            if style_seen:
                raise ValueError("Rendering style was specified more than once.")
            style = token.partition("=")[2]
            style_seen = True
        elif token in _STYLES:
            if style_seen:
                raise ValueError("Rendering style was specified more than once.")
            style = token
            style_seen = True
        elif token.isdecimal():
            if target is not None:
                raise ValueError("Cell number was specified more than once.")
            target = int(token)
        else:
            raise ValueError(
                "Usage: %scribe [CELL] [scientific|educational|concise] "
                "or %scribe [CELL] --style STYLE"
            )
        index += 1

    _validate_style(style)
    target = default_cell if target is None else target
    if target < 1:
        raise ValueError("There is no previous input cell to explain.")
    return target, style


def _parse_cell_style(line: str) -> str:
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        raise ValueError(
            "Usage: %%scribe [scientific|educational|concise] or %%scribe --style STYLE"
        ) from exc

    if not tokens:
        return "scientific"
    if len(tokens) == 1 and tokens[0] in _STYLES:
        return tokens[0]
    if len(tokens) == 1 and tokens[0].startswith("--style="):
        style = tokens[0].partition("=")[2]
        _validate_style(style)
        return style
    if len(tokens) == 2 and tokens[0] == "--style":
        _validate_style(tokens[1])
        return tokens[1]
    raise ValueError("Usage: %%scribe [scientific|educational|concise] or %%scribe --style STYLE")


def _input_history(ipython: Any) -> Sequence[str]:
    history_manager = getattr(ipython, "history_manager", None)
    history = getattr(history_manager, "input_hist_raw", None)
    if history is None:
        raise RuntimeError("ASTScribe could not read the current IPython input history.")
    return cast(Sequence[str], history)


def load_ipython_extension(ipython: Any) -> None:
    try:
        from IPython.display import Markdown, display
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install ASTScribe with the 'ipython' extra to use %scribe.") from exc

    def scribe_line(line: str) -> None:
        history = _input_history(ipython)
        execution_count = int(getattr(ipython, "execution_count", len(history)))
        cell, style = _parse_scribe_line(line, default_cell=execution_count - 1)
        display(Markdown(explain_history_cell(history, cell, style=style)))

    def scribe_cell(line: str, cell: str) -> None:
        history = _input_history(ipython)
        execution_count = int(getattr(ipython, "execution_count", len(history)))
        style = _parse_cell_style(line)
        rendered = _explain_source_with_history(
            history,
            cell,
            cell_index=execution_count,
            style=style,
        )
        display(Markdown(rendered))

    ipython.register_magic_function(scribe_line, magic_kind="line", magic_name="scribe")
    ipython.register_magic_function(scribe_cell, magic_kind="cell", magic_name="scribe")
