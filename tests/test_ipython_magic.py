import pytest

from astscribe.ipython.magic import (
    _explain_source_with_history,
    _parse_cell_style,
    _parse_scribe_line,
    explain_history_cell,
)


def test_explain_history_cell_uses_prior_notebook_context() -> None:
    history = (
        "",
        "import torch as t",
        "with t.no_grad():\n    outputs = model(inputs)",
    )

    rendered = explain_history_cell(history, 2)

    assert "Gradient tracking is disabled" in rendered
    assert "forward pass" in rendered.lower()


def test_ipython_only_input_is_a_conservative_context_barrier() -> None:
    history = (
        "",
        "import torch as t",
        "%run mutate_namespace.py",
        "with t.no_grad():\n    outputs = model(inputs)",
    )

    rendered = explain_history_cell(history, 3)

    assert "Gradient tracking is disabled" not in rendered


def test_cell_magic_source_uses_prior_history_context() -> None:
    history = ("", "import torch as t")

    rendered = _explain_source_with_history(
        history,
        "with t.no_grad():\n    outputs = model(inputs)",
        cell_index=2,
        style="concise",
    )

    assert "pytorch inference" in rendered.lower()


def test_line_magic_request_parsing_supports_cell_and_style() -> None:
    assert _parse_scribe_line("12 concise", default_cell=4) == (12, "concise")
    assert _parse_scribe_line("--style educational", default_cell=4) == (4, "educational")
    assert _parse_scribe_line("7 --style=scientific", default_cell=4) == (7, "scientific")


def test_cell_magic_style_parsing_keeps_existing_short_form() -> None:
    assert _parse_cell_style("") == "scientific"
    assert _parse_cell_style("concise") == "concise"
    assert _parse_cell_style("--style educational") == "educational"


def test_missing_history_cell_has_clear_error() -> None:
    with pytest.raises(ValueError, match=r"Cell In\[4\] is not available"):
        explain_history_cell(("", "x = 1"), 4)


def test_target_magic_cell_is_not_treated_as_python() -> None:
    with pytest.raises(ValueError, match=r"Cell In\[1\] is not valid Python"):
        explain_history_cell(("", "%time train()"), 1)
