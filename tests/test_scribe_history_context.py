from astscribe.ipython.magic import explain_history_cell


def test_own_line_magic_does_not_erase_import_context() -> None:
    history = ("", "import torch as t", "%scribe 1 concise", "with t.no_grad():\n    x = compute()")
    result = explain_history_cell(history, 3)
    assert "Gradient tracking is disabled" in result


def test_own_cell_magic_does_not_erase_import_context() -> None:
    history = ("", "import torch as t", "%%scribe concise\nt.randn(2)", "with t.no_grad():\n    x = compute()")
    result = explain_history_cell(history, 3)
    assert "Gradient tracking is disabled" in result


def test_unrelated_magic_still_invalidates_context() -> None:
    history = ("", "import torch as t", "%run mutate_namespace.py", "with t.no_grad():\n    x = compute()")
    result = explain_history_cell(history, 3)
    assert "Gradient tracking is disabled" not in result
