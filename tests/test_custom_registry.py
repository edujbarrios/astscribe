from types import SimpleNamespace

import pytest

from astscribe import NotebookAnalyzer, SemanticRegistry, analyze, explain
from astscribe.sir import Operation


def custom_analyzer(parsed, imports, symbols):
    if "custom_run()" in parsed.source:
        return SimpleNamespace(operations=[Operation("custom_run", "my_framework")], claims=[])
    return SimpleNamespace(operations=[], claims=[])


def test_user_registered_framework_is_detected() -> None:
    registry = SemanticRegistry()
    registry.register("my_framework", custom_analyzer)
    result = analyze("custom_run()", registry=registry)
    assert "my_framework" in result.frameworks
    assert any(op.kind == "custom_run" for op in result.operations)
    assert "my_framework" not in analyze("custom_run()").frameworks


def test_custom_registry_is_used_by_notebooks_and_explain() -> None:
    registry = SemanticRegistry()
    registry.register("my_framework", custom_analyzer)
    notebook = NotebookAnalyzer.from_cells(["custom_run()"], registry=registry)
    assert any(op.kind == "custom_run" for op in notebook.results[0].operations)
    assert explain("custom_run()", registry=registry, return_result=True).frameworks[-1] == "my_framework"


def test_registration_validates_framework_and_analyzer() -> None:
    registry = SemanticRegistry()
    with pytest.raises(ValueError, match="framework"):
        registry.register("", custom_analyzer)
    with pytest.raises(TypeError, match="callable"):
        registry.register("x", None)  # type: ignore[arg-type]
