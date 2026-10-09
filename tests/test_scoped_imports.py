from astscribe import analyze
from astscribe.parser import build_import_table, parse_source


def test_function_local_torch_import_is_not_global() -> None:
    result = analyze(
        "def local():\n"
        "    import torch as t\n"
        "t.zeros(2)\n"
    )
    assert not any(op.kind == "tensor_creation" for op in result.operations)


def test_aliased_import_rebinding_is_not_torch() -> None:
    result = analyze(
        "import torch as t\n"
        "t = custom_object\n"
        "t.randn(2)\n"
    )
    assert not any(op.kind == "tensor_creation" for op in result.operations)


def test_module_scope_bare_import_is_preserved() -> None:
    parsed = parse_source("import torchaudio.transforms\nimport torchvision.ops as ops\n")
    imports = build_import_table(parsed.tree)
    assert imports.resolve_dotted("torchaudio.transforms.Resample") == (
        "torchaudio.transforms.Resample"
    )
    assert imports.resolve_dotted("ops.nms") == "torchvision.ops.nms"


def test_import_inside_unsupported_conditional_is_not_assumed_global() -> None:
    result = analyze("if runtime_condition:\n    import torch as t\nt.ones(3)\n")
    assert not any(op.kind == "tensor_creation" for op in result.operations)


def test_from_import_and_shadowing_are_respected() -> None:
    result = analyze(
        "from torch import randn as sample\n"
        "sample = other_factory\n"
        "out = sample(2)\n"
    )
    assert not any(op.kind == "tensor_creation" for op in result.operations)
