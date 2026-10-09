from astscribe import analyze
from astscribe.parser import build_import_table, parse_source
from astscribe.parser.imports import update_import_table


def test_dotted_import_without_alias_binds_only_root() -> None:
    tree = parse_source("import torchvision.transforms\nimport torch.nn").tree
    for imports in (build_import_table(tree), update_import_table(tree)):
        assert imports.resolve_dotted("torchvision.transforms.Resize") == (
            "torchvision.transforms.Resize"
        )
        assert imports.resolve_dotted("torch.nn.Linear") == "torch.nn.Linear"


def test_qualified_import_with_alias_binds_full_module() -> None:
    tree = parse_source("import torchvision.transforms as T\nimport torch.nn as nn").tree
    for imports in (build_import_table(tree), update_import_table(tree)):
        assert imports.resolve_dotted("T.Resize") == "torchvision.transforms.Resize"
        assert imports.resolve_dotted("nn.Linear") == "torch.nn.Linear"


def test_vision_transform_from_qualified_import_is_detected() -> None:
    result = analyze(
        "import torchvision.transforms\n"
        "resize = torchvision.transforms.Resize((224, 224))\n"
    )
    assert any(
        op.kind == "preprocessing_transform" and op.subject == "Resize"
        for op in result.operations
    )
