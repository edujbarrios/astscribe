from astscribe.parser import build_import_table, build_symbol_table, parse_source


def test_alias_resolution() -> None:
    parsed = parse_source("import torch as t\nfrom torch.optim import AdamW")
    imports = build_import_table(parsed.tree)
    assert imports.resolve_dotted("t.no_grad") == "torch.no_grad"
    assert imports.resolve_dotted("AdamW") == "torch.optim.AdamW"


def test_static_constant_resolution() -> None:
    parsed = parse_source("learning_rate = 2e-5")
    imports = build_import_table(parsed.tree)
    symbols = build_symbol_table(parsed.tree, imports)
    assert symbols.constants["learning_rate"] == 2e-5



def test_symbol_table_clears_constant_on_function_rebinding() -> None:
    parsed = parse_source("seed = 7\ndef seed():\n    return 9")
    imports = build_import_table(parsed.tree)
    symbols = build_symbol_table(parsed.tree, imports)

    assert "seed" not in symbols.constants


def test_symbol_table_clears_constructor_on_delete() -> None:
    parsed = parse_source("from torch.optim import AdamW\nopt = AdamW(params)\ndel opt")
    imports = build_import_table(parsed.tree)
    symbols = build_symbol_table(parsed.tree, imports)

    assert symbols.resolve_constructor("opt") is None
