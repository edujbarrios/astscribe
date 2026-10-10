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


def test_constant_self_assignment_retains_previously_known_value() -> None:
    parsed = parse_source("seed = 7\nseed = seed\ncopied = seed")
    imports = build_import_table(parsed.tree)
    symbols = build_symbol_table(parsed.tree, imports)

    assert symbols.constants["seed"] == 7
    assert symbols.constants["copied"] == 7


def test_constructor_alias_preserves_context_and_origin_across_cells() -> None:
    first = parse_source(
        "from transformers import TrainingArguments\n"
        'args = TrainingArguments(output_dir="runs", num_train_epochs=2)'
    )
    imports = build_import_table(first.tree)
    original = build_symbol_table(first.tree, imports, source=first.source, cell=0)
    second = parse_source("alias = args\nargs = object()")
    aliased = build_symbol_table(second.tree, imports, base=original, source=second.source, cell=1)

    context = aliased.constructor_context("alias")
    assert context is not None
    constructor, arguments, origin = context
    assert constructor == "transformers.TrainingArguments"
    assert arguments == {"output_dir": "runs", "num_train_epochs": 2}
    assert origin is not None and origin.cell == 0
    assert aliased.resolve_constructor("args") == "object"


def test_constructor_keyword_uses_value_before_target_is_rebound() -> None:
    source = (
        "from transformers import TrainingArguments\n"
        "epochs = 3\n"
        'epochs = TrainingArguments(output_dir="runs", num_train_epochs=epochs)'
    )
    parsed = parse_source(source)
    symbols = build_symbol_table(parsed.tree, build_import_table(parsed.tree))
    context = symbols.constructor_context("epochs")
    assert context is not None
    assert context[1]["num_train_epochs"] == 3


def test_dynamic_attribute_call_is_not_a_named_constructor() -> None:
    source = "value = factory().AutoTokenizer.from_pretrained('model')"
    parsed = parse_source(source)
    symbols = build_symbol_table(parsed.tree, build_import_table(parsed.tree))
    assert symbols.resolve_constructor("value") is None
