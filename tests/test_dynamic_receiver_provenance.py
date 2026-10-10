"""Dynamic call results must not inherit provenance from imported aliases."""

from astscribe import analyze


def test_dynamic_dataset_loader_does_not_claim_huggingface_dataset() -> None:
    result = analyze(
        "from datasets import load_dataset\n"
        "result = factory().load_dataset('imdb')\n"
    )
    assert not any(claim.rule.startswith("datasets.") for claim in result.claims)


def test_dynamic_model_loader_does_not_claim_transformers_checkpoint() -> None:
    result = analyze(
        "from transformers import AutoModelForSequenceClassification\n"
        "model = factory().AutoModelForSequenceClassification.from_pretrained('name')\n"
    )
    assert not any(claim.rule.startswith("transformers.") for claim in result.claims)


def test_direct_imported_model_loader_remains_recognized() -> None:
    result = analyze(
        "from transformers import AutoModelForSequenceClassification\n"
        "model = AutoModelForSequenceClassification.from_pretrained('name')\n"
    )
    assert any(
        claim.rule == "transformers.model_from_pretrained" for claim in result.claims
    )


def test_direct_dataset_loader_remains_recognized() -> None:
    result = analyze(
        "from datasets import load_dataset\n"
        "dataset = load_dataset('imdb')\n"
    )
    assert any(claim.rule.startswith("datasets.") for claim in result.claims)
