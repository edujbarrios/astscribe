from astscribe import analyze
from astscribe.notebook import NotebookAnalyzer


def test_load_dataset_recovers_source_split_and_streaming() -> None:
    result = analyze(
        "from datasets import load_dataset\n"
        "dataset = load_dataset('allenai/c4', split='train', streaming=True)\n"
    )

    operation = next(item for item in result.operations if item.kind == "dataset_configuration")
    assert operation.framework == "datasets"
    assert operation.subject == "allenai/c4"
    assert operation.attributes["split"] == "train"
    assert operation.attributes["streaming"] is True
    assert "datasets" in result.frameworks


def test_dataset_lineage_supports_map_then_train_test_split_across_cells() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("from datasets import load_dataset\ndataset = load_dataset('imdb', split='train')")
    mapped = notebook.add_cell("tokenized = dataset.map(tokenize, batched=True, num_proc=2)")
    split = notebook.add_cell("splits = tokenized.train_test_split(test_size=0.2, seed=42)")

    map_operation = next(item for item in mapped.operations if item.kind == "dataset_mapping")
    assert map_operation.attributes["batched"] is True
    assert map_operation.attributes["num_proc"] == 2

    split_operation = next(item for item in split.operations if item.kind == "dataset_split")
    assert split_operation.attributes["test_size"] == 0.2
    assert split_operation.attributes["seed"] == 42


def test_dataset_lineage_survives_fluent_self_reassignment() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("from datasets import load_dataset\ndataset = load_dataset('imdb', split='train')")
    mapped = notebook.add_cell("dataset = dataset.map(tokenize, batched=True)")
    shuffled = notebook.add_cell("dataset = dataset.shuffle(seed=7)")

    assert any(item.kind == "dataset_mapping" for item in mapped.operations)
    shuffle = next(item for item in shuffled.operations if item.kind == "dataset_shuffle")
    assert shuffle.attributes["seed"] == 7


def test_bitsandbytes_config_and_quantized_model_load_are_linked_across_cells() -> None:
    notebook = NotebookAnalyzer()
    config = notebook.add_cell(
        "from transformers import BitsAndBytesConfig\n"
        "bnb = BitsAndBytesConfig(\n"
        "    load_in_4bit=True,\n"
        "    bnb_4bit_quant_type='nf4',\n"
        "    bnb_4bit_use_double_quant=True,\n"
        ")\n"
    )
    model = notebook.add_cell(
        "from transformers import AutoModelForCausalLM\n"
        "model = AutoModelForCausalLM.from_pretrained('example/model', quantization_config=bnb)\n"
    )

    config_operation = next(
        item for item in config.operations if item.kind == "quantization_configuration"
    )
    assert config_operation.attributes["bits"] == 4
    assert config_operation.attributes["bnb_4bit_quant_type"] == "nf4"
    assert config_operation.attributes["bnb_4bit_use_double_quant"] is True

    load_operation = next(item for item in model.operations if item.kind == "quantized_model_load")
    assert load_operation.attributes["bits"] == 4
    assert load_operation.attributes["quantization_config"] == "BitsAndBytesConfig"


def test_non_model_named_transformers_class_is_treated_as_model() -> None:
    notebook = NotebookAnalyzer()
    loaded = notebook.add_cell(
        "from transformers import LlavaForConditionalGeneration\n"
        "model = LlavaForConditionalGeneration.from_pretrained('example/vlm')\n"
    )
    generated = notebook.add_cell("tokens = model.generate(**inputs, max_new_tokens=32)")

    model_operation = next(
        item for item in loaded.operations if item.kind == "pretrained_model_configuration"
    )
    assert model_operation.subject == "LlavaForConditionalGeneration"
    assert any(item.kind == "generation" for item in generated.operations)


def test_qlora_requires_4bit_lora_and_training_evidence() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell(
        "from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer\n"
        "bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')\n"
        "model = AutoModelForCausalLM.from_pretrained('example/model', quantization_config=bnb)\n"
    )
    notebook.add_cell(
        "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
        "model = prepare_model_for_kbit_training(model)\n"
        "lora = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'])\n"
        "model = get_peft_model(model, lora)\n"
    )
    notebook.add_cell("trainer = Trainer(model=model)\ntrainer.train()")

    findings = notebook.techniques()
    assert [finding.key for finding in findings] == ["qlora"]
    assert "4-bit" in findings[0].summary
    rendered = notebook.render_techniques()
    assert "## QLoRA" in rendered
    assert "transformers.quantized_model_load" in rendered
    assert "peft.get_peft_model" in rendered
    assert "transformers.trainer_train" in rendered


def test_4bit_preparation_without_lora_training_is_not_called_qlora() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell(
        "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n"
        "bnb = BitsAndBytesConfig(load_in_4bit=True)\n"
        "model = AutoModelForCausalLM.from_pretrained('example/model', quantization_config=bnb)\n"
    )
    notebook.add_cell(
        "from peft import prepare_model_for_kbit_training\n"
        "model = prepare_model_for_kbit_training(model)\n"
    )

    assert notebook.techniques() == ()
    assert notebook.render_techniques().startswith("No composite training technique")


def test_pipeline_and_methods_include_new_v05_stages() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell("from datasets import load_dataset\ndataset = load_dataset('imdb', split='train')")
    notebook.add_cell("tokenized = dataset.map(tokenize, batched=True)")
    notebook.add_cell(
        "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n"
        "bnb = BitsAndBytesConfig(load_in_4bit=True)\n"
        "model = AutoModelForCausalLM.from_pretrained('example/model', quantization_config=bnb)\n"
    )
    notebook.add_cell(
        "from peft import LoraConfig, get_peft_model\n"
        "lora = LoraConfig(r=8)\n"
        "model = get_peft_model(model, lora)\n"
    )

    keys = [stage.key for stage in notebook.pipeline().stages]
    assert keys[:5] == ["dataset", "dataset_preparation", "model", "quantization", "adaptation"]

    methods = notebook.render_methodology()
    assert "## Dataset" in methods
    assert "## Dataset preparation" in methods
    assert "## Quantization" in methods
    assert "## Parameter-efficient fine-tuning" in methods
