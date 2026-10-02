from astscribe import NotebookAnalyzer, analyze


def test_lora_configuration_recovers_explicit_adapter_hyperparameters() -> None:
    result = analyze(
        "from peft import LoraConfig, TaskType\n"
        "config = LoraConfig(\n"
        "    r=16,\n"
        "    lora_alpha=32,\n"
        "    lora_dropout=0.05,\n"
        "    bias='none',\n"
        "    target_modules=['q_proj', 'v_proj'],\n"
        "    task_type=TaskType.CAUSAL_LM,\n"
        ")\n"
    )

    assert "peft" in result.frameworks
    operation = next(
        item for item in result.operations if item.kind == "adapter_configuration"
    )
    assert operation.subject == "LoRA"
    assert operation.attributes["r"] == 16
    assert operation.attributes["lora_alpha"] == 32
    assert operation.attributes["lora_dropout"] == 0.05
    assert operation.attributes["bias"] == "none"
    assert operation.attributes["target_modules"] == ["q_proj", "v_proj"]
    assert operation.attributes["task_type"] == "peft.TaskType.CAUSAL_LM"


def test_get_peft_model_uses_cross_cell_config_context() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell(
        "from peft import LoraConfig, get_peft_model\n"
        "lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1)\n"
    )
    result = notebook.add_cell("peft_model = get_peft_model(model, lora_config)\n")

    operation = next(
        item for item in result.operations if item.kind == "adapter_application"
    )
    assert operation.attributes["adapter_config"] == "LoraConfig"
    assert operation.attributes["r"] == 8
    assert operation.attributes["lora_alpha"] == 16
    assert operation.attributes["lora_dropout"] == 0.1


def test_peft_model_methods_use_forward_only_constructor_context() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell(
        "from peft import LoraConfig, get_peft_model\n"
        "config = LoraConfig(r=4, lora_alpha=8)\n"
        "peft_model = get_peft_model(model, config)\n"
    )
    result = notebook.add_cell(
        "peft_model.save_pretrained('adapter')\n"
        "merged_model = peft_model.merge_and_unload()\n"
    )

    kinds = {operation.kind for operation in result.operations}
    assert "adapter_checkpoint_save" in kinds
    assert "adapter_merge" in kinds


def test_peft_checkpoint_loading_and_adapter_activation_are_detected() -> None:
    notebook = NotebookAnalyzer()
    result = notebook.add_cell(
        "from peft import PeftModel\n"
        "model = PeftModel.from_pretrained(base_model, 'org/adapter', is_trainable=True)\n"
    )

    load = next(
        operation
        for operation in result.operations
        if operation.kind == "adapter_checkpoint_load"
    )
    assert load.attributes["adapter_id"] == "org/adapter"
    assert load.attributes["is_trainable"] is True

    result = notebook.add_cell("model.set_adapter('domain-adapter')\n")
    activation = next(
        operation for operation in result.operations if operation.kind == "adapter_activation"
    )
    assert activation.attributes["adapter_name"] == "domain-adapter"


def test_kbit_preparation_is_reported_without_claiming_qlora() -> None:
    result = analyze(
        "from peft import prepare_model_for_kbit_training\n"
        "model = prepare_model_for_kbit_training(\n"
        "    model, use_gradient_checkpointing=True\n"
        ")\n"
    )

    operation = next(
        item for item in result.operations if item.kind == "kbit_training_preparation"
    )
    assert operation.attributes["use_gradient_checkpointing"] is True
    rendered = result.render("scientific")
    assert "k-bit training" in rendered
    assert "QLoRA" not in rendered


def test_peft_methods_and_pipeline_have_dedicated_adaptation_stage() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell(
        "from transformers import AutoModelForCausalLM\n"
        "model = AutoModelForCausalLM.from_pretrained('gpt2')\n"
    )
    notebook.add_cell(
        "from peft import LoraConfig, get_peft_model\n"
        "config = LoraConfig(r=8, lora_alpha=16)\n"
        "model = get_peft_model(model, config)\n"
    )

    report = notebook.render_methodology(include_evidence=True)
    assert "## Model architecture" in report
    assert "## Parameter-efficient fine-tuning" in report
    assert "peft.adapter_configuration" in report
    assert "peft.get_peft_model" in report

    stages = [stage.key for stage in notebook.pipeline().stages]
    assert stages.index("model") < stages.index("adaptation")
    assert notebook.pipeline().stages[stages.index("adaptation")].title == (
        "Parameter-efficient fine-tuning"
    )
