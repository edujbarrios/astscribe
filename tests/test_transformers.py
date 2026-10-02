from astscribe import NotebookAnalyzer, analyze


def test_transformers_pretrained_components_and_forward_call_are_detected() -> None:
    result = analyze(
        "from transformers import AutoModelForSequenceClassification, AutoTokenizer\n"
        "tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')\n"
        "model = AutoModelForSequenceClassification.from_pretrained(\n"
        "    'distilbert-base-uncased', num_labels=3\n"
        ")\n"
        "encoded = tokenizer(\n"
        "    text, padding=True, truncation=True, max_length=128, return_tensors='pt'\n"
        ")\n"
        "outputs = model(**encoded, labels=labels)\n"
    )

    assert "transformers" in result.frameworks
    kinds = {operation.kind for operation in result.operations}
    assert "tokenizer_configuration" in kinds
    assert "pretrained_model_configuration" in kinds
    assert "tokenization" in kinds
    assert "forward_pass" in kinds
    assert "supervision_labels" in kinds

    tokenizer = next(
        operation for operation in result.operations if operation.kind == "tokenization"
    )
    assert tokenizer.attributes["padding"] is True
    assert tokenizer.attributes["truncation"] is True
    assert tokenizer.attributes["max_length"] == 128
    assert tokenizer.attributes["return_tensors"] == "pt"

    model = next(
        operation
        for operation in result.operations
        if operation.kind == "pretrained_model_configuration"
    )
    assert model.subject == "AutoModelForSequenceClassification"
    assert model.attributes["pretrained_model_name_or_path"] == "distilbert-base-uncased"
    assert model.attributes["num_labels"] == 3


def test_transformers_generation_and_decoding_use_cross_cell_context() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell(
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "tokenizer = AutoTokenizer.from_pretrained('gpt2')\n"
        "model = AutoModelForCausalLM.from_pretrained('gpt2')\n"
    )
    result = notebook.add_cell(
        "inputs = tokenizer(prompt, return_tensors='pt')\n"
        "generated = model.generate(\n"
        "    **inputs, max_new_tokens=32, do_sample=True, temperature=0.7, top_p=0.9\n"
        ")\n"
        "text = tokenizer.batch_decode(generated, skip_special_tokens=True)\n"
    )

    generation = next(
        operation for operation in result.operations if operation.kind == "generation"
    )
    assert generation.attributes["max_new_tokens"] == 32
    assert generation.attributes["do_sample"] is True
    assert generation.attributes["temperature"] == 0.7
    assert generation.attributes["top_p"] == 0.9
    assert any(operation.kind == "output_decoding" for operation in result.operations)


def test_transformers_trainer_and_training_arguments_are_detected() -> None:
    result = analyze(
        "from transformers import Trainer, TrainingArguments, set_seed\n"
        "set_seed(17)\n"
        "args = TrainingArguments(\n"
        "    output_dir='runs',\n"
        "    num_train_epochs=4,\n"
        "    learning_rate=3e-5,\n"
        "    per_device_train_batch_size=8,\n"
        "    weight_decay=0.01,\n"
        ")\n"
        "trainer = Trainer(\n"
        "    model=model,\n"
        "    args=args,\n"
        "    train_dataset=train_dataset,\n"
        "    eval_dataset=validation_dataset,\n"
        ")\n"
        "trainer.train()\n"
        "trainer.evaluate()\n"
    )

    kinds = {operation.kind for operation in result.operations}
    assert "seed_configuration" in kinds
    assert "training_arguments_configuration" in kinds
    assert "trainer_configuration" in kinds
    assert "trainer_train" in kinds
    assert "trainer_evaluation" in kinds

    arguments = next(
        operation
        for operation in result.operations
        if operation.kind == "training_arguments_configuration"
    )
    assert arguments.attributes["num_train_epochs"] == 4
    assert arguments.attributes["learning_rate"] == 3e-5
    assert arguments.attributes["per_device_train_batch_size"] == 8
    assert arguments.attributes["weight_decay"] == 0.01


def test_transformers_pipeline_is_reconstructed_in_methods_and_pipeline() -> None:
    notebook = NotebookAnalyzer()
    notebook.add_cell(
        "from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed\n"
        "set_seed(42)\n"
        "tokenizer = AutoTokenizer.from_pretrained('gpt2')\n"
        "model = AutoModelForCausalLM.from_pretrained('gpt2')\n"
    )
    notebook.add_cell(
        "inputs = tokenizer(prompt, return_tensors='pt')\n"
        "generated = model.generate(**inputs, max_new_tokens=20)\n"
        "text = tokenizer.decode(generated[0], skip_special_tokens=True)\n"
    )

    report = notebook.render_methodology(include_evidence=True)
    assert "## Reproducibility" in report
    assert "## Preprocessing and augmentation" in report
    assert "## Model architecture" in report
    assert "## Evaluation and inference" in report
    assert "transformers.model_from_pretrained" in report
    assert "transformers.generate" in report

    stages = [stage.key for stage in notebook.pipeline().stages]
    assert "preprocessing" in stages
    assert "model" in stages
    assert "evaluation" in stages


def test_transformers_inference_pipeline_and_data_collator_are_detected() -> None:
    result = analyze(
        "from transformers import DataCollatorWithPadding, pipeline\n"
        "collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)\n"
        "classifier = pipeline('text-classification', model='model-id')\n"
    )

    collator = next(
        operation
        for operation in result.operations
        if operation.kind == "data_collator_configuration"
    )
    assert collator.subject == "DataCollatorWithPadding"
    assert collator.attributes["pad_to_multiple_of"] == 8

    inference_pipeline = next(
        operation
        for operation in result.operations
        if operation.kind == "inference_pipeline_configuration"
    )
    assert inference_pipeline.attributes["task"] == "text-classification"
