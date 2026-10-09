from astscribe import NotebookAnalyzer, analyze


def test_vlm_processor_vision_inputs_and_generation_across_cells() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from transformers import AutoProcessor, LlavaForConditionalGeneration\n"
        "from PIL import Image\n"
        "processor = AutoProcessor.from_pretrained('sample/processor')\n"
        "model = LlavaForConditionalGeneration.from_pretrained('sample/model')\n",
        "image = Image.open('input.png')\n"
        "batch = processor(text='Describe it', images=image, return_tensors='pt')\n"
        "formatted = processor.apply_chat_template(messages, add_generation_prompt=True)\n"
        "tokens = model.generate(**batch, max_new_tokens=64)\n",
    ])
    first, second = notebook.results
    assert "multimodal" in first.frameworks
    model = next(op for op in first.operations if op.kind == "vision_language_model_configuration")
    assert model.subject == "LlavaForConditionalGeneration"
    assert model.attributes["pretrained_model_name_or_path"] == "sample/model"
    processing = next(op for op in second.operations if op.kind == "multimodal_input_processing")
    assert processing.attributes["has_images"] is True
    assert processing.attributes["has_text"] is True
    assert processing.evidence is not None and processing.evidence.cell == 1
    generation = next(op for op in second.operations if op.kind == "multimodal_generation")
    assert generation.attributes["unpacked_inputs"] is True
    assert generation.attributes["max_new_tokens"] == 64
    assert {op.kind for op in second.operations} >= {
        "image_input", "multimodal_chat_template", "multimodal_generation"
    }
    stages = {stage.key for stage in notebook.pipeline().stages}
    assert {"preprocessing", "model", "evaluation"} <= stages


def test_qwen_vlm_aliases_and_explicit_visual_generation() -> None:
    result = analyze(
        "import transformers as hf\n"
        "model = hf.Qwen2VLForConditionalGeneration.from_pretrained('example/checkpoint')\n"
        "model.generate(pixel_values=pixels, max_new_tokens=16)\n"
    )
    generation = next(op for op in result.operations if op.kind == "multimodal_generation")
    assert generation.attributes["visual_kwargs_explicit"] is True
    assert any(op.subject == "Qwen2VLForConditionalGeneration"
               for op in result.operations)


def test_explicit_pil_and_torchvision_image_loading() -> None:
    result = analyze(
        "import torchvision.io\n"
        "from PIL import Image as Picture\n"
        "a = torchvision.io.read_image('x.png')\n"
        "b = Picture.open('x.jpg')\n"
    )
    assert len([op for op in result.operations if op.kind == "image_input"]) == 2


def test_non_vlm_generation_and_unbound_processor_not_misclassified() -> None:
    result = analyze(
        "from transformers import AutoModelForCausalLM\n"
        "model = AutoModelForCausalLM.from_pretrained('gpt2')\n"
        "model.generate(input_ids=tokens)\n"
        "processor(images=raw)\n"
    )
    assert not any(op.framework == "multimodal" for op in result.operations)


def test_vision_pipeline_is_recorded_without_model_loading() -> None:
    result = analyze(
        "from transformers import pipeline\n"
        "reader = pipeline('image-text-to-text')\n"
    )
    assert any(op.kind == "vision_language_pipeline"
               and op.attributes["task"] == "image-text-to-text"
               for op in result.operations)
