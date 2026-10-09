# Vision and vision-language notebook analysis

ASTScribe **statically** explains source code; it does not load images,
download checkpoints, run a neural network, or claim to understand the image
contents. You do not need PyTorch, PIL, torchvision, or Transformers installed
to inspect code snippets that *mention* those libraries.

## Recognized patterns

- **Image and video inputs:** `PIL.Image.open`, `torchvision.io.read_image`,
  `torchvision.io.decode_image`, `torchvision.io.read_video`, and `cv2.imread`.
- **Vision pipelines:** Hugging Face `pipeline` tasks such as `image-to-text`,
  `image-text-to-text`, `visual-question-answering`, and
  `zero-shot-image-classification`.
- **VLM architectures:** source-visible `from_pretrained` configurations for
  LLaVA/LLaVA-NeXT, Qwen2-VL, PaliGemma, Idefics, BLIP/BLIP-2, InstructBLIP,
  Pix2Struct, Mllama, Gemma 3, CLIP, SigLIP, and relevant Transformers auto models.
  Detection describes *configured architecture*, not a verified model capability.
- **Multimodal inputs:** known Transformers processors called with explicit
  `images=` or `videos=` arguments, source-visible `apply_chat_template`
  calls, and `generate` calls on configured generative VLMs.
- **Neural networks:** common `torch.nn` convolutional layers, recurrent and
  Transformer modules, attention, normalization, embeddings, activations,
  tensor factories and functional operations.

## Example

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_cells([
    """from transformers import AutoProcessor, LlavaForConditionalGeneration
processor = AutoProcessor.from_pretrained('org/processor')
model = LlavaForConditionalGeneration.from_pretrained('org/model')""",
    """from PIL import Image
image = Image.open('photo.png')
inputs = processor(text='Describe this', images=image, return_tensors='pt')
answer = model.generate(**inputs, max_new_tokens=64)""",
])

for result in notebook.results:
    for operation in result.operations:
        if operation.framework == "multimodal":
            print(operation.kind, operation.evidence.line_start if operation.evidence else None)
print(notebook.render_pipeline())
```

All `operation.evidence` records point back to source lines and notebook
cell indices. See the `result.claims` and `result.to_dict()` APIs for
structured evidence and export.

### Important limits

- `model.generate(**inputs)` proves that generation was invoked, **not**
  that `inputs` contains image tensors or that the model ran successfully.
- A processor call with `images=` proves an image argument was supplied in
  source; it does not reveal pixel content, image resolution, patches, or output.
- Arbitrary user-defined model subclasses and custom multimodal pipelines
  cannot be automatically classified without identifiable source operations.
- Static information can become stale across unknown notebook code. Unsupported
  notebook cells act as conservative context boundaries.
