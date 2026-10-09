# Static PyTorch ecosystem coverage (ASTScribe 0.11.1)

ASTScribe **does not** execute notebooks or import heavy ML frameworks. It
recognizes calls by resolving **explicit Python imports** and records
`Operation` objects, `Claim`s, source lines and notebook cell indices.

| Family | Examples of recognized source-level APIs |
| --- | --- |
| `torch` tensors and neural layers | `zeros`, `randn`, `topk`, `cat`, `nn.Conv2d`, `nn.Embedding`, `nn.Transformer`, `nn.functional.cross_entropy` |
| `torch` training and modern internals | `torch.compile`, `torch.export.export`, `torch.autograd.grad`, `torch.func.jacrev`, `torch.distributed.all_reduce`, `torch.fft.rfft`, `torch.linalg.svd`, `torch.utils.checkpoint.checkpoint` |
| `torchvision` | `datasets.Food101`, `models.resnet50`, `models.detection.fasterrcnn_resnet50_fpn`, `ops.nms`, `transforms.v2.functional.resize`, `io.decode_jpeg`, `tv_tensors.BoundingBoxes` |
| `torchaudio` | `load`, `save`, `datasets.LIBRISPEECH`, `transforms.MelSpectrogram`, `functional.resample`, `models.Conformer`, `pipelines.WAV2VEC2_ASR_BASE_960H.get_model` |

Additional common APIs and architecture families in these namespaces are
recognized, but **the sets are curated and deliberately not exhaustive**.
An unrecognized call is not proof of a missing operation at runtime.

## Inspect a source-only multimodal notebook

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_cells([
    """import torch
import torchvision
import torchaudio""",
    """vision_model = torchvision.models.resnet18(weights=None)
vision_data = torchvision.datasets.Food101('/data')""",
    """waveform, sr = torchaudio.load('speech.wav')
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
compiled = torch.compile(vision_model)""",
])
for cell in notebook.results:
    for operation in cell.operations:
        print(operation.framework, operation.kind, operation.evidence.cell if operation.evidence else None)
```

This only analyzes the **strings**: it does not load Food101, speech.wav,
model weights or a compiler. `result.to_dict()` serializes the supporting
evidence.

## Precision over guesses

- Static source does not establish whether an operation executed successfully,
  actual tensor shapes, images, audio, scores, trained parameters or runtime
  capabilities.
- Framework provenance is import-based. Imports inside unknown control-flow
  branches and shadowed imports do not authorize unconditional claims.
- Distinct versions expose different APIs; the catalog does not guarantee
  availability in your installed version.
- `torchaudio.load` and `save` delegate to TorchCodec starting with
  TorchAudio 2.9, and some older APIs have been removed. See
  [the TorchAudio guide](torchaudio.md).
- Unknown code and unparseable notebook magics form conservative boundaries.

This release focuses on common, documented families of PyTorch and its
vision/audio ecosystem rather than making an untestable "infallibility"
promise.
