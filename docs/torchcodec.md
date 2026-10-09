# TorchCodec media support

TorchCodec is used for recent PyTorch ecosystem image, video and audio I/O.
ASTScribe recognizes **source-level** constructor calls to
`torchcodec.decoders.AudioDecoder`, `VideoDecoder`, `WavDecoder`,
image decoders such as `decode_image`, `torchcodec.samplers` clip
sampling functions, and `torchcodec.encoders` audio/video encoding
constructors. Known `decoder.get_frames_at(...)` and
`encoder.to_file(...)` methods are recognized when the variable has a
visible supported constructor.

The analysis is deterministic and **does not** open files, validate codecs,
download models, decode media, infer the actual number of frames, or predict
whether the operation will succeed.

```python
from astscribe import NotebookAnalyzer

notebook = NotebookAnalyzer.from_cells([
    """from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_indices
decoder = VideoDecoder('video.mp4')""",
    """clips = clips_at_regular_indices(decoder, num_clips=4)""",
])
print(notebook.render_pipeline())
```

Python snippets are analyzed as text; TorchCodec does not need to be installed
to inspect their semantics.

Official API references:
- https://meta-pytorch.org/torchcodec/stable/api_ref_decoders.html
- https://meta-pytorch.org/torchcodec/stable/api_ref_samplers.html
- https://meta-pytorch.org/torchcodec/stable/api_ref_encoders.html
