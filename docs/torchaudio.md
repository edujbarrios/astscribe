# TorchAudio static analysis

ASTScribe recognizes common `torchaudio` source operations without importing
TorchAudio, downloading datasets or running audio models. It cannot listen to
recordings, determine acoustic content, validate sample rates, or establish ASR
accuracy from source code alone.

Supported groups include top-level audio load/save, datasets (including
LIBRISPEECH and SPEECHCOMMANDS), transforms (Resample, Spectrogram, MFCC,
MelSpectrogram, masks, RNN-T loss, beamforming), `functional` DSP calls,
model architectures (Wav2Vec2, HuBERT, Conformer, RNNT, Tacotron2, etc.),
Kaldi feature functions and **explicit** pretrained pipeline bundle methods.

```python
from astscribe import analyze

result = analyze("""
import torchaudio as ta
from torchaudio.transforms import MelSpectrogram
wav, sr = ta.load("speech.wav")
mel = MelSpectrogram(sample_rate=16000, n_mels=80)
features = mel(wav)
""")

for operation in result.operations:
    if operation.framework == "torchaudio":
        print(operation.kind, operation.attributes, operation.evidence)
```

ASTScribe classifies the loading call and transform construction, *not*
the unknown runtime tensor passed to `mel`.

## Version compatibility and explicit limitations

TorchAudio **2.9+** delegates audio loading and saving to TorchCodec;
`torchaudio.load` and `torchaudio.save` are compatibility APIs. Some
older TorchAudio symbols were removed in 2.9. ASTScribe classifies an
import-resolved, catalogued call but does **not** verify that an API exists
in the installed TorchAudio version or that TorchCodec is available. We
only guarantee coverage for the named, tested vocabulary; not every public,
experimental, custom or removed entry point.

Documentation: https://docs.pytorch.org/audio/stable/torchaudio.html
