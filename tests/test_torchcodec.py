from astscribe import NotebookAnalyzer, analyze


def test_codec_audio_video_and_image_inputs() -> None:
    result = analyze(
        "import torchcodec.decoders as dec\n"
        "audio = dec.AudioDecoder('speech.wav', sample_rate=16000)\n"
        "video = dec.VideoDecoder('sample.mp4')\n"
        "image = dec.decode_image(encoded_bytes)\n"
    )
    kinds = [op.kind for op in result.operations if op.framework == "torchcodec"]
    assert kinds.count("media_decoder_configuration") == 2
    assert "image_decoding" in kinds
    assert any(op.attributes.get("sample_rate") == 16000 for op in result.operations)
    assert all(op.evidence and op.evidence.line_start is not None
               for op in result.operations)


def test_sampling_encoder_and_cross_cell_media_calls() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from torchcodec.decoders import VideoDecoder\n"
        "from torchcodec.encoders import VideoEncoder\n"
        "from torchcodec.samplers import clips_at_random_indices\n"
        "decoder = VideoDecoder('movie.mp4')\n"
        "encoder = VideoEncoder(frames, frame_rate=25.0)",
        "clips = clips_at_random_indices(decoder, num_clips=2)\n"
        "frame = decoder.get_frames_at([0, 5])\n"
        "encoder.to_file('out.mp4')",
    ])
    second = notebook.results[1]
    kinds = {op.kind for op in second.operations}
    assert {"video_clip_sampling", "media_frame_access", "media_encoding_operation"} <= kinds
    assert any(op.attributes.get("num_clips") == 2 for op in second.operations)
    assert {"preprocessing", "checkpointing"} <= {stage.key for stage in notebook.pipeline().stages}


def test_media_api_provenance_prevents_similar_names_from_other_libraries() -> None:
    result = analyze(
        "from unrelated.decoders import AudioDecoder\n"
        "from unrelated.samplers import clips_at_random_indices\n"
        "decoder = AudioDecoder('fake.mp3')\n"
        "decoder.get_all_samples()\n"
        "clips_at_random_indices(decoder, num_clips=2)\n"
    )
    assert not any(op.framework == "torchcodec" for op in result.operations)


def test_codec_decoder_rebinding_invalidates_method_semantics() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from torchcodec.decoders import AudioDecoder\n"
        "decoder = AudioDecoder('sample.mp3')",
        "decoder = custom_decoder",
        "samples = decoder.get_all_samples()",
    ])
    assert not any(op.kind == "media_frame_access" for op in notebook.results[-1].operations)
