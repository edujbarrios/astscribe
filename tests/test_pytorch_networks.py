from astscribe import NotebookAnalyzer, analyze


def test_common_nn_modules_and_static_hyperparameters() -> None:
    result = analyze(
        "import torch.nn as nn\n"
        "layer = nn.Conv2d(3, 64, 3)\n"
        "embedding = nn.Embedding(32000, 256)\n"
        "attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)\n"
        "activation = nn.GELU()\n"
    )
    conv = next(op for op in result.operations if op.subject == "Conv2d")
    assert conv.attributes == {"in_channels": 3, "out_channels": 64, "kernel_size": 3}
    assert any(op.subject == "Embedding" and op.attributes["embedding_dim"] == 256
               for op in result.operations)
    assert any(op.kind == "activation_configuration" and op.subject == "GELU"
               for op in result.operations)
    assert any(op.kind == "neural_layer_configuration" and op.subject == "MultiheadAttention"
               for op in result.operations)
    assert conv.evidence is not None and conv.evidence.line_start == 2


def test_tensor_creation_and_attention_function() -> None:
    result = analyze(
        "import torch\n"
        "from torch.nn import functional as F\n"
        "x = torch.randn(2, 3, 224, 224)\n"
        "flat = x.flatten(1)\n"
        "out = F.scaled_dot_product_attention(query, key, value)\n"
    )
    kinds = {op.kind for op in result.operations}
    assert {"tensor_creation", "tensor_operation", "attention_operation"} <= kinds
    assert any(c.rule == "pytorch.attention_operation" for c in result.claims)


def test_no_false_torch_inference_for_unrelated_objects() -> None:
    result = analyze(
        "from unrelated import Conv2d\n"
        "layer = Conv2d(3, 8, 3)\n"
        "thing.flatten()\n"
        "other.relu()\n"
    )
    assert not any(op.kind in {
        "neural_layer_configuration", "tensor_operation", "functional_neural_operation"
    } for op in result.operations)


def test_network_semantics_appear_in_notebook_pipeline() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from torch.nn import Linear\nlayer = Linear(8, 4)",
        "import torch\nx = torch.zeros(2, 8)",
    ])
    stages = {stage.key for stage in notebook.pipeline().stages}
    assert "model" in stages
    assert "preprocessing" in stages
