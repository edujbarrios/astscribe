from astscribe import NotebookAnalyzer, analyze


def test_modern_graph_autograd_and_fft_apis() -> None:
    result = analyze(
        "import torch as t\n"
        "compiled = t.compile(model, fullgraph=True)\n"
        "grads = t.autograd.grad(loss, params)\n"
        "jac = t.func.jacrev(fn)(features)\n"
        "spectral = t.fft.rfft(x)\n"
        "values = t.linalg.svd(matrix)\n"
        "t.nn.init.xavier_uniform_(layer.weight)\n"
    )
    kinds = {op.kind for op in result.operations}
    assert {"graph_compilation", "autograd_operation", "frequency_operation",
            "linear_algebra_operation", "parameter_initialization"} <= kinds
    operation = next(op for op in result.operations if op.kind == "graph_compilation")
    assert operation.attributes["fullgraph"] is True
    assert operation.evidence is not None and operation.evidence.line_start == 2


def test_distributed_loader_and_checkpoint_calls() -> None:
    result = analyze(
        "from torch.distributed import all_reduce as reduce, init_process_group\n"
        "from torch.utils.data import WeightedRandomSampler\n"
        "import torch.utils.checkpoint as cp\n"
        "init_process_group(backend='gloo')\n"
        "reduce(tensor)\n"
        "sampler = WeightedRandomSampler(weights, num_samples=10)\n"
        "y = cp.checkpoint(block, x)\n"
    )
    kinds = {op.kind for op in result.operations}
    assert {"distributed_communication", "data_pipeline_configuration",
            "activation_checkpointing"} <= kinds
    assert any(op.attributes.get("backend") == "gloo" for op in result.operations)
    assert any(op.subject == "torch.utils.data.WeightedRandomSampler" for op in result.operations)


def test_more_functional_and_nn_tensor_vocabulary() -> None:
    result = analyze(
        "from torch.nn import ConvTranspose2d\n"
        "from torch.nn.functional import cross_entropy as loss_fn\n"
        "from torch import topk\n"
        "conv = ConvTranspose2d(8, 3, 3)\n"
        "loss = loss_fn(scores, target)\n"
        "indices = topk(scores, 2)\n"
    )
    kinds = {op.kind for op in result.operations}
    assert "neural_layer_configuration" in kinds
    assert "functional_neural_operation" in kinds
    assert "tensor_operation" in kinds


def test_other_library_does_not_gain_torch_semantics_from_similar_method_names() -> None:
    result = analyze(
        "from own_lib import init_process_group, checkpoint, compile\n"
        "init_process_group()\ncheckpoint(fn, x)\ncompile(fn)\n"
    )
    assert not any(op.kind in {"distributed_communication", "graph_compilation",
                               "activation_checkpointing"} for op in result.operations)


def test_ecosystem_pipeline_across_cells() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "import torch\ncompiled = torch.compile(model)",
        "from torch.utils.data import TensorDataset\ndataset = TensorDataset(x, y)",
    ])
    stages = {stage.key for stage in notebook.pipeline().stages}
    assert "data_loading" in stages
    assert "model" in stages
