from astscribe import NotebookAnalyzer, analyze


def test_functional_classification_metric_without_constructor_claim() -> None:
    result = analyze(
        "from torchmetrics.functional.classification import multiclass_accuracy\n"
        "score = multiclass_accuracy(preds, target, num_classes=5, average='macro')"
    )
    assert any(op.kind == "metric_computation" and op.attributes["num_classes"] == 5
               for op in result.operations)
    assert not any(op.kind == "metric_configuration" for op in result.operations)
    assert not any(op.kind == "trainer_evaluation" for op in result.operations)


def test_stateful_metric_across_cells_and_stale_context() -> None:
    notebook = NotebookAnalyzer.from_cells([
        "from torchmetrics.classification import MulticlassF1Score\n"
        "metric = MulticlassF1Score(num_classes=6)",
        "metric.update(preds, target)\nvalue = metric.compute()\nmetric.reset()",
        "metric = unknown",
        "metric.compute()",
    ])
    first, second, _, fourth = notebook.results
    assert any(op.kind == "metric_configuration" for op in first.operations)
    assert {op.kind for op in second.operations if op.framework == "torchmetrics"} == {
        "metric_update", "metric_computation", "metric_reset",
    }
    assert not any(op.kind == "metric_computation" for op in fourth.operations)
    assert "metrics" in {stage.key for stage in notebook.pipeline().stages}


def test_unknown_torchmetrics_symbols_do_not_create_metric_configurations() -> None:
    result = analyze(
        "from torchmetrics import EntirelyMadeUpMetric\n"
        "from torchmetrics.functional import custom_operation\n"
        "fake = EntirelyMadeUpMetric()\n"
        "computed = custom_operation(scores, labels)"
    )
    assert not any(op.kind.startswith("metric_") for op in result.operations)


def test_unrelated_compute_method_is_not_recognized() -> None:
    result = analyze("from otherlib import Metric\nm = Metric()\nm.compute()")
    assert not any(op.framework == "torchmetrics" for op in result.operations)
