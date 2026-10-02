from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from astscribe.sir import AnalysisResult, Operation


@dataclass(frozen=True)
class PipelineStage:
    key: str
    title: str
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class ExperimentPipeline:
    stages: tuple[PipelineStage, ...]

    def render(self) -> str:
        if not self.stages:
            return "No evidence-backed experiment pipeline could be reconstructed."
        return "\n    ↓\n".join(stage.title for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("dataset", "Dataset"),
    ("preprocessing", "Preprocessing and augmentation"),
    ("data_loading", "Data loading"),
    ("model", "Model architecture"),
    ("objective", "Objective"),
    ("optimization", "Optimization"),
    ("precision", "Numerical precision"),
    ("training", "Training procedure"),
    ("evaluation", "Evaluation and inference"),
    ("metrics", "Metrics"),
    ("checkpointing", "Checkpointing"),
)

_OPERATION_STAGE = {
    "dataset_configuration": "dataset",
    "dataset_split": "dataset",
    "preprocessing_pipeline": "preprocessing",
    "preprocessing_transform": "preprocessing",
    "tokenizer_configuration": "preprocessing",
    "processor_configuration": "preprocessing",
    "tokenization": "preprocessing",
    "input_processing": "preprocessing",
    "dataloader_configuration": "data_loading",
    "data_collator_configuration": "data_loading",
    "model_configuration": "model",
    "model_head_replacement": "model",
    "parameter_freeze": "model",
    "model_config_load": "model",
    "pretrained_model_configuration": "model",
    "loss_configuration": "objective",
    "optimizer_configuration": "optimization",
    "scheduler_configuration": "optimization",
    "scheduler_step": "optimization",
    "gradient_clipping": "optimization",
    "automatic_mixed_precision": "precision",
    "grad_scaler_configuration": "precision",
    "loss_scaling": "precision",
    "scaled_optimizer_step": "precision",
    "grad_scaler_update": "precision",
    "training_mode": "training",
    "gradient_reset": "training",
    "forward_pass": "training",
    "loss_computation": "training",
    "backward_pass": "training",
    "parameter_update": "training",
    "epoch_loop": "training",
    "training_arguments_configuration": "training",
    "trainer_configuration": "training",
    "trainer_train": "training",
    "supervision_labels": "training",
    "evaluation_mode": "evaluation",
    "gradient_tracking_disabled": "evaluation",
    "inference_mode": "evaluation",
    "trainer_evaluation": "evaluation",
    "generation": "evaluation",
    "inference_pipeline_configuration": "evaluation",
    "output_decoding": "evaluation",
    "prediction_selection": "metrics",
    "softmax": "metrics",
    "metric_configuration": "metrics",
    "checkpoint_save": "checkpointing",
    "checkpoint_load": "checkpointing",
    "state_dict_load": "checkpointing",
}


def build_experiment_pipeline(results: tuple[AnalysisResult, ...]) -> ExperimentPipeline:
    buckets: dict[str, list[Operation]] = {key: [] for key, _ in _STAGE_ORDER}

    for result in results:
        for operation in result.operations:
            stage = _operation_stage(operation, result)
            if stage is not None:
                buckets[stage].append(operation)

    stages = tuple(
        PipelineStage(key=key, title=title, operations=tuple(buckets[key]))
        for key, title in _STAGE_ORDER
        if buckets[key]
    )
    return ExperimentPipeline(stages=stages)


def _operation_stage(operation: Operation, result: AnalysisResult) -> str | None:
    # A forward pass can belong to training or evaluation depending on the
    # surrounding pattern detected for that cell. Transformers calls that
    # explicitly pass labels are treated as supervised training evidence.
    if operation.kind == "forward_pass":
        if result.inference is not None and result.training_step is None:
            return "evaluation"
        if any(item.kind == "supervision_labels" for item in result.operations):
            return "training"
        if operation.framework == "transformers" and result.training_step is None:
            return None
        return "training"
    return _OPERATION_STAGE.get(operation.kind)
