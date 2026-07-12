from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class DataConfig:
    source: str = "hf"
    dataset_name: str = "ZODAOfficial/MindEdit-Bench"
    configs: list[str] = field(default_factory=lambda: ["single_view_spatial_editing", "cross_view_visibility"])
    split: str = "test"
    jsonl_path: str | None = None
    cache_dir: str | None = None
    task_filter: list[str] = field(default_factory=list)
    image_columns: list[str] = field(default_factory=lambda: ["view1", "view2", "view3"])
    id_column: str = "id"
    question_column: str = "question"
    choices_column: str = "choices"
    answer_column: str = "answer"
    task_column: str = "task"


@dataclass
class ModelConfig:
    backend: str = "qwen"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    attn_implementation: str | None = None
    max_new_tokens: int = 1200
    temperature: float = 0.2
    top_p: float = 0.9
    min_pixels: int | None = 200704
    max_pixels: int | None = 1003520
    api_base: str | None = None
    api_key: str | None = None
    api_timeout: float = 180.0
    option_score_temperature: float = 1.0


@dataclass
class CriticConfig:
    mode: str = "hybrid"
    symbolic_weight: float = 0.35
    vlm_weight: float = 0.65
    learned_weight: float = 0.0
    learned_model_path: str | None = None
    transition_threshold: float = 0.58
    causal_threshold: float = 0.08
    global_threshold: float = 0.62
    intervention_top_k: int = 4
    intervention_types: list[str] = field(default_factory=lambda: [
        "frame_swap", "operation_inverse", "relation_flip", "object_swap", "visibility_flip", "view_dropout"
    ])
    judge_repeats: int = 1


@dataclass
class RepairConfig:
    enabled: bool = True
    num_candidates: int = 3
    score_consistency_weight: float = 0.55
    score_confidence_weight: float = 0.25
    score_minimality_weight: float = 0.20
    abstain_margin: float = 0.02
    preserve_verified_prefix: bool = True


@dataclass
class RunConfig:
    output_dir: str = "outputs/mindedit_qwen25vl7b"
    seed: int = 42
    start_index: int = 0
    end_index: int | None = None
    limit: int | None = None
    num_traces: int = 3
    max_reasoning_steps: int = 8
    overwrite: bool = False
    run_baselines: bool = True
    baseline_methods: list[str] = field(default_factory=lambda: ["direct", "cot", "self_consistency", "full_reflection"])
    save_raw_responses: bool = True
    fail_fast: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls: type[T], payload: dict[str, Any] | None) -> T:
    payload = payload or {}
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - valid)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**payload)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    cfg = ExperimentConfig(
        data=_merge_dataclass(DataConfig, payload.get("data")),
        model=_merge_dataclass(ModelConfig, payload.get("model")),
        critic=_merge_dataclass(CriticConfig, payload.get("critic")),
        repair=_merge_dataclass(RepairConfig, payload.get("repair")),
        run=_merge_dataclass(RunConfig, payload.get("run")),
    )
    for dotted_key, value in (overrides or {}).items():
        target: Any = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise ValueError(f"Unknown override: {dotted_key}")
        setattr(target, parts[-1], value)
    return cfg
