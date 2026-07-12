from __future__ import annotations

from pathlib import Path

from PIL import Image

from .config import DataConfig
from .io import read_jsonl
from .schemas import SpatialExample
from .utils import normalize_label


def _coerce_choices(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value)]
    raise ValueError(f"Unsupported choices value: {type(value)!r}")


def _open_local_image(value, base_dir: Path) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("path"):
            value = value["path"]
        elif value.get("bytes"):
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return Image.open(path).convert("RGB")


def load_local_jsonl(config: DataConfig) -> list[SpatialExample]:
    if not config.jsonl_path:
        raise ValueError("data.jsonl_path is required when source=jsonl")
    path = Path(config.jsonl_path)
    rows = read_jsonl(path)
    examples: list[SpatialExample] = []
    excluded = set(
        config.image_columns
        + [
            config.id_column,
            config.task_column,
            config.question_column,
            config.choices_column,
            config.answer_column,
            "images",
        ]
    )
    for idx, row in enumerate(rows):
        choices = _coerce_choices(row[config.choices_column])
        image_values = row.get("images")
        if image_values is None:
            image_values = [row[col] for col in config.image_columns if row.get(col) is not None]
        images = [_open_local_image(item, path.parent) for item in image_values]
        answer = normalize_label(row.get(config.answer_column, ""), len(choices))
        examples.append(
            SpatialExample(
                example_id=str(row.get(config.id_column, idx)),
                task=str(row.get(config.task_column, "local")),
                question=str(row[config.question_column]),
                choices=choices,
                answer=answer,
                images=images,
                metadata={k: v for k, v in row.items() if k not in excluded},
            )
        )
    return examples


def load_hf(config: DataConfig) -> list[SpatialExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install SceneRepair with the hf extra: pip install -e '.[hf]'") from exc
    configs = config.configs or [None]
    examples: list[SpatialExample] = []
    excluded = set(
        config.image_columns
        + [
            config.id_column,
            config.task_column,
            config.question_column,
            config.choices_column,
            config.answer_column,
        ]
    )
    for subset in configs:
        dataset = load_dataset(config.dataset_name, subset, split=config.split, cache_dir=config.cache_dir)
        for idx, row in enumerate(dataset):
            task = str(row.get(config.task_column, subset or "default"))
            if config.task_filter and task not in config.task_filter:
                continue
            choices = _coerce_choices(row[config.choices_column])
            images: list[Image.Image] = []
            for column in config.image_columns:
                image = row.get(column)
                if image is None:
                    continue
                if isinstance(image, Image.Image):
                    images.append(image.convert("RGB"))
                elif isinstance(image, dict) and image.get("bytes"):
                    import io

                    images.append(Image.open(io.BytesIO(image["bytes"])).convert("RGB"))
                elif isinstance(image, dict) and image.get("path"):
                    images.append(Image.open(image["path"]).convert("RGB"))
                else:
                    images.append(Image.open(image).convert("RGB"))
            answer = normalize_label(row.get(config.answer_column, ""), len(choices))
            example_id = str(row.get(config.id_column, f"{subset}-{idx}"))
            metadata = {k: v for k, v in row.items() if k not in excluded}
            metadata["subset"] = subset
            examples.append(
                SpatialExample(
                    example_id=example_id,
                    task=task,
                    question=str(row[config.question_column]),
                    choices=choices,
                    answer=answer,
                    images=images,
                    metadata=metadata,
                )
            )
    return examples


def load_examples(
    config: DataConfig,
    start_index: int = 0,
    end_index: int | None = None,
    limit: int | None = None,
) -> list[SpatialExample]:
    source = config.source.lower()
    if source == "hf":
        examples = load_hf(config)
    elif source == "jsonl":
        examples = load_local_jsonl(config)
    else:
        raise ValueError(f"Unknown data source: {config.source}")
    stop = len(examples) if end_index is None else min(len(examples), end_index)
    selected = examples[max(0, start_index) : stop]
    if limit is not None:
        selected = selected[:limit]
    return selected
