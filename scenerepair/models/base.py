from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ..utils import labels_for_choices


@dataclass
class ModelStats:
    generate_calls: int = 0
    score_calls: int = 0
    judge_calls: int = 0
    input_images: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generate_calls": self.generate_calls,
            "score_calls": self.score_calls,
            "judge_calls": self.judge_calls,
            "input_images": self.input_images,
            "errors": self.errors,
        }


class VisionLanguageModel(ABC):
    def __init__(self) -> None:
        self.stats = ModelStats()

    @abstractmethod
    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def score_options(
        self,
        images: list[Image.Image],
        prompt: str,
        choices: list[str],
    ) -> dict[str, float]:
        raise NotImplementedError

    def judge_json(self, images: list[Image.Image], prompt: str) -> str:
        self.stats.judge_calls += 1
        return self.generate(images, prompt, temperature=0.0, top_p=1.0)

    @staticmethod
    def uniform_distribution(choices: list[str]) -> dict[str, float]:
        labels = labels_for_choices(len(choices))
        probability = 1.0 / len(labels)
        return {label: probability for label in labels}
