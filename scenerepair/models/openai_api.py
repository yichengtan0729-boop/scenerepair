from __future__ import annotations

import base64
import io
import json
from typing import Any

from PIL import Image

from .base import VisionLanguageModel
from ..config import ModelConfig
from ..parsing import extract_json_object
from ..utils import labels_for_choices, normalize_label


class OpenAICompatibleVisionLanguageModel(VisionLanguageModel):
    """OpenAI-compatible backend for hosted models and vLLM multimodal servers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install API support with: pip install -e '.[api]'") from exc
        self.client = OpenAI(
            api_key=config.api_key or "EMPTY",
            base_url=config.api_base,
            timeout=config.api_timeout,
        )

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _content(self, images: list[Image.Image], prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for idx, image in enumerate(images, start=1):
            content.append({"type": "text", "text": f"View {idx}:"})
            content.append({"type": "image_url", "image_url": {"url": self._data_url(image)}})
        content.append({"type": "text", "text": prompt})
        return content

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
        self.stats.generate_calls += 1
        self.stats.input_images += len(images)
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": self._content(images, prompt)}],
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
            "max_tokens": self.config.max_new_tokens if max_new_tokens is None else max_new_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def score_options(self, images: list[Image.Image], prompt: str, choices: list[str]) -> dict[str, float]:
        self.stats.score_calls += 1
        self.stats.input_images += len(images)
        labels = labels_for_choices(len(choices))
        scoring_prompt = prompt + "\nReturn JSON only with a normalized probability for every answer letter: " + json.dumps(
            {label: 0.0 for label in labels}
        )
        try:
            raw = self.generate(images, scoring_prompt, temperature=0.0, top_p=1.0, max_new_tokens=256)
            payload = extract_json_object(raw)
            values = [max(0.0, float(payload.get(label, 0.0))) for label in labels]
            total = sum(values)
            if total <= 0:
                raise ValueError("non-positive probability sum")
            return {label: value / total for label, value in zip(labels, values)}
        except Exception as exc:
            self.stats.errors.append(f"score_options:{exc}")
            raw = self.generate(images, prompt, temperature=0.0, top_p=1.0, max_new_tokens=16)
            prediction = normalize_label(raw, len(choices))
            if prediction not in labels:
                return self.uniform_distribution(choices)
            off = 0.02 / max(len(labels) - 1, 1)
            return {label: 0.98 if label == prediction else off for label in labels}
