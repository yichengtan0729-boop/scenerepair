from __future__ import annotations

from ..config import ModelConfig
from .base import VisionLanguageModel


def build_model(config: ModelConfig) -> VisionLanguageModel:
    backend = config.backend.lower()
    if backend == "mock":
        from .mock import MockVisionLanguageModel

        return MockVisionLanguageModel(config.model_name)
    if backend in {"qwen", "hf", "transformers"}:
        from .qwen_vl import QwenVisionLanguageModel

        return QwenVisionLanguageModel(config)
    if backend in {"api", "openai", "vllm"}:
        from .openai_api import OpenAICompatibleVisionLanguageModel

        return OpenAICompatibleVisionLanguageModel(config)
    raise ValueError(f"Unknown model backend: {config.backend}")
