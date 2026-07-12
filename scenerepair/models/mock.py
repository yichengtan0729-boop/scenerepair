from __future__ import annotations

import json
import random
import re

from PIL import Image

from .base import VisionLanguageModel
from ..utils import labels_for_choices, softmax, stable_hash


class MockVisionLanguageModel(VisionLanguageModel):
    """Deterministic backend used for tests and pipeline validation."""

    def __init__(self, model_name: str = "mock") -> None:
        super().__init__()
        self.model_name = model_name

