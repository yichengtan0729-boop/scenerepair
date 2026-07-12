from __future__ import annotations

import numpy as np

from ..models.base import VisionLanguageModel
from ..parsing import parse_score_payload
from ..prompts import global_judge_prompt, transition_judge_prompt
from ..schemas import ReasoningTrace, SpatialExample, SpatialState


class VLMTransitionCritic:
    def __init__(self, model: VisionLanguageModel, repeats: int = 1) -> None:
        self.model = model
        self.repeats = max(1, int(repeats))

    def score_transition(
        self,
        example: SpatialExample,
        previous: SpatialState | None,
        current: SpatialState,
        images=None,
    ) -> dict:
        prompt = transition_judge_prompt(example, previous, current)
        payloads = [parse_score_payload(self.model.judge_json(images or example.images, prompt)) for _ in range(self.repeats)]
        scores = [float(item["score"]) for item in payloads]
        best = payloads[int(np.argmax(scores))]
        components: dict[str, float] = {}
        for key in {key for item in payloads for key in (item.get("components") or {})}:
            values = []
            for item in payloads:
                try:
                    values.append(float((item.get("components") or {}).get(key, item["score"])))
                except Exception:
                    pass
            if values:
                components[key] = float(np.mean(values))
        return {
            "score": float(np.mean(scores)),
            "error_type": best.get("error_type", "none"),
            "rationale": best.get("rationale", ""),
            "components": components,
        }

    def score_trace(self, example: SpatialExample, trace: ReasoningTrace, images=None) -> dict:
        payload = parse_score_payload(self.model.judge_json(images or example.images, global_judge_prompt(example, trace)))
        return payload
