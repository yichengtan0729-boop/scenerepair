from __future__ import annotations

from .models.base import VisionLanguageModel
from .prompts import cot_prompt, direct_prompt, reflection_prompt
from .schemas import SpatialExample
from .utils import normalize_label, weighted_vote


def run_baselines(
    model: VisionLanguageModel,
    example: SpatialExample,
    methods: list[str],
    num_traces: int,
    seed: int,
) -> dict:
    results: dict[str, dict] = {}
    cot_response = ""
    if "direct" in methods:
        response = model.generate(example.images, direct_prompt(example), temperature=0.0, max_new_tokens=32, seed=seed)
        results["direct"] = {"prediction": normalize_label(response, len(example.choices)), "response": response}
    if any(method in methods for method in {"cot", "self_consistency", "full_reflection"}):
        cot_response = model.generate(example.images, cot_prompt(example), seed=seed + 1)
        results["cot"] = {"prediction": normalize_label(cot_response, len(example.choices)), "response": cot_response}
    if "self_consistency" in methods:
        responses = []
        labels = []
        for idx in range(max(1, num_traces)):
            response = model.generate(example.images, cot_prompt(example), seed=seed + 100 + idx)
            responses.append(response)
            labels.append(normalize_label(response, len(example.choices)))
        winner, distribution = weighted_vote(labels, [1.0] * len(labels))
        results["self_consistency"] = {
            "prediction": winner,
            "vote_distribution": distribution,
            "responses": responses,
        }
    if "full_reflection" in methods:
        if not cot_response:
            cot_response = model.generate(example.images, cot_prompt(example), seed=seed + 1)
        response = model.generate(example.images, reflection_prompt(example, cot_response), seed=seed + 200)
        results["full_reflection"] = {
            "prediction": normalize_label(response, len(example.choices)),
            "response": response,
            "original_response": cot_response,
        }
    return results
