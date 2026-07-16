from __future__ import annotations

import copy
import json
from difflib import SequenceMatcher

from .calibration import lower_confidence_bound
from .config import RepairConfig
from .critics import TransitionCritic
from .models.base import VisionLanguageModel
from .parsing import parse_reasoning_trace
from .prompts import option_scoring_prompt, repair_prompt
from .schemas import DiagnosisResult, ReasoningTrace, SpatialExample


class MinimalSuffixRepairer:
    """Descendant-closed suffix repair with calibrated label-free acceptance."""

    def __init__(self, model: VisionLanguageModel, critic: TransitionCritic, config: RepairConfig, max_reasoning_steps: int) -> None:
        self.model = model
        self.critic = critic
        self.config = config
        self.max_reasoning_steps = max_reasoning_steps

    @staticmethod
    def _minimality(original: ReasoningTrace, candidate: ReasoningTrace, localized_step: int) -> float:
        original_text = json.dumps(
            [state.to_dict() for state in original.states[localized_step:]],
            sort_keys=True, ensure_ascii=False,
        )
        candidate_text = json.dumps(
            [state.to_dict() for state in candidate.states[localized_step:]],
            sort_keys=True, ensure_ascii=False,
        )
        return float(SequenceMatcher(None, original_text, candidate_text).ratio())

    def _combine_suffix(self, original: ReasoningTrace, suffix: ReasoningTrace, localized_step: int) -> ReasoningTrace:
        prefix = copy.deepcopy(original.states[:localized_step])
        new_suffix = copy.deepcopy(suffix.states)
        states = prefix + new_suffix
        for idx, state in enumerate(states):
            state.step_idx = idx
            state.depends_on = sorted(parent for parent in state.depends_on if parent < idx)
        return ReasoningTrace(
            trace_id=original.trace_id,
            states=states[: self.max_reasoning_steps],
            predicted_label=suffix.predicted_label or original.predicted_label,
            raw_response=suffix.raw_response,
            parse_warnings=list(suffix.parse_warnings),
            generation_seed=suffix.generation_seed,
        )

    def _score_once(
        self,
        example: SpatialExample,
        original: ReasoningTrace,
        candidate: ReasoningTrace,
        localized_step: int,
    ) -> dict:
        consistency = self.critic.score_trace(example, candidate)
        distribution = self.model.score_options(
            example.images,
            option_scoring_prompt(example, candidate),
            example.choices,
        )
        candidate.option_distribution = distribution
        confidence = max(distribution.values()) if distribution else 0.0
        minimality = self._minimality(original, candidate, localized_step)
        total = (
            self.config.score_consistency_weight * consistency
            + self.config.score_confidence_weight * confidence
            + self.config.score_minimality_weight * minimality
        )
        return {
            "consistency": float(consistency),
            "confidence": float(confidence),
            "minimality": float(minimality),
            "total": float(total),
            "distribution": distribution,
        }

    @staticmethod
    def _mean_record(samples: list[dict]) -> dict:
        if not samples:
            return {"consistency": 0.0, "confidence": 0.0, "minimality": 0.0, "total": 0.0, "distribution": {}}
        keys = ("consistency", "confidence", "minimality", "total")
        output = {key: sum(float(item[key]) for item in samples) / len(samples) for key in keys}
        output["distribution"] = samples[-1].get("distribution", {})
        return output

    def _score_samples(
        self,
        example: SpatialExample,
        original: ReasoningTrace,
        candidate: ReasoningTrace,
        localized_step: int,
    ) -> tuple[list[dict], dict]:
        samples = [
            self._score_once(example, original, candidate, localized_step)
            for _ in range(max(1, int(self.config.acceptance_repeats)))
        ]
        return samples, self._mean_record(samples)

    def repair(self, example: SpatialExample, trace: ReasoningTrace, diagnosis: DiagnosisResult) -> dict:
        localized_step = diagnosis.localized_step
        if localized_step is not None and not self.config.preserve_verified_prefix:
            localized_step = 0
        if (
            not self.config.enabled
            or not diagnosis.should_repair
            or localized_step is None
            or localized_step >= len(trace.states)
        ):
            distribution = diagnosis.original_distribution
            confidence = max(distribution.values(), default=0.0)
            original_record = {
                "consistency": diagnosis.global_consistency,
                "confidence": confidence,
                "minimality": 1.0,
                "total": (
                    self.config.score_consistency_weight * diagnosis.global_consistency
                    + self.config.score_confidence_weight * confidence
                    + self.config.score_minimality_weight
                ),
                "distribution": distribution,
            }
            return {
                "repair_attempted": False,
                "repair_applied": False,
                "abstained": True,
                "reason": "diagnosis_below_root_threshold",
                "localized_step": localized_step,
                "descendant_steps": diagnosis.descendant_steps,
                "original_trace": trace.to_dict(),
                "repaired_trace": trace.to_dict(),
                "original_score": original_record,
                "selected_score": original_record,
                "acceptance": {"lcb": float("-inf"), "mean_gain": 0.0, "n": 0},
                "candidates": [],
            }

        prompt = repair_prompt(
            example,
            trace,
            localized_step,
            diagnosis.to_dict(),
            max(1, self.max_reasoning_steps - localized_step),
        )
        original_samples, original_record = self._score_samples(example, trace, trace, localized_step)
        candidates: list[dict] = []
        for candidate_id in range(max(1, self.config.num_candidates)):
            raw = self.model.generate(
                example.images,
                prompt,
                seed=(trace.generation_seed or 0) + 1000 + candidate_id,
            )
            suffix = parse_reasoning_trace(raw, trace.trace_id, len(example.choices))
            suffix.generation_seed = (trace.generation_seed or 0) + 1000 + candidate_id
            combined = self._combine_suffix(trace, suffix, localized_step)
            score_samples, score = self._score_samples(example, trace, combined, localized_step)
            gains = [
                float(candidate_sample["total"]) - float(original_sample["total"])
                for candidate_sample, original_sample in zip(score_samples, original_samples)
            ]
            calibration = lower_confidence_bound(gains, confidence=self.config.acceptance_confidence)
            consistency_gain = float(score["consistency"]) - float(original_record["consistency"])
            candidates.append({
                "candidate_id": candidate_id,
                "trace": combined.to_dict(),
                "score": score,
                "score_samples": score_samples,
                "gain_samples": gains,
                "calibration": calibration,
                "consistency_gain": consistency_gain,
            })

        best = max(candidates, key=lambda item: (item["calibration"]["lcb"], item["score"]["total"]))
        apply_repair = bool(
            best["calibration"]["lcb"] > self.config.abstain_margin
            and best["consistency_gain"] >= self.config.minimum_consistency_gain
        )
        selected_trace = best["trace"] if apply_repair else trace.to_dict()
        selected_score = best["score"] if apply_repair else original_record
        return {
            "repair_attempted": True,
            "repair_applied": apply_repair,
            "abstained": not apply_repair,
            "reason": "positive_calibrated_lower_bound" if apply_repair else "calibrated_gain_not_positive",
            "localized_step": localized_step,
            "descendant_steps": diagnosis.descendant_steps,
            "original_trace": trace.to_dict(),
            "repaired_trace": selected_trace,
            "original_score": original_record,
            "selected_score": selected_score,
            "acceptance": {
                "candidate_id": best["candidate_id"],
                "mean_gain": best["calibration"]["mean"],
                "lcb": best["calibration"]["lcb"],
                "confidence": self.config.acceptance_confidence,
                "n": best["calibration"]["n"],
                "consistency_gain": best["consistency_gain"],
            },
            "candidates": candidates,
        }
