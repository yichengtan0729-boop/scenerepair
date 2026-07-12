from __future__ import annotations

import copy
import json
from difflib import SequenceMatcher

from .config import RepairConfig
from .critics import TransitionCritic
from .models.base import VisionLanguageModel
from .parsing import parse_reasoning_trace
from .prompts import option_scoring_prompt, repair_prompt
from .schemas import DiagnosisResult, ReasoningTrace, SpatialExample


class MinimalSuffixRepairer:
    def __init__(
        self,
        model: VisionLanguageModel,
        critic: TransitionCritic,
        config: RepairConfig,
        max_reasoning_steps: int,
    ) -> None:
        self.model = model
        self.critic = critic
        self.config = config
        self.max_reasoning_steps = max_reasoning_steps

    @staticmethod
    def _minimality(original: ReasoningTrace, candidate: ReasoningTrace, localized_step: int) -> float:
        original_text = json.dumps(
            [state.to_dict() for state in original.states[localized_step:]],
            sort_keys=True,
            ensure_ascii=False,
        )
        candidate_text = json.dumps(
            [state.to_dict() for state in candidate.states[localized_step:]],
            sort_keys=True,
            ensure_ascii=False,
        )
        return float(SequenceMatcher(None, original_text, candidate_text).ratio())

    def _combine_suffix(
        self,
        original: ReasoningTrace,
        suffix: ReasoningTrace,
        localized_step: int,
    ) -> ReasoningTrace:
        prefix = copy.deepcopy(original.states[:localized_step])
        new_suffix = copy.deepcopy(suffix.states)
        states = prefix + new_suffix
        for idx, state in enumerate(states):
            state.step_idx = idx
        return ReasoningTrace(
            trace_id=original.trace_id,
            states=states[: self.max_reasoning_steps],
            predicted_label=suffix.predicted_label or original.predicted_label,
            raw_response=suffix.raw_response,
            parse_warnings=list(suffix.parse_warnings),
            generation_seed=suffix.generation_seed,
        )

    def _candidate_score(
        self,
        example: SpatialExample,
        original: ReasoningTrace,
        candidate: ReasoningTrace,
        localized_step: int,
    ) -> tuple[float, dict]:
        consistency = self.critic.score_trace(example, candidate)
        distribution = self.model.score_options(
            example.images,
            option_scoring_prompt(example, candidate),
            example.choices,
        )
        candidate.option_distribution = distribution
        confidence = max(distribution.values()) if distribution else 0.0
        minimality = self._minimality(original, candidate, localized_step)
        score = (
            self.config.score_consistency_weight * consistency
            + self.config.score_confidence_weight * confidence
            + self.config.score_minimality_weight * minimality
        )
        return float(score), {
            "consistency": float(consistency),
            "confidence": float(confidence),
            "minimality": float(minimality),
            "total": float(score),
            "distribution": distribution,
        }

    def repair(
        self,
        example: SpatialExample,
        trace: ReasoningTrace,
        diagnosis: DiagnosisResult,
    ) -> dict:
        localized_step = diagnosis.localized_step
        if localized_step is not None and not self.config.preserve_verified_prefix:
            localized_step = 0
        original_distribution = diagnosis.original_distribution
        original_confidence = max(original_distribution.values()) if original_distribution else 0.0
        original_score = (
            self.config.score_consistency_weight * diagnosis.global_consistency
            + self.config.score_confidence_weight * original_confidence
            + self.config.score_minimality_weight
        )
        original_record = {
            "consistency": diagnosis.global_consistency,
            "confidence": original_confidence,
            "minimality": 1.0,
            "total": original_score,
            "distribution": original_distribution,
        }
        if (
            not self.config.enabled
            or not diagnosis.should_repair
            or localized_step is None
            or localized_step >= len(trace.states)
        ):
            return {
                "repair_attempted": False,
                "repair_applied": False,
                "abstained": True,
                "reason": "diagnosis_below_threshold",
                "localized_step": localized_step,
                "original_trace": trace.to_dict(),
                "repaired_trace": trace.to_dict(),
                "original_score": original_record,
                "selected_score": original_record,
                "candidates": [],
            }

        prompt = repair_prompt(
            example,
            trace,
            localized_step,
            diagnosis.to_dict(),
            max(1, self.max_reasoning_steps - localized_step),
        )
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
            score, components = self._candidate_score(example, trace, combined, localized_step)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "trace": combined.to_dict(),
                    "score": components,
                }
            )
        best = max(candidates, key=lambda item: item["score"]["total"])
        apply_repair = best["score"]["total"] > original_score + self.config.abstain_margin
        selected_trace = best["trace"] if apply_repair else trace.to_dict()
        selected_score = best["score"] if apply_repair else original_record
        return {
            "repair_attempted": True,
            "repair_applied": apply_repair,
            "abstained": not apply_repair,
            "reason": "candidate_improved_objective" if apply_repair else "no_candidate_cleared_abstention_margin",
            "localized_step": localized_step,
            "original_trace": trace.to_dict(),
            "repaired_trace": selected_trace,
            "original_score": original_record,
            "selected_score": selected_score,
            "candidates": candidates,
        }
