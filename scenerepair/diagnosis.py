from __future__ import annotations

from collections import defaultdict

from .config import CriticConfig
from .critics import TransitionCritic
from .interventions import apply_intervention, generate_interventions
from .models.base import VisionLanguageModel
from .prompts import option_scoring_prompt
from .schemas import DiagnosisResult, ReasoningTrace, SpatialExample, TransitionAssessment
from .utils import js_divergence


class CausalDiagnoser:
    """Label-free transition diagnosis using consistency and intervention sensitivity."""

    def __init__(
        self,
        model: VisionLanguageModel,
        critic: TransitionCritic,
        config: CriticConfig,
    ) -> None:
        self.model = model
        self.critic = critic
        self.config = config

    def _distribution(self, example: SpatialExample, trace: ReasoningTrace, images=None) -> dict[str, float]:
        return self.model.score_options(images or example.images, option_scoring_prompt(example, trace), example.choices)

    @staticmethod
    def _aligned(distribution: dict[str, float], labels: list[str]) -> list[float]:
        return [float(distribution.get(label, 0.0)) for label in labels]

    def diagnose(self, example: SpatialExample, trace: ReasoningTrace) -> DiagnosisResult:
        if not trace.states:
            return DiagnosisResult(
                trace_id=trace.trace_id,
                original_distribution=self.model.uniform_distribution(example.choices),
                assessments=[],
                interventions=[],
                localized_step=None,
                localized_error_type="format",
                anomaly_score=1.0,
                causal_score=0.0,
                should_repair=False,
                global_consistency=0.0,
            )
        original_distribution = self._distribution(example, trace)
        trace.option_distribution = original_distribution
        labels = list(original_distribution)
        assessments: list[TransitionAssessment] = []
        previous = None
        for state in trace.states:
            assessment = self.critic.score_transition(example, previous, state)
            assessments.append(assessment)
            previous = state
        global_consistency = sum(item.consistency for item in assessments) / len(assessments)

        ranked = sorted(range(len(assessments)), key=lambda idx: (assessments[idx].consistency, idx))
        selected = set(ranked[: max(1, self.config.intervention_top_k)])
        for idx, assessment in enumerate(assessments):
            if assessment.consistency < self.config.transition_threshold:
                selected.add(idx)

        intervention_rows: list[dict] = []
        per_step_causal: dict[int, float] = defaultdict(float)
        per_step_improvement: dict[int, float] = defaultdict(float)
        for step_idx in sorted(selected):
            state = trace.states[step_idx]
            previous_state = trace.states[step_idx - 1] if step_idx > 0 else None
            base_consistency = assessments[step_idx].consistency
            for intervention in generate_interventions(
                state,
                allowed=self.config.intervention_types,
                num_views=len(example.images),
            ):
                modified_trace = apply_intervention(trace, step_idx, intervention)
                images = example.images
                if intervention.dropped_view is not None:
                    images = [
                        image
                        for image_idx, image in enumerate(example.images, start=1)
                        if image_idx != intervention.dropped_view
                    ]
                modified_distribution = self._distribution(example, modified_trace, images=images)
                modified_assessment = self.critic.score_transition(
                    example,
                    previous_state,
                    modified_trace.states[step_idx],
                    images=images,
                )
                divergence = js_divergence(
                    self._aligned(original_distribution, labels),
                    self._aligned(modified_distribution, labels),
                )
                improvement = max(0.0, modified_assessment.consistency - base_consistency)
                causal_effect = 0.7 * divergence + 0.3 * improvement
                per_step_causal[step_idx] = max(per_step_causal[step_idx], causal_effect)
                per_step_improvement[step_idx] = max(per_step_improvement[step_idx], improvement)
                intervention_rows.append(
                    {
                        "step_idx": step_idx,
                        "name": intervention.name,
                        "description": intervention.description,
                        "dropped_view": intervention.dropped_view,
                        "original_consistency": base_consistency,
                        "intervened_consistency": modified_assessment.consistency,
                        "consistency_improvement": improvement,
                        "answer_js_divergence": divergence,
                        "causal_effect": causal_effect,
                        "intervened_distribution": modified_distribution,
                        "intervened_state": intervention.state.to_dict(),
                    }
                )

        candidates: list[dict] = []
        for assessment in assessments:
            anomaly = 1.0 - assessment.consistency
            causal = per_step_causal.get(assessment.step_idx, 0.0)
            candidates.append(
                {
                    "step_idx": assessment.step_idx,
                    "anomaly": anomaly,
                    "causal": causal,
                    "improvement": per_step_improvement.get(assessment.step_idx, 0.0),
                    "error_type": assessment.error_type,
                }
            )
        valid = [
            item
            for item in candidates
            if item["anomaly"] >= (1.0 - self.config.transition_threshold)
            and item["causal"] >= self.config.causal_threshold
        ]
        if valid:
            localized = min(valid, key=lambda item: item["step_idx"])
            should_repair = True
        else:
            localized = max(candidates, key=lambda item: (item["anomaly"] * (item["causal"] + 1e-8), -item["step_idx"]))
            should_repair = bool(
                global_consistency < self.config.global_threshold
                and localized["causal"] >= self.config.causal_threshold
            )
        return DiagnosisResult(
            trace_id=trace.trace_id,
            original_distribution=original_distribution,
            assessments=assessments,
            interventions=intervention_rows,
            localized_step=int(localized["step_idx"]),
            localized_error_type=str(localized["error_type"]),
            anomaly_score=float(localized["anomaly"]),
            causal_score=float(localized["causal"]),
            should_repair=should_repair,
            global_consistency=float(global_consistency),
        )
