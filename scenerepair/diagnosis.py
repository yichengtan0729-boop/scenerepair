from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from .calibration import interventional_agreement, weighted_geometric_mean
from .causal_graph import CausalDependencyGraph, build_causal_graph
from .config import CriticConfig
from .critics import TransitionCritic
from .interventions import Intervention, apply_intervention, generate_interventions
from .models.base import VisionLanguageModel
from .prompts import option_scoring_prompt
from .schemas import DiagnosisResult, ReasoningTrace, SpatialExample, SpatialState, StepType, TransitionAssessment
from .utils import js_divergence


class CausalDiagnoser:
    """Label-free causal-root diagnosis with necessity and sufficiency tests."""

    def __init__(self, model: VisionLanguageModel, critic: TransitionCritic, config: CriticConfig) -> None:
        self.model = model
        self.critic = critic
        self.config = config

    def _distribution(self, example: SpatialExample, trace: ReasoningTrace, images=None) -> dict[str, float]:
        return self.model.score_options(images or example.images, option_scoring_prompt(example, trace), example.choices)

    @staticmethod
    def _aligned(distribution: dict[str, float], labels: list[str]) -> list[float]:
        return [float(distribution.get(label, 0.0)) for label in labels]

    @staticmethod
    def _confidence(distribution: dict[str, float]) -> float:
        return max(distribution.values(), default=0.0)

    def _assess_trace(self, example: SpatialExample, trace: ReasoningTrace, images=None) -> list[TransitionAssessment]:
        output: list[TransitionAssessment] = []
        previous = None
        for state in trace.states:
            output.append(self.critic.score_transition(example, previous, state, images=images))
            previous = state
        return output

    @staticmethod
    def _mean_consistency(assessments: list[TransitionAssessment], nodes: list[int] | None = None) -> float:
        if nodes is None:
            values = [item.consistency for item in assessments]
        else:
            node_set = set(nodes)
            values = [item.consistency for item in assessments if item.step_idx in node_set]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _equivalent_variant(intervention: Intervention, variant_idx: int) -> Intervention:
        """Create lexical realizations that preserve the structured intervention."""
        variant = copy.deepcopy(intervention)
        state = variant.state
        if variant_idx == 0:
            return variant
        canonical = (
            f"type={state.step_type}; objects={','.join(state.objects) or 'none'}; frame={state.reference_frame}; "
            f"operation={state.operation}; before={'; '.join(state.relations_before) or 'none'}; "
            f"after={'; '.join(state.relations_after) or 'none'}; visibility={'; '.join(state.visibility) or 'none'}"
        )
        if variant_idx % 2 == 1:
            state.description = f"Equivalent structured state: {canonical}."
        else:
            state.description = f"Restated without changing semantics: {canonical}."
        state.metadata["equivalent_variant"] = variant_idx
        return variant

    @staticmethod
    def _task_relevant_interventions(state: SpatialState, configured: list[str]) -> list[str]:
        mapping = {
            StepType.OBSERVATION.value: {"object_swap", "view_dropout"},
            StepType.CORRESPONDENCE.value: {"object_swap", "view_dropout"},
            StepType.REFERENCE_FRAME.value: {"frame_swap", "view_dropout"},
            StepType.INITIAL_STATE.value: {"relation_flip", "object_swap", "frame_swap"},
            StepType.COUNTERFACTUAL_OPERATION.value: {"operation_inverse", "frame_swap"},
            StepType.UPDATED_STATE.value: {"relation_flip", "operation_inverse", "object_swap"},
            StepType.VISIBILITY.value: {"visibility_flip", "view_dropout", "relation_flip"},
        }
        relevant = mapping.get(state.step_type, set(configured))
        return [name for name in configured if name in relevant]

    def _intervention_payload(
        self,
        example: SpatialExample,
        trace: ReasoningTrace,
        graph: CausalDependencyGraph,
        original_distribution: dict[str, float],
        original_assessments: list[TransitionAssessment],
        step_idx: int,
        intervention: Intervention,
        labels: list[str],
        structural_cache: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        modified_trace = apply_intervention(trace, step_idx, intervention)
        images = example.images
        if intervention.dropped_view is not None:
            images = [
                image
                for image_idx, image in enumerate(example.images, start=1)
                if image_idx != intervention.dropped_view
            ]
        modified_distribution = self._distribution(example, modified_trace, images=images)
        affected_nodes = [step_idx] + graph.children.get(step_idx, [])

        if structural_cache is None:
            previous_state = modified_trace.states[step_idx - 1] if step_idx > 0 else None
            modified_local = self.critic.score_transition(
                example, previous_state, modified_trace.states[step_idx], images=images
            )
            modified_scores = [modified_local.consistency]
            original_scores = [original_assessments[step_idx].consistency]
            for child_idx in graph.children.get(step_idx, []):
                child = copy.deepcopy(modified_trace.states[child_idx])
                child.metadata["intervened_parent"] = modified_trace.states[step_idx].to_dict()
                previous = modified_trace.states[child_idx - 1] if child_idx > 0 else None
                modified_child = self.critic.score_transition(example, previous, child, images=images)
                modified_scores.append(modified_child.consistency)
                original_scores.append(original_assessments[child_idx].consistency)
            original_affected = sum(original_scores) / len(original_scores)
            modified_affected = sum(modified_scores) / len(modified_scores)
            structural_cache = {
                "intervened_consistency": modified_local.consistency,
                "original_descendant_consistency": original_affected,
                "intervened_descendant_consistency": modified_affected,
            }
        original_affected = structural_cache["original_descendant_consistency"]
        modified_affected = structural_cache["intervened_descendant_consistency"]
        answer_shift = js_divergence(
            self._aligned(original_distribution, labels),
            self._aligned(modified_distribution, labels),
        )
        consistency_drop = max(0.0, original_affected - modified_affected)
        consistency_gain = max(0.0, modified_affected - original_affected)
        confidence_gain = max(0.0, self._confidence(modified_distribution) - self._confidence(original_distribution))
        necessity = (
            self.config.answer_shift_weight * answer_shift
            + self.config.descendant_consistency_weight * consistency_drop
        )
        sufficiency = (
            0.65 * consistency_gain
            + 0.20 * max(
                0.0,
                structural_cache["intervened_consistency"] - original_assessments[step_idx].consistency,
            )
            + 0.15 * confidence_gain
        )
        predicted_label = max(modified_distribution, key=modified_distribution.get) if modified_distribution else ""
        return {
            "step_idx": step_idx,
            "name": intervention.name,
            "description": intervention.description,
            "dropped_view": intervention.dropped_view,
            "original_consistency": original_assessments[step_idx].consistency,
            "intervened_consistency": structural_cache["intervened_consistency"],
            "original_descendant_consistency": original_affected,
            "intervened_descendant_consistency": modified_affected,
            "answer_js_divergence": answer_shift,
            "necessity": necessity,
            "sufficiency": sufficiency,
            "consistency_drop": consistency_drop,
            "consistency_gain": consistency_gain,
            "predicted_label": predicted_label,
            "intervened_distribution": modified_distribution,
            "intervened_state": intervention.state.to_dict(),
            "affected_nodes": affected_nodes,
            "structural_cache": structural_cache,
        }

    def _select_root(self, candidates: list[dict[str, Any]], graph: CausalDependencyGraph) -> dict[str, Any]:
        valid = [
            item for item in candidates
            if item["anomaly"] >= (1.0 - self.config.transition_threshold)
            and item["necessity"] >= self.config.necessity_threshold
            and item["sufficiency"] >= self.config.sufficiency_threshold
            and item["root_score"] >= self.config.root_threshold
        ]
        if not valid:
            return max(candidates, key=lambda item: (item["root_score"], -graph.depth(item["step_idx"]), -item["step_idx"]))

        best_score = max(item["root_score"] for item in valid)
        competitive = [item for item in valid if item["root_score"] >= best_score - self.config.root_margin]
        competitive_steps = {item["step_idx"] for item in competitive}
        frontier = [
            item for item in competitive
            if not any(ancestor in competitive_steps for ancestor in graph.ancestors(item["step_idx"]))
        ]
        pool = frontier or competitive
        return min(pool, key=lambda item: (graph.depth(item["step_idx"]), item["step_idx"], -item["root_score"]))

    def diagnose(self, example: SpatialExample, trace: ReasoningTrace) -> DiagnosisResult:
        if not trace.states:
            return DiagnosisResult(
                trace_id=trace.trace_id,
                original_distribution=self.model.uniform_distribution(example.choices),
                assessments=[], interventions=[], localized_step=None,
                localized_error_type="format", anomaly_score=1.0, causal_score=0.0,
                should_repair=False, global_consistency=0.0,
            )

        graph = build_causal_graph(trace)
        for idx, state in enumerate(trace.states):
            state.depends_on = list(graph.parents.get(idx, []))
        original_distribution = self._distribution(example, trace)
        trace.option_distribution = original_distribution
        labels = list(original_distribution)
        assessments = self._assess_trace(example, trace)
        global_consistency = self._mean_consistency(assessments)

        eligible = [
            idx for idx, state in enumerate(trace.states)
            if state.step_type != StepType.ANSWER.value
        ] or list(range(len(assessments)))
        ranked = sorted(eligible, key=lambda idx: (assessments[idx].consistency, idx))
        selected = set(ranked[: max(1, self.config.intervention_top_k)])
        for idx in eligible:
            if assessments[idx].consistency < self.config.transition_threshold:
                selected.add(idx)

        intervention_rows: list[dict[str, Any]] = []
        per_step: dict[int, dict[str, list[Any]]] = defaultdict(lambda: {"families": [], "labels": []})
        variants = max(1, int(self.config.equivalent_variants))
        for step_idx in sorted(selected):
            allowed = self._task_relevant_interventions(
                trace.states[step_idx], self.config.intervention_types
            )
            for intervention in generate_interventions(
                trace.states[step_idx],
                allowed=allowed,
                num_views=len(example.images),
            ):
                family_rows: list[dict[str, Any]] = []
                structural_cache = None
                for variant_idx in range(variants):
                    variant = self._equivalent_variant(intervention, variant_idx)
                    row = self._intervention_payload(
                        example, trace, graph, original_distribution, assessments,
                        step_idx, variant, labels, structural_cache=structural_cache,
                    )
                    structural_cache = row["structural_cache"]
                    row.pop("structural_cache", None)
                    row["variant_idx"] = variant_idx
                    family_rows.append(row)
                    intervention_rows.append(row)
                necessity_values = [row["necessity"] for row in family_rows]
                sufficiency_values = [row["sufficiency"] for row in family_rows]
                family_effects = [max(row["necessity"], row["sufficiency"]) for row in family_rows]
                family_labels = [row["predicted_label"] for row in family_rows]
                agreement = interventional_agreement(family_effects, family_labels)
                family_necessity = sum(necessity_values) / len(necessity_values)
                family_sufficiency = sum(sufficiency_values) / len(sufficiency_values)
                per_step[step_idx]["families"].append({
                    "necessity": family_necessity,
                    "sufficiency": family_sufficiency,
                    "agreement": agreement,
                    "effect": max(family_necessity, family_sufficiency),
                    "name": intervention.name,
                })
                per_step[step_idx]["labels"].extend(family_labels)

        weights = {
            "anomaly": self.config.anomaly_root_weight,
            "necessity": self.config.necessity_root_weight,
            "sufficiency": self.config.sufficiency_root_weight,
            "ict": self.config.interventional_consistency_root_weight,
        }
        candidates: list[dict[str, Any]] = []
        for assessment in assessments:
            values = per_step.get(assessment.step_idx, {})
            families = list(values.get("families", []))
            necessity = max((item["necessity"] for item in families), default=0.0)
            sufficiency = max((item["sufficiency"] for item in families), default=0.0)
            effect_mass = sum(item["effect"] for item in families)
            ict = (
                sum(item["effect"] * item["agreement"] for item in families) / effect_mass
                if effect_mass > 1e-8 else 0.0
            )
            anomaly = 1.0 - assessment.consistency
            root_score = weighted_geometric_mean(
                {"anomaly": anomaly, "necessity": necessity, "sufficiency": sufficiency, "ict": ict},
                weights,
            )
            candidates.append({
                "step_idx": assessment.step_idx,
                "depth": graph.depth(assessment.step_idx),
                "anomaly": anomaly,
                "necessity": necessity,
                "sufficiency": sufficiency,
                "interventional_consistency": ict,
                "root_score": root_score,
                "error_type": assessment.error_type,
                "ancestors": graph.ancestors(assessment.step_idx),
                "descendants": graph.descendants(assessment.step_idx),
            })

        localized = self._select_root(candidates, graph)
        strict_valid = (
            localized["anomaly"] >= (1.0 - self.config.transition_threshold)
            and localized["necessity"] >= self.config.necessity_threshold
            and localized["sufficiency"] >= self.config.sufficiency_threshold
            and localized["root_score"] >= self.config.root_threshold
        )
        should_repair = bool(strict_valid and global_consistency < 1.0)
        return DiagnosisResult(
            trace_id=trace.trace_id,
            original_distribution=original_distribution,
            assessments=assessments,
            interventions=intervention_rows,
            localized_step=int(localized["step_idx"]),
            localized_error_type=str(localized["error_type"]),
            anomaly_score=float(localized["anomaly"]),
            causal_score=float(localized["root_score"]),
            should_repair=should_repair,
            global_consistency=float(global_consistency),
            necessity_score=float(localized["necessity"]),
            sufficiency_score=float(localized["sufficiency"]),
            interventional_consistency=float(localized["interventional_consistency"]),
            root_score=float(localized["root_score"]),
            dependency_graph=graph.to_dict(),
            descendant_steps=[int(localized["step_idx"])] + graph.descendants(int(localized["step_idx"])),
            candidate_roots=sorted(candidates, key=lambda item: item["root_score"], reverse=True),
        )
