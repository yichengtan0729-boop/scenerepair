from __future__ import annotations

from dataclasses import dataclass

from ..schemas import SpatialState, StepType


VALID_FRAMES = {"world", "unknown", "camera_1", "camera_2", "camera_3"}
OPERATION_WORDS = {
    "move",
    "translate",
    "translation",
    "rotate",
    "rotation",
    "clockwise",
    "counterclockwise",
    "counter-clockwise",
    "left",
    "right",
    "forward",
    "backward",
    "toward",
    "away",
    "pivot",
}
RELATIONS = {
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "near",
    "far",
    "inside",
    "overlap",
    "visible",
    "occluded",
    "absent",
}


@dataclass
class SymbolicAssessment:
    score: float
    error_type: str
    rationale: str
    components: dict[str, float]


class SymbolicTransitionCritic:
    """Schema- and operation-aware critic that never accesses the answer label."""

    @staticmethod
    def _frame_score(state: SpatialState) -> float:
        frame = state.reference_frame.lower().replace(" ", "_")
        if frame in VALID_FRAMES or frame.startswith("object:") or frame.startswith("camera_"):
            return 1.0
        if frame in {"camera", "object", "egocentric", "allocentric"}:
            return 0.65
        return 0.25

    @staticmethod
    def _format_score(state: SpatialState) -> float:
        score = 1.0
        if not state.description:
            score -= 0.45
        if state.step_idx < 0:
            score -= 0.25
        if not state.step_type:
            score -= 0.3
        if not 0.0 <= state.confidence <= 1.0:
            score -= 0.2
        return min(1.0, max(0.0, score))

    @staticmethod
    def _operation_score(state: SpatialState) -> float:
        text = f"{state.operation} {state.description}".lower()
        has_operation = any(word in text for word in OPERATION_WORDS)
        if state.step_type == StepType.COUNTERFACTUAL_OPERATION.value:
            if state.operation.lower() in {"", "none", "unknown"}:
                return 0.1
            return 1.0 if has_operation else 0.45
        if state.operation.lower() not in {"", "none", "unknown"} and not has_operation:
            return 0.55
        return 0.9

    @staticmethod
    def _state_score(previous: SpatialState | None, state: SpatialState) -> float:
        score = 0.85
        if state.step_type in {StepType.UPDATED_STATE.value, StepType.VISIBILITY.value}:
            if not state.relations_after and not state.visibility:
                score -= 0.5
        if state.step_type == StepType.INITIAL_STATE.value and not state.relations_before and not state.relations_after:
            score -= 0.35
        relation_text = " ".join(state.relations_before + state.relations_after + state.visibility).lower()
        if relation_text and not any(token in relation_text for token in RELATIONS):
            score -= 0.15
        if previous is not None:
            if state.step_idx <= previous.step_idx:
                score -= 0.4
            if (
                previous.reference_frame not in {"", "unknown"}
                and state.reference_frame not in {"", "unknown", previous.reference_frame}
                and state.step_type not in {StepType.REFERENCE_FRAME.value, StepType.COUNTERFACTUAL_OPERATION.value}
            ):
                score -= 0.2
        return min(1.0, max(0.0, score))

    @staticmethod
    def _view_score(state: SpatialState) -> float:
        if not state.evidence_views:
            return 0.75 if state.step_type in {StepType.COUNTERFACTUAL_OPERATION.value, StepType.ANSWER.value} else 0.55
        valid = sum(1 for value in state.evidence_views if value in {1, 2, 3})
        return valid / len(state.evidence_views)

    def score(self, previous: SpatialState | None, state: SpatialState) -> SymbolicAssessment:
        components = {
            "format": self._format_score(state),
            "frame": self._frame_score(state),
            "operation": self._operation_score(state),
            "state": self._state_score(previous, state),
            "view": self._view_score(state),
        }
        weights = {"format": 0.12, "frame": 0.23, "operation": 0.24, "state": 0.29, "view": 0.12}
        score = sum(components[key] * weights[key] for key in weights)
        dominant = min(components, key=components.get)
        error_map = {
            "format": "format",
            "frame": "reference_frame",
            "operation": "operation",
            "state": "relation_update",
            "view": "unsupported",
        }
        error_type = "none" if score >= 0.7 else error_map[dominant]
        rationale = f"Lowest symbolic component: {dominant}={components[dominant]:.3f}."
        return SymbolicAssessment(score=float(score), error_type=error_type, rationale=rationale, components=components)
