from __future__ import annotations

from ..config import CriticConfig
from ..models.base import VisionLanguageModel
from ..schemas import ReasoningTrace, SpatialExample, SpatialState, TransitionAssessment
from .symbolic import SymbolicTransitionCritic
from .vlm import VLMTransitionCritic


class TransitionCritic:
    def __init__(self, config: CriticConfig, model: VisionLanguageModel) -> None:
        self.config = config
        self.symbolic = SymbolicTransitionCritic()
        self.vlm = VLMTransitionCritic(model, repeats=config.judge_repeats)
        self.learned = None
        if config.learned_model_path:
            from .learned import LearnedTransitionCritic

            self.learned = LearnedTransitionCritic(config.learned_model_path)

    def _enabled_weights(self) -> dict[str, float]:
        mode = self.config.mode.lower()
        weights = {
            "symbolic": self.config.symbolic_weight if mode in {"symbolic", "hybrid", "all"} else 0.0,
            "vlm": self.config.vlm_weight if mode in {"vlm", "hybrid", "all"} else 0.0,
            "learned": self.config.learned_weight if self.learned is not None and mode in {"learned", "hybrid", "all"} else 0.0,
        }
        if mode == "symbolic":
            weights = {"symbolic": 1.0, "vlm": 0.0, "learned": 0.0}
        elif mode == "vlm":
            weights = {"symbolic": 0.0, "vlm": 1.0, "learned": 0.0}
        elif mode == "learned":
            weights = {"symbolic": 0.0, "vlm": 0.0, "learned": 1.0}
        total = sum(weights.values())
        if total <= 0:
            return {"symbolic": 1.0, "vlm": 0.0, "learned": 0.0}
        return {key: value / total for key, value in weights.items()}

    def score_transition(
        self,
        example: SpatialExample,
        previous: SpatialState | None,
        current: SpatialState,
        images=None,
    ) -> TransitionAssessment:
        weights = self._enabled_weights()
        symbolic = self.symbolic.score(previous, current)
        vlm_payload = None
        if weights["vlm"] > 0:
            vlm_payload = self.vlm.score_transition(example, previous, current, images=images)
        learned_score = self.learned.score(previous, current) if weights["learned"] > 0 and self.learned else None
        score = weights["symbolic"] * symbolic.score
        if vlm_payload is not None:
            score += weights["vlm"] * float(vlm_payload["score"])
        if learned_score is not None:
            score += weights["learned"] * learned_score
        error_type = symbolic.error_type
        rationale_parts = [symbolic.rationale]
        components = {f"symbolic_{key}": value for key, value in symbolic.components.items()}
        if vlm_payload is not None:
            if float(vlm_payload["score"]) < symbolic.score or symbolic.error_type == "none":
                error_type = str(vlm_payload.get("error_type", error_type))
            rationale_parts.append(str(vlm_payload.get("rationale", "")))
            components.update({f"vlm_{key}": value for key, value in (vlm_payload.get("components") or {}).items()})
        if learned_score is not None:
            components["learned"] = learned_score
        if score >= 0.72:
            error_type = "none"
        return TransitionAssessment(
            step_idx=current.step_idx,
            consistency=float(min(1.0, max(0.0, score))),
            symbolic_score=symbolic.score,
            vlm_score=float(vlm_payload["score"]) if vlm_payload is not None else None,
            learned_score=learned_score,
            error_type=error_type,
            rationale=" ".join(part for part in rationale_parts if part),
            components=components,
        )

    def score_trace(self, example: SpatialExample, trace: ReasoningTrace, images=None) -> float:
        if not trace.states:
            return 0.0
        transition_scores = []
        previous = None
        for state in trace.states:
            transition_scores.append(self.score_transition(example, previous, state, images=images).consistency)
            previous = state
        local_score = sum(transition_scores) / len(transition_scores)
        weights = self._enabled_weights()
        if weights["vlm"] <= 0:
            return local_score
        global_payload = self.vlm.score_trace(example, trace, images=images)
        global_score = float(global_payload["score"])
        return 0.65 * local_score + 0.35 * global_score
