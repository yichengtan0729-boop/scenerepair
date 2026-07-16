from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .utils import normalize_label


class StepType(str, Enum):
    OBSERVATION = "observation"
    CORRESPONDENCE = "correspondence"
    REFERENCE_FRAME = "reference_frame"
    INITIAL_STATE = "initial_state"
    COUNTERFACTUAL_OPERATION = "counterfactual_operation"
    UPDATED_STATE = "updated_state"
    VISIBILITY = "visibility"
    ANSWER = "answer"
    OTHER = "other"


@dataclass
class SpatialState:
    step_idx: int
    step_type: str
    description: str
    objects: list[str] = field(default_factory=list)
    reference_frame: str = "unknown"
    operation: str = "none"
    relations_before: list[str] = field(default_factory=list)
    relations_after: list[str] = field(default_factory=list)
    visibility: list[str] = field(default_factory=list)
    evidence_views: list[int] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], step_idx: int | None = None) -> "SpatialState":
        idx = int(payload.get("step_idx", step_idx if step_idx is not None else 0))
        step_type = str(payload.get("step_type", StepType.OTHER.value)).lower()
        if step_type not in {item.value for item in StepType}:
            step_type = StepType.OTHER.value

        def _list(key: str) -> list[str]:
            value = payload.get(key, [])
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            return [str(item) for item in value]

        def _int_list(key: str) -> list[int]:
            values = payload.get(key, []) or []
            output: list[int] = []
            for value in values:
                try:
                    output.append(int(value))
                except Exception:
                    continue
            return sorted(set(output))

        try:
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.5))))
        except Exception:
            confidence = 0.5
        return cls(
            step_idx=idx,
            step_type=step_type,
            description=str(payload.get("description", "")).strip(),
            objects=_list("objects"),
            reference_frame=str(payload.get("reference_frame", "unknown")).strip() or "unknown",
            operation=str(payload.get("operation", "none")).strip() or "none",
            relations_before=_list("relations_before"),
            relations_after=_list("relations_after"),
            visibility=_list("visibility"),
            evidence_views=_int_list("evidence_views"),
            depends_on=_int_list("depends_on"),
            confidence=confidence,
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningTrace:
    trace_id: int
    states: list[SpatialState]
    predicted_label: str
    raw_response: str = ""
    option_distribution: dict[str, float] = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)
    generation_seed: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], trace_id: int = 0, n_choices: int | None = None) -> "ReasoningTrace":
        raw_states = payload.get("states", payload.get("steps", [])) or []
        states = [SpatialState.from_dict(item, step_idx=i) for i, item in enumerate(raw_states) if isinstance(item, dict)]
        for idx, state in enumerate(states):
            state.step_idx = idx
            state.depends_on = sorted(parent for parent in state.depends_on if 0 <= parent < idx)
        label = normalize_label(payload.get("final_answer", payload.get("predicted_label", "")), n_choices)
        distribution = payload.get("option_distribution", {}) or {}
        return cls(
            trace_id=int(payload.get("trace_id", trace_id)),
            states=states,
            predicted_label=label,
            raw_response=str(payload.get("raw_response", "")),
            option_distribution={str(k): float(v) for k, v in distribution.items()},
            parse_warnings=[str(item) for item in payload.get("parse_warnings", [])],
            generation_seed=payload.get("generation_seed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "states": [state.to_dict() for state in self.states],
            "predicted_label": self.predicted_label,
            "final_answer": self.predicted_label,
            "raw_response": self.raw_response,
            "option_distribution": self.option_distribution,
            "parse_warnings": self.parse_warnings,
            "generation_seed": self.generation_seed,
        }

    def state_text(self) -> str:
        lines = []
        for state in self.states:
            lines.append(
                f"[{state.step_idx}:{state.step_type}] {state.description} | parents={state.depends_on} | "
                f"frame={state.reference_frame} | operation={state.operation} | before={state.relations_before} | "
                f"after={state.relations_after} | visibility={state.visibility}"
            )
        return "\n".join(lines)


@dataclass
class SpatialExample:
    example_id: str
    task: str
    question: str
    choices: list[str]
    answer: str
    images: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task": self.task,
            "question": self.question,
            "choices": self.choices,
            "answer": self.answer,
            "num_images": len(self.images),
            "metadata": self.metadata,
        }


@dataclass
class TransitionAssessment:
    step_idx: int
    consistency: float
    symbolic_score: float
    vlm_score: float | None = None
    learned_score: float | None = None
    error_type: str = "none"
    rationale: str = ""
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisResult:
    trace_id: int
    original_distribution: dict[str, float]
    assessments: list[TransitionAssessment]
    interventions: list[dict[str, Any]]
    localized_step: int | None
    localized_error_type: str
    anomaly_score: float
    causal_score: float
    should_repair: bool
    global_consistency: float
    necessity_score: float = 0.0
    sufficiency_score: float = 0.0
    interventional_consistency: float = 0.0
    root_score: float = 0.0
    dependency_graph: dict[str, Any] = field(default_factory=dict)
    descendant_steps: list[int] = field(default_factory=list)
    candidate_roots: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiagnosisResult":
        return cls(
            trace_id=int(payload.get("trace_id", 0)),
            original_distribution={str(k): float(v) for k, v in (payload.get("original_distribution", {}) or {}).items()},
            assessments=[TransitionAssessment(**item) for item in payload.get("assessments", [])],
            interventions=list(payload.get("interventions", [])),
            localized_step=payload.get("localized_step"),
            localized_error_type=str(payload.get("localized_error_type", "none")),
            anomaly_score=float(payload.get("anomaly_score", 0.0)),
            causal_score=float(payload.get("causal_score", payload.get("root_score", 0.0))),
            should_repair=bool(payload.get("should_repair", False)),
            global_consistency=float(payload.get("global_consistency", 0.0)),
            necessity_score=float(payload.get("necessity_score", 0.0)),
            sufficiency_score=float(payload.get("sufficiency_score", 0.0)),
            interventional_consistency=float(payload.get("interventional_consistency", 0.0)),
            root_score=float(payload.get("root_score", payload.get("causal_score", 0.0))),
            dependency_graph=dict(payload.get("dependency_graph", {}) or {}),
            descendant_steps=[int(item) for item in payload.get("descendant_steps", [])],
            candidate_roots=list(payload.get("candidate_roots", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "original_distribution": self.original_distribution,
            "assessments": [item.to_dict() for item in self.assessments],
            "interventions": self.interventions,
            "localized_step": self.localized_step,
            "localized_error_type": self.localized_error_type,
            "anomaly_score": self.anomaly_score,
            "causal_score": self.causal_score,
            "should_repair": self.should_repair,
            "global_consistency": self.global_consistency,
            "necessity_score": self.necessity_score,
            "sufficiency_score": self.sufficiency_score,
            "interventional_consistency": self.interventional_consistency,
            "root_score": self.root_score,
            "dependency_graph": self.dependency_graph,
            "descendant_steps": self.descendant_steps,
            "candidate_roots": self.candidate_roots,
        }
