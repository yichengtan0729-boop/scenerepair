from __future__ import annotations

import json

from .schemas import ReasoningTrace, SpatialExample, SpatialState
from .utils import labels_for_choices


def format_choices(choices: list[str]) -> str:
    labels = labels_for_choices(len(choices))
    lines: list[str] = []
    for label, choice in zip(labels, choices):
        text = str(choice).strip()
        if text.upper().startswith(f"{label}.") or text.upper().startswith(f"{label})"):
            lines.append(text)
        else:
            lines.append(f"{label}. {text}")
    return "\n".join(lines)


def structured_reasoning_prompt(example: SpatialExample, max_steps: int = 8) -> str:
    return f"""You are solving a multi-view counterfactual spatial reasoning problem. Use the images as evidence, but do not assume the edited scene is directly visible.

Question:
{example.question}

Choices:
{format_choices(example.choices)}

Construct a typed scene-state trajectory. Explicitly distinguish camera-centric, object-centric, and world-centric reference frames. Apply every hypothetical translation or rotation before answering. Use at most {max_steps} states.

Return ONLY one valid JSON object with this schema:
{{
  "states": [
    {{
      "step_idx": 0,
      "step_type": "observation|correspondence|reference_frame|initial_state|counterfactual_operation|updated_state|visibility|answer",
      "description": "concise factual state or transition",
      "objects": ["object names"],
      "reference_frame": "camera_1|camera_2|camera_3|object:<name>|world|unknown",
      "operation": "none or explicit translation/rotation",
      "relations_before": ["subject relation object"],
      "relations_after": ["subject relation object"],
      "visibility": ["object visible/occluded/absent in view k"],
      "evidence_views": [1,2,3],
      "confidence": 0.0
    }}
  ],
  "final_answer": "one answer letter"
}}
Do not include the ground-truth answer, external coordinates, markdown, or prose outside JSON."""


def direct_prompt(example: SpatialExample) -> str:
    return f"""Inspect all supplied views and answer the spatial multiple-choice question.
Question: {example.question}
Choices:
{format_choices(example.choices)}
Return only the answer letter."""


def cot_prompt(example: SpatialExample) -> str:
    return f"""Inspect all supplied views. Reason step by step about object correspondence, reference frame, the hypothetical operation, and the resulting relation or visibility.
Question: {example.question}
Choices:
{format_choices(example.choices)}
End with exactly: Answer: <letter>."""


def option_scoring_prompt(example: SpatialExample, trace: ReasoningTrace) -> str:
    return f"""Use the images and the proposed structured reasoning state below to answer the question. Treat the state as a hypothesis, not as guaranteed truth.
Question: {example.question}
Choices:
{format_choices(example.choices)}
Reasoning state:
{trace.state_text()}
Return only one answer letter."""


def transition_judge_prompt(
    example: SpatialExample,
    previous: SpatialState | None,
    current: SpatialState,
) -> str:
    prev = json.dumps(previous.to_dict() if previous else {"scene": "initial visual evidence"}, ensure_ascii=False)
    cur = json.dumps(current.to_dict(), ensure_ascii=False)
    return f"""Act as a label-free spatial transition verifier. You must not infer or use the benchmark answer. Judge whether the current state follows from the previous state, the visible views, the stated reference frame, and any hypothetical operation.
Question: {example.question}
Previous state: {prev}
Current state: {cur}
Check object identity across views, reference-frame consistency, operation direction and magnitude, relation update, visibility or occlusion, and unsupported claims.
Return ONLY JSON:
{{"score": 0.0, "error_type": "none|object_identity|reference_frame|operation|relation_update|visibility|unsupported|format", "rationale": "short reason", "components": {{"view":0.0,"frame":0.0,"operation":0.0,"state":0.0}}}}
A score of 1 means fully consistent and 0 means clearly inconsistent."""


def global_judge_prompt(example: SpatialExample, trace: ReasoningTrace) -> str:
    return f"""Act as a label-free verifier. Do not use or guess the benchmark label. Rate whether this entire spatial reasoning trajectory is internally coherent and grounded in the supplied images.
Question: {example.question}
Trajectory:
{trace.state_text()}
Return only JSON: {{"score":0.0,"error_type":"none or dominant error","rationale":"short reason"}}."""


def repair_prompt(
    example: SpatialExample,
    trace: ReasoningTrace,
    localized_step: int,
    diagnosis: dict,
    num_states: int,
) -> str:
    prefix = [state.to_dict() for state in trace.states[:localized_step]]
    faulty = trace.states[localized_step].to_dict() if localized_step < len(trace.states) else {}
    suffix = [state.to_dict() for state in trace.states[localized_step:]]
    return f"""Repair a counterfactual spatial reasoning trajectory without access to the correct answer. Preserve the verified prefix exactly. Recompute the localized transition and all downstream states from the images, question, and explicit reference frame.

Question: {example.question}
Choices:
{format_choices(example.choices)}
Verified prefix JSON: {json.dumps(prefix, ensure_ascii=False)}
Localized step: {localized_step}
Faulty state: {json.dumps(faulty, ensure_ascii=False)}
Original suffix: {json.dumps(suffix, ensure_ascii=False)}
Diagnosis: {json.dumps(diagnosis, ensure_ascii=False)}

Return ONLY JSON with at most {num_states} newly generated suffix states:
{{"states":[same state schema],"final_answer":"letter"}}
The returned states must start at step_idx={localized_step}. Do not repeat the verified prefix. Prefer the smallest correction that restores view, frame, operation, relation, and visibility consistency."""


def reflection_prompt(example: SpatialExample, original_response: str) -> str:
    return f"""Review the proposed answer to this multi-view spatial problem. Locate any object correspondence, reference-frame, transformation, relation, or visibility error, then solve again.
Question: {example.question}
Choices:
{format_choices(example.choices)}
Original response:
{original_response}
End with exactly: Answer: <letter>."""
