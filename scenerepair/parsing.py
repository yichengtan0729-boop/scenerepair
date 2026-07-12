from __future__ import annotations

import ast
import json
import re
from typing import Any

from .schemas import ReasoningTrace, SpatialState, StepType
from .utils import normalize_label


def extract_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    errors: list[str] = []
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                payload = loader(candidate.strip())
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                errors.append(str(exc))
    raise ValueError(f"Could not parse JSON object: {' | '.join(errors[-3:])}")


def fallback_trace_from_text(text: str, trace_id: int, n_choices: int) -> ReasoningTrace:
    label = normalize_label(text, n_choices)
    sentences = [item.strip() for item in re.split(r"(?:\n+|(?<=[.!?])\s+)", text) if item.strip()]
    default_types = [
        StepType.OBSERVATION.value,
        StepType.CORRESPONDENCE.value,
        StepType.REFERENCE_FRAME.value,
        StepType.INITIAL_STATE.value,
        StepType.COUNTERFACTUAL_OPERATION.value,
        StepType.UPDATED_STATE.value,
        StepType.VISIBILITY.value,
        StepType.ANSWER.value,
    ]
    states = [
        SpatialState(
            step_idx=idx,
            step_type=default_types[min(idx, len(default_types) - 1)],
            description=sentence,
        )
        for idx, sentence in enumerate(sentences[:8])
    ]
    if not states:
        states = [SpatialState(step_idx=0, step_type=StepType.OTHER.value, description="Unparsed response")]
    return ReasoningTrace(
        trace_id=trace_id,
        states=states,
        predicted_label=label,
        raw_response=text,
        parse_warnings=["fallback_text_parser"],
    )


def parse_reasoning_trace(text: str, trace_id: int, n_choices: int) -> ReasoningTrace:
    try:
        payload = extract_json_object(text)
        payload["raw_response"] = text
        payload["trace_id"] = trace_id
        trace = ReasoningTrace.from_dict(payload, trace_id=trace_id, n_choices=n_choices)
        if not trace.states:
            raise ValueError("JSON contained no states")
        if not trace.predicted_label:
            trace.predicted_label = normalize_label(text, n_choices)
            trace.parse_warnings.append("answer_recovered_from_raw_text")
        return trace
    except Exception as exc:
        trace = fallback_trace_from_text(text, trace_id, n_choices)
        trace.parse_warnings.append(f"json_parse_error:{exc}")
        return trace


def parse_score_payload(text: str) -> dict[str, Any]:
    try:
        payload = extract_json_object(text)
    except Exception:
        match = re.search(r"(?:score|consistency)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", text, re.I)
        score = float(match.group(1)) if match else 0.5
        return {"score": score, "error_type": "unparsed", "rationale": text[:500]}
    try:
        score = float(payload.get("score", payload.get("consistency", 0.5)))
    except Exception:
        score = 0.5
    payload["score"] = min(1.0, max(0.0, score))
    payload["error_type"] = str(payload.get("error_type", "none"))
    payload["rationale"] = str(payload.get("rationale", payload.get("reason", "")))
    return payload
