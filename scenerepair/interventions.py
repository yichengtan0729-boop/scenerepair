from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from .schemas import ReasoningTrace, SpatialState


@dataclass
class Intervention:
    name: str
    description: str
    state: SpatialState
    dropped_view: int | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state.to_dict(),
            "dropped_view": self.dropped_view,
        }


INVERSE_OPERATIONS = [
    (r"counter[- ]?clockwise", "__CLOCKWISE__"),
    (r"clockwise", "counterclockwise"),
    (r"__CLOCKWISE__", "clockwise"),
    (r"move\s+left", "__MOVE_RIGHT__"),
    (r"move\s+right", "move left"),
    (r"__MOVE_RIGHT__", "move right"),
    (r"translate\s+left", "__TRANSLATE_RIGHT__"),
    (r"translate\s+right", "translate left"),
    (r"__TRANSLATE_RIGHT__", "translate right"),
    (r"forward", "__BACKWARD__"),
    (r"backward", "forward"),
    (r"__BACKWARD__", "backward"),
    (r"toward", "__AWAY__"),
    (r"away", "toward"),
    (r"__AWAY__", "away"),
]

RELATION_PAIRS = [
    ("left of", "right of"),
    ("in front of", "behind"),
    ("above", "below"),
    ("near", "far"),
    ("visible", "occluded"),
    ("present", "absent"),
]


def _swap_phrases(text: str, pairs: list[tuple[str, str]]) -> str:
    result = text
    for idx, (left, right) in enumerate(pairs):
        left_token = f"__LEFT_{idx}__"
        right_token = f"__RIGHT_{idx}__"
        result = re.sub(re.escape(left), left_token, result, flags=re.IGNORECASE)
        result = re.sub(re.escape(right), right_token, result, flags=re.IGNORECASE)
        result = result.replace(left_token, right).replace(right_token, left)
    return result


def invert_operation(operation: str) -> str:
    result = operation
    for pattern, replacement in INVERSE_OPERATIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def flip_relations(relations: list[str]) -> list[str]:
    return [_swap_phrases(item, RELATION_PAIRS) for item in relations]


def frame_swap(state: SpatialState) -> Intervention:
    modified = copy.deepcopy(state)
    frame = modified.reference_frame.lower()
    if frame.startswith("camera_1"):
        modified.reference_frame = "camera_2"
    elif frame.startswith("camera_2"):
        modified.reference_frame = "camera_1"
    elif frame.startswith("camera_3"):
        modified.reference_frame = "camera_1"
    elif frame.startswith("object:"):
        modified.reference_frame = "world"
    elif frame == "world":
        modified.reference_frame = "camera_1"
    else:
        modified.reference_frame = "world"
    modified.metadata["intervention"] = "frame_swap"
    return Intervention("frame_swap", f"Swap reference frame from {state.reference_frame} to {modified.reference_frame}", modified)


def operation_inverse(state: SpatialState) -> Intervention:
    modified = copy.deepcopy(state)
    modified.operation = invert_operation(modified.operation)
    modified.description = invert_operation(modified.description)
    modified.metadata["intervention"] = "operation_inverse"
    return Intervention("operation_inverse", "Invert translation or rotation direction", modified)


def relation_flip(state: SpatialState) -> Intervention:
    modified = copy.deepcopy(state)
    modified.relations_before = flip_relations(modified.relations_before)
    modified.relations_after = flip_relations(modified.relations_after)
    modified.description = _swap_phrases(modified.description, RELATION_PAIRS)
    modified.metadata["intervention"] = "relation_flip"
    return Intervention("relation_flip", "Flip directional, distance, or visibility relations", modified)


def object_swap(state: SpatialState) -> Intervention | None:
    if len(state.objects) < 2:
        return None
    modified = copy.deepcopy(state)
    first, second = modified.objects[0], modified.objects[1]
    modified.objects[0], modified.objects[1] = second, first
    for field_name in ("description", "operation"):
        text = getattr(modified, field_name)
        text = text.replace(first, "__FIRST_OBJECT__").replace(second, first).replace("__FIRST_OBJECT__", second)
        setattr(modified, field_name, text)
    modified.relations_before = [
        item.replace(first, "__FIRST_OBJECT__").replace(second, first).replace("__FIRST_OBJECT__", second)
        for item in modified.relations_before
    ]
    modified.relations_after = [
        item.replace(first, "__FIRST_OBJECT__").replace(second, first).replace("__FIRST_OBJECT__", second)
        for item in modified.relations_after
    ]
    modified.metadata["intervention"] = "object_swap"
    return Intervention("object_swap", f"Swap object identities {first} and {second}", modified)


def visibility_flip(state: SpatialState) -> Intervention:
    modified = copy.deepcopy(state)
    modified.visibility = flip_relations(modified.visibility)
    modified.description = _swap_phrases(modified.description, [("visible", "occluded"), ("present", "absent")])
    modified.metadata["intervention"] = "visibility_flip"
    return Intervention("visibility_flip", "Flip visibility or occlusion state", modified)


def view_dropout(state: SpatialState, num_views: int) -> Intervention | None:
    if num_views <= 1:
        return None
    candidates = state.evidence_views or list(range(1, num_views + 1))
    dropped = candidates[-1]
    modified = copy.deepcopy(state)
    modified.evidence_views = [idx for idx in modified.evidence_views if idx != dropped]
    modified.metadata["intervention"] = "view_dropout"
    return Intervention("view_dropout", f"Remove visual evidence from view {dropped}", modified, dropped_view=dropped)


def generate_interventions(
    state: SpatialState,
    allowed: list[str],
    num_views: int,
) -> list[Intervention]:
    output: list[Intervention] = []
    for name in allowed:
        intervention: Intervention | None
        if name == "frame_swap":
            intervention = frame_swap(state)
        elif name == "operation_inverse":
            intervention = operation_inverse(state)
        elif name == "relation_flip":
            intervention = relation_flip(state)
        elif name == "object_swap":
            intervention = object_swap(state)
        elif name == "visibility_flip":
            intervention = visibility_flip(state)
        elif name == "view_dropout":
            intervention = view_dropout(state, num_views)
        else:
            continue
        if intervention is not None:
            output.append(intervention)
    return output


def apply_intervention(trace: ReasoningTrace, step_idx: int, intervention: Intervention) -> ReasoningTrace:
    modified = copy.deepcopy(trace)
    modified.states[step_idx] = copy.deepcopy(intervention.state)
    modified.states[step_idx].step_idx = step_idx
    return modified
