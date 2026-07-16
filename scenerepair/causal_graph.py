from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import ReasoningTrace, SpatialState, StepType


_TYPE_PARENTS: dict[str, tuple[str, ...]] = {
    StepType.OBSERVATION.value: (),
    StepType.CORRESPONDENCE.value: (StepType.OBSERVATION.value,),
    StepType.REFERENCE_FRAME.value: (StepType.CORRESPONDENCE.value, StepType.OBSERVATION.value),
    StepType.INITIAL_STATE.value: (StepType.REFERENCE_FRAME.value, StepType.CORRESPONDENCE.value),
    StepType.COUNTERFACTUAL_OPERATION.value: (StepType.INITIAL_STATE.value, StepType.REFERENCE_FRAME.value),
    StepType.UPDATED_STATE.value: (StepType.COUNTERFACTUAL_OPERATION.value, StepType.INITIAL_STATE.value),
    StepType.VISIBILITY.value: (StepType.UPDATED_STATE.value, StepType.REFERENCE_FRAME.value),
    StepType.ANSWER.value: (
        StepType.VISIBILITY.value,
        StepType.UPDATED_STATE.value,
        StepType.COUNTERFACTUAL_OPERATION.value,
        StepType.REFERENCE_FRAME.value,
    ),
    StepType.OTHER.value: (),
}


@dataclass
class CausalDependencyGraph:
    parents: dict[int, list[int]]
    children: dict[int, list[int]]
    step_types: dict[int, str]

    def ancestors(self, node: int) -> list[int]:
        seen: set[int] = set()
        stack = list(self.parents.get(node, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.parents.get(current, []))
        return sorted(seen)

    def descendants(self, node: int) -> list[int]:
        seen: set[int] = set()
        stack = list(self.children.get(node, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.children.get(current, []))
        return sorted(seen)

    def depth(self, node: int) -> int:
        parents = self.parents.get(node, [])
        if not parents:
            return 0
        return 1 + max(self.depth(parent) for parent in parents)

    def to_dict(self) -> dict:
        return {
            "parents": {str(key): value for key, value in self.parents.items()},
            "children": {str(key): value for key, value in self.children.items()},
            "step_types": {str(key): value for key, value in self.step_types.items()},
            "depth": {str(key): self.depth(key) for key in self.step_types},
        }


def _nearest_previous(states: list[SpatialState], idx: int, accepted_types: Iterable[str]) -> list[int]:
    accepted = set(accepted_types)
    matches = [j for j in range(idx - 1, -1, -1) if states[j].step_type in accepted]
    return matches[:1]


def _object_parent(states: list[SpatialState], idx: int) -> list[int]:
    objects = set(states[idx].objects)
    if not objects:
        return []
    for j in range(idx - 1, -1, -1):
        if objects.intersection(states[j].objects):
            return [j]
    return []


def build_causal_graph(trace: ReasoningTrace) -> CausalDependencyGraph:
    states = trace.states
    parents: dict[int, list[int]] = {}
    step_types = {idx: state.step_type for idx, state in enumerate(states)}
    for idx, state in enumerate(states):
        explicit = sorted({int(parent) for parent in state.depends_on if 0 <= int(parent) < idx})
        inferred: list[int] = []
        if not explicit:
            inferred.extend(_nearest_previous(states, idx, _TYPE_PARENTS.get(state.step_type, ())))
            inferred.extend(_object_parent(states, idx))
            if not inferred and idx > 0:
                inferred.append(idx - 1)
        if state.step_type == StepType.ANSWER.value:
            terminal_types = {
                StepType.VISIBILITY.value,
                StepType.UPDATED_STATE.value,
                StepType.COUNTERFACTUAL_OPERATION.value,
            }
            inferred.extend(j for j in range(idx) if states[j].step_type in terminal_types)
        parents[idx] = sorted(set(explicit or inferred))
    children: dict[int, list[int]] = {idx: [] for idx in range(len(states))}
    for child, node_parents in parents.items():
        for parent in node_parents:
            children.setdefault(parent, []).append(child)
    for value in children.values():
        value.sort()
    return CausalDependencyGraph(parents=parents, children=children, step_types=step_types)
