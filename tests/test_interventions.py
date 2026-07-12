from scenerepair.interventions import frame_swap, operation_inverse, relation_flip
from scenerepair.schemas import SpatialState


def make_state():
    return SpatialState(
        step_idx=2,
        step_type="counterfactual_operation",
        description="Move chair left and rotate clockwise",
        objects=["chair", "table"],
        reference_frame="camera_1",
        operation="move left; rotate clockwise 90 degrees",
        relations_before=["chair left of table"],
        relations_after=["chair right of table"],
        visibility=["chair visible"],
        evidence_views=[1, 2],
    )


def test_frame_swap():
    result = frame_swap(make_state())
    assert result.state.reference_frame == "camera_2"


def test_operation_inverse():
    result = operation_inverse(make_state())
    assert "right" in result.state.operation.lower()
    assert "counterclockwise" in result.state.operation.lower()


def test_relation_flip():
    result = relation_flip(make_state())
    assert "right of" in result.state.relations_before[0]
