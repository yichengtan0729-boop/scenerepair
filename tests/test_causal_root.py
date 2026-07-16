from scenerepair.calibration import interventional_agreement, lower_confidence_bound, weighted_geometric_mean
from scenerepair.causal_graph import build_causal_graph
from scenerepair.schemas import ReasoningTrace, SpatialState


def _state(idx, kind, objects=None, depends=None):
    return SpatialState(
        step_idx=idx,
        step_type=kind,
        description=kind,
        objects=objects or [],
        depends_on=depends or [],
    )


def test_dependency_graph_prefers_explicit_and_tracks_descendants():
    trace = ReasoningTrace(
        trace_id=0,
        states=[
            _state(0, "observation", ["chair"]),
            _state(1, "correspondence", ["chair"], [0]),
            _state(2, "reference_frame", ["chair"], [1]),
            _state(3, "counterfactual_operation", ["chair"], [2]),
            _state(4, "updated_state", ["chair"], [3]),
            _state(5, "answer", [], [4]),
        ],
        predicted_label="A",
    )
    graph = build_causal_graph(trace)
    assert graph.parents[3] == [2]
    assert graph.descendants(2) == [3, 4, 5]
    assert graph.ancestors(5) == [0, 1, 2, 3, 4]


def test_calibrated_statistics_are_monotonic():
    low = lower_confidence_bound([0.01, 0.02, 0.03], confidence=0.90)
    high = lower_confidence_bound([0.10, 0.11, 0.12], confidence=0.90)
    assert high["lcb"] > low["lcb"]
    score = weighted_geometric_mean(
        {"anomaly": 0.8, "necessity": 0.7, "sufficiency": 0.6, "ict": 0.9},
        {"anomaly": 1.0, "necessity": 1.0, "sufficiency": 1.0, "ict": 0.5},
    )
    assert 0.6 < score < 0.9
    assert interventional_agreement([0.2, 0.21, 0.19], ["A", "A", "A"]) > 0.9
