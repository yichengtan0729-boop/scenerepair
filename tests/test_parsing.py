from scenerepair.parsing import parse_reasoning_trace


def test_parse_structured_trace():
    text = '''{"states":[{"step_idx":0,"step_type":"observation","description":"see objects","objects":["a","b"],"reference_frame":"camera_1","operation":"none","relations_before":["a left of b"],"relations_after":[],"visibility":[],"evidence_views":[1],"confidence":0.8}],"final_answer":"B"}'''
    trace = parse_reasoning_trace(text, 0, 4)
    assert trace.predicted_label == "B"
    assert len(trace.states) == 1
    assert trace.states[0].step_type == "observation"


def test_fallback_parser():
    trace = parse_reasoning_trace("First inspect the image. Answer: C", 0, 4)
    assert trace.predicted_label == "C"
    assert trace.states
    assert trace.parse_warnings
