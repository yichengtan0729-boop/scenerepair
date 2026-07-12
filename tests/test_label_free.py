from PIL import Image

from scenerepair.config import CriticConfig, ModelConfig
from scenerepair.critics import TransitionCritic
from scenerepair.diagnosis import CausalDiagnoser
from scenerepair.models.factory import build_model
from scenerepair.parsing import parse_reasoning_trace
from scenerepair.prompts import structured_reasoning_prompt
from scenerepair.schemas import SpatialExample


def test_diagnosis_does_not_depend_on_ground_truth():
    model = build_model(ModelConfig(backend="mock", model_name="mock"))
    critic_cfg = CriticConfig(
        mode="symbolic",
        symbolic_weight=1.0,
        vlm_weight=0.0,
        intervention_top_k=2,
        intervention_types=["frame_swap", "operation_inverse"],
        causal_threshold=0.0,
    )
    critic = TransitionCritic(critic_cfg, model)
    diagnoser = CausalDiagnoser(model, critic, critic_cfg)
    image = Image.new("RGB", (32, 32), "white")
    base = SpatialExample("x", "task", "Where is A after moving right?", ["left", "right"], "A", [image])
    raw = model.generate(base.images, structured_reasoning_prompt(base), seed=1)
    trace = parse_reasoning_trace(raw, 0, 2)
    first = diagnoser.diagnose(base, trace).to_dict()
    changed = SpatialExample("x", "task", base.question, base.choices, "B", [image])
    second = diagnoser.diagnose(changed, trace).to_dict()
    assert first["localized_step"] == second["localized_step"]
    assert first["causal_score"] == second["causal_score"]
