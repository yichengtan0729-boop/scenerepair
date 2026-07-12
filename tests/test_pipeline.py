import json
from pathlib import Path

from PIL import Image

from scenerepair.config import (
    CriticConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RepairConfig,
    RunConfig,
)
from scenerepair.pipeline import SceneRepairPipeline


def test_complete_mock_pipeline(tmp_path: Path):
    image_paths = []
    for idx in range(3):
        path = tmp_path / f"view{idx}.png"
        Image.new("RGB", (64, 64), "white").save(path)
        image_paths.append(path.name)
    data_path = tmp_path / "toy.jsonl"
    row = {
        "id": "toy",
        "task": "single_view_spatial_editing",
        "question": "If A moves right of B, where is A?",
        "choices": ["A. left", "B. right", "C. above", "D. below"],
        "answer": "B",
        "images": image_paths,
    }
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cfg = ExperimentConfig(
        data=DataConfig(source="jsonl", jsonl_path=str(data_path)),
        model=ModelConfig(backend="mock", model_name="mock"),
        critic=CriticConfig(
            mode="symbolic",
            symbolic_weight=1.0,
            vlm_weight=0.0,
            transition_threshold=0.99,
            causal_threshold=0.0,
            global_threshold=0.99,
            intervention_top_k=2,
            intervention_types=["frame_swap", "operation_inverse"],
        ),
        repair=RepairConfig(enabled=True, num_candidates=2, abstain_margin=-1.0),
        run=RunConfig(
            output_dir=str(tmp_path / "outputs"),
            num_traces=2,
            max_reasoning_steps=6,
            overwrite=True,
            run_baselines=True,
            fail_fast=True,
        ),
    )
    result = SceneRepairPipeline(cfg).run("all")
    assert (tmp_path / "outputs" / "traces" / "toy.traces.json").exists()
    assert (tmp_path / "outputs" / "diagnoses" / "toy.diagnosis.json").exists()
    assert (tmp_path / "outputs" / "repairs" / "toy.repair.json").exists()
    assert result["method_summaries"]
    repair_payload = json.loads((tmp_path / "outputs" / "repairs" / "toy.repair.json").read_text())
    assert repair_payload["repair_attempted_count"] > 0
