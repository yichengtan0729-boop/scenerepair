# SceneRepair

**Label-Free Necessity-Sufficiency Causal-Root Localization and Minimal Repair for Counterfactual Spatial Reasoning**

SceneRepair is a method-oriented experiment codebase for frozen vision-language models. It targets multi-view counterfactual spatial reasoning, with MindEdit-Bench L4 (`single_view_spatial_editing`) and L5 (`cross_view_visibility`) as the primary tasks. The benchmark answer is not used by generation, diagnosis, intervention, candidate ranking, or repair; it is consumed only after predictions are saved by the evaluation module.

## Final method

For each example, SceneRepair performs:

1. typed scene-state trajectory generation with explicit prerequisite indices;
2. construction of a spatial causal dependency graph;
3. label-free symbolic, VLM, and optional learned transition verification;
4. paired necessity and sufficiency interventions over reference frames, operations, relations, object identity, visibility, and visual evidence;
5. interventional-consistency testing across semantically equivalent variants;
6. ancestor-minimal causal-root localization rather than downstream symptom selection;
7. descendant-closed minimal suffix regeneration;
8. calibrated label-free acceptance using a one-sided lower confidence bound;
9. consistency-weighted aggregation over multiple traces.

For a state node `t`, the implementation computes:

```text
anomaly_t     = 1 - transition_consistency_t
necessity_t   = answer_shift + descendant_consistency_drop
sufficiency_t = descendant_consistency_gain + local_recovery + confidence_gain
ICT_t         = agreement across equivalent intervention realizations
root_t        = weighted_geometric_mean(anomaly_t, necessity_t, sufficiency_t, ICT_t)
```

A node must pass all component thresholds. Among nodes whose root evidence is close to the strongest valid node, SceneRepair selects an ancestor-minimal node. This prevents a large downstream symptom from replacing the upstream cause.

A candidate repair is accepted only when:

```text
LCB(candidate_objective - original_objective) > abstain_margin
and consistency_gain >= minimum_consistency_gain
```

The unchanged trace therefore remains the default no-op decision.

## Included components

- MindEdit-Bench Hugging Face loader and generic multi-image JSONL loader;
- local Qwen2.5-VL and Qwen3-VL backends;
- OpenAI-compatible/vLLM multimodal backend;
- deterministic mock backend for full pipeline validation;
- typed spatial states with explicit dependency edges;
- symbolic, VLM, and learned transition critics;
- six structured intervention families and equivalent variants;
- necessity-sufficiency causal-root localization;
- descendant-closed suffix repair with calibrated abstention;
- direct, CoT, self-consistency, full-reflection, and structured-original baselines;
- automatic lightweight critic training from generated corruptions;
- bootstrap confidence intervals, McNemar testing, no-harm and repair metrics;
- synthetic known-corruption localization evaluation;
- multi-seed, ablation, visualization, and aggregation scripts;
- unit tests for label-free invariance, dependency graphs, calibration, and end-to-end execution.

## Installation

```bash
conda create -n scenerepair python=3.10 -y
conda activate scenerepair
pip install -e ".[hf,dev]"
```

For 4-bit inference:

```bash
pip install -e ".[hf,quant,dev]"
```

For an OpenAI-compatible or vLLM endpoint:

```bash
pip install -e ".[api,dev]"
```

## Verify without a GPU

```bash
python scripts/make_toy_data.py
python -m scenerepair.cli --config configs/mock_smoke.yaml --task all
python -m pytest -q
```

## Main causal-root experiment

```bash
python -m scenerepair.cli \
  --config configs/mindedit_rootrepair.yaml \
  --task all

python -m scenerepair.cli \
  --config configs/mindedit_rootrepair.yaml \
  --task synthetic_localization
```

A five-example real-model smoke run:

```bash
python -m scenerepair.cli \
  --config configs/mindedit_rootrepair.yaml \
  --task all \
  --limit 5 \
  --output-dir outputs/rootrepair_smoke5
```

## Resumable stages

```bash
python -m scenerepair.cli --config CONFIG --task sample
python -m scenerepair.cli --config CONFIG --task diagnose
python -m scenerepair.cli --config CONFIG --task repair
python -m scenerepair.cli --config CONFIG --task baselines
python -m scenerepair.cli --config CONFIG --task evaluate
python -m scenerepair.cli --config CONFIG --task plot
```

All per-example files are written atomically and reused unless `--overwrite` is supplied.

## Ablation overrides

```bash
# Anomaly-only approximation
python -m scenerepair.cli --config configs/mindedit_rootrepair.yaml --task all \
  --set critic.necessity_threshold=0.0 \
  --set critic.sufficiency_threshold=0.0 \
  --set critic.interventional_consistency_root_weight=0.0

# Sequence-only / weak graph ablation is implemented by removing depends_on
# from saved traces and rerunning diagnosis with --overwrite.

# Fixed-margin acceptance
python -m scenerepair.cli --config configs/mindedit_rootrepair.yaml --task all \
  --set repair.acceptance_repeats=1

# Full regeneration
python -m scenerepair.cli --config configs/mindedit_rootrepair.yaml --task all \
  --set repair.preserve_verified_prefix=false
```

## Output structure

```text
OUTPUT_DIR/
  examples/
  traces/
  diagnoses/
  repairs/
  baselines/
  synthetic/
  checkpoints/
  tables/
    per_example_methods.csv
    per_example_repair.csv
    per_trace_root_localization.csv
    summary_methods.csv
    summary_by_task.csv
    summary.json
    synthetic_localization.json
  figures/
```

Reported statistics include overall and task-level accuracy, 95% bootstrap confidence intervals, wrong-to-correct, correct-to-wrong, no-harm rate, repair attempt/apply rate, consistency gain, necessity, sufficiency, root score, acceptance LCB, exact paired McNemar p-value, and synthetic localization accuracy.

## Local JSONL format

```json
{
  "id": "example_001",
  "task": "counterfactual_spatial_editing",
  "question": "...",
  "choices": ["A. ...", "B. ..."],
  "answer": "B",
  "images": ["view1.jpg", "view2.jpg", "view3.jpg"]
}
```

Image paths are resolved relative to the JSONL file. See `docs/METHOD.md` and `docs/EXPERIMENTS.md` for the formal method and recommended paper protocol.
