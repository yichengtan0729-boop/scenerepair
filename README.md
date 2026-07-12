# SceneRepair

**Label-Free Causal Localization and Minimal Suffix Repair for Counterfactual Spatial Reasoning**

SceneRepair is a complete method-oriented experiment codebase for frozen vision-language models. It targets multi-view counterfactual spatial reasoning, with MindEdit-Bench L4 (`single_view_spatial_editing`) and L5 (`cross_view_visibility`) as the primary evaluation tasks.

The method does not use the ground-truth answer during inference, diagnosis, intervention, candidate selection, or repair. Labels are consumed only by the evaluation module after predictions have been written.

## Method

For every example, SceneRepair performs:

1. typed scene-state trajectory generation;
2. label-free symbolic, VLM, and optional learned transition criticism;
3. causal interventions on reference frame, operation, relation, object identity, visibility, and visual evidence;
4. earliest consequential-error localization;
5. minimal suffix regeneration while preserving the verified prefix;
6. no-harm candidate selection with an explicit abstention margin;
7. consistency-weighted aggregation over multiple traces.

For transition `t`:

```text
anomaly_t = 1 - consistency_t
causal_t  = 0.7 * JS(p_original, p_intervened)
            + 0.3 * max(0, consistency_intervened - consistency_t)
```

The earliest transition passing both anomaly and causal thresholds is repaired. The unchanged trace remains an implicit no-op candidate, so a repair is applied only when it improves the configured objective by more than the abstention margin.

## Included components

- MindEdit-Bench Hugging Face loader for L4/L5 and all other published subsets;
- generic local multi-image JSONL loader;
- local Qwen2.5-VL and Qwen3-VL backends;
- OpenAI-compatible/vLLM multimodal backend;
- deterministic mock backend for full pipeline validation;
- typed state schema and robust JSON fallback parser;
- symbolic, VLM, and learned transition critics;
- six causal intervention families;
- earliest-error localization;
- minimal suffix repair and full-regeneration ablation;
- direct, CoT, self-consistency, full-reflection, and structured-original baselines;
- automatic transition-critic training from generated corruptions;
- bootstrap confidence intervals, paired McNemar testing, no-harm and repair metrics;
- synthetic known-corruption localization evaluation;
- multi-seed, ablation, visualization, and aggregation scripts;
- unit tests including a label-free invariance test and complete end-to-end pipeline test.

## Installation

### Local Qwen-VL

```bash
conda create -n scenerepair python=3.10 -y
conda activate scenerepair
pip install -e ".[hf,dev]"
```

For 4-bit inference:

```bash
pip install -e ".[hf,quant,dev]"
```

### OpenAI-compatible or vLLM endpoint

```bash
pip install -e ".[api,dev]"
```

## Verify without a GPU

```bash
python scripts/make_toy_data.py
python -m scenerepair.cli --config configs/mock_smoke.yaml --task all
python -m pytest -q
```

The mock backend uses the same data, parsing, diagnosis, intervention, repair, aggregation, and evaluation paths as a real VLM.

## Main experiments

Qwen2.5-VL-7B:

```bash
python -m scenerepair.cli \
  --config configs/mindedit_qwen25vl7b.yaml \
  --task all
```

Qwen3-VL-8B:

```bash
python -m scenerepair.cli \
  --config configs/mindedit_qwen3vl8b.yaml \
  --task all
```

vLLM/OpenAI-compatible server:

```bash
python -m scenerepair.cli \
  --config configs/mindedit_vllm_api.yaml \
  --task all
```

Complete wrapper:

```bash
bash scripts/run_full_mindedit.sh configs/mindedit_qwen25vl7b.yaml
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

Subset run:

```bash
python -m scenerepair.cli \
  --config configs/mindedit_qwen25vl7b.yaml \
  --task all \
  --start-index 0 \
  --end-index 20 \
  --output-dir outputs/debug_20
```

Dotted config overrides:

```bash
python -m scenerepair.cli \
  --config configs/mindedit_qwen25vl7b.yaml \
  --task all \
  --set model.load_in_4bit=true \
  --set run.num_traces=5 \
  --set repair.num_candidates=5
```

## Localization and learned critic

Synthetic known-corruption localization:

```bash
python -m scenerepair.cli --config CONFIG --task synthetic_localization
```

Train the optional lightweight transition critic after collecting diagnosis files:

```bash
python -m scenerepair.cli --config CONFIG --task diagnose
python -m scenerepair.cli --config CONFIG --task train_critic
```

Then set:

```yaml
critic:
  mode: all
  learned_model_path: outputs/your_run/checkpoints/transition_critic.joblib
  symbolic_weight: 0.25
  vlm_weight: 0.50
  learned_weight: 0.25
```

## Multi-seed and ablations

```bash
bash scripts/run_multiseed.sh configs/mindedit_qwen25vl7b.yaml
bash scripts/run_ablations.sh configs/mindedit_qwen25vl7b.yaml
```

The default ablation script includes symbolic-only diagnosis, no causal intervention, full regeneration, and no abstention. Any intervention, threshold, critic weight, trace count, candidate count, or objective term can also be changed through `--set`.

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
    summary_methods.csv
    summary_by_task.csv
    summary.json
    synthetic_localization.json
  figures/
```

Reported statistics include overall and task-level accuracy, 95% bootstrap confidence intervals, wrong-to-correct, correct-to-wrong, no-harm rate, repair attempt/apply rate, consistency gain, exact paired McNemar p-value, and synthetic localization accuracy.

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

Image paths are resolved relative to the JSONL file. The answer is retained for evaluation only and is never included in prompts, diagnosis, intervention, or repair.

See `docs/METHOD.md` and `docs/EXPERIMENTS.md` for the formal method and recommended paper protocol.
