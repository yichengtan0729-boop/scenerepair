# Recommended paper experiment protocol

## Main task

Use MindEdit-Bench L4 (`single_view_spatial_editing`) and L5 (`cross_view_visibility`) as the main table. Report pooled results and both task rows separately because their answer spaces and failure modes differ.

## Models

Recommended minimum:

1. Qwen2.5-VL-7B-Instruct;
2. Qwen3-VL-8B-Instruct;
3. one stronger API or larger open-weight VLM as a scaling reference.

Use the same visual resolution, decoding policy, trace count, and candidate count for method comparisons.

## Main comparisons

- direct;
- standard CoT;
- self-consistency;
- full reflection;
- structured original trajectory;
- SceneRepair.

## Main metrics

- final accuracy;
- wrong-to-correct;
- correct-to-wrong;
- no-harm rate;
- repair apply rate;
- consistency gain;
- McNemar p-value;
- inference calls and runtime from saved model statistics.

## Localization

Run `synthetic_localization` on exactly the same evaluated examples. Report exact and within-one accuracy. For a stronger paper, manually annotate the earliest failure state on a stratified sample and report agreement between annotators.

## Ablations

At minimum:

- no reference-frame intervention;
- no operation intervention;
- no view dropout;
- symbolic only;
- VLM critic only;
- no earliest-step rule, selecting maximum causal score instead;
- full regeneration;
- no minimality term;
- no abstention;
- one versus three versus five traces;
- one versus three versus five repair candidates.

## Seeds

Run three seeds for all stochastic open-weight results. Report mean and standard deviation. Since the benchmark itself is fixed, also retain paired per-example predictions for significance testing.

## Training the optional critic

Train the learned critic only from training-free diagnosis outputs and automatically corrupted transitions. Do not select its hyperparameters on test labels. Either use a held-out subset of generated transitions or fix all hyperparameters before the final benchmark run.
