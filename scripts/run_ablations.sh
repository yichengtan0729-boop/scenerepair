#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/mindedit_qwen25vl7b.yaml}
python -m scenerepair.cli --config "$CONFIG" --task all --output-dir outputs/ablation_symbolic \
  --set critic.mode=symbolic --set critic.symbolic_weight=1.0 --set critic.vlm_weight=0.0
python -m scenerepair.cli --config "$CONFIG" --task all --output-dir outputs/ablation_no_intervention \
  --set critic.intervention_types='[]' --set critic.causal_threshold=1.0
python -m scenerepair.cli --config "$CONFIG" --task all --output-dir outputs/ablation_full_regeneration \
  --set repair.preserve_verified_prefix=false
python -m scenerepair.cli --config "$CONFIG" --task all --output-dir outputs/ablation_no_abstention \
  --set repair.abstain_margin=-1.0
