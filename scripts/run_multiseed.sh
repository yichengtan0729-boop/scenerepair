#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/mindedit_qwen25vl7b.yaml}
for SEED in 42 123 2026; do
  OUT="outputs/mindedit_seed${SEED}"
  python -m scenerepair.cli --config "$CONFIG" --task all --seed "$SEED" --output-dir "$OUT"
done
python scripts/aggregate_seeds.py outputs/mindedit_seed42 outputs/mindedit_seed123 outputs/mindedit_seed2026
