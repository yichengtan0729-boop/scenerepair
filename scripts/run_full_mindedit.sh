#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/mindedit_qwen25vl7b.yaml}
python -m scenerepair.cli --config "$CONFIG" --task all
python -m scenerepair.cli --config "$CONFIG" --task synthetic_localization
python -m scenerepair.cli --config "$CONFIG" --task plot
