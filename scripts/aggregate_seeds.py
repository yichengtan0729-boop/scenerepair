from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/aggregate_seeds.py OUTPUT_DIR [OUTPUT_DIR ...]")
frames = []
for seed_idx, directory in enumerate(sys.argv[1:]):
    path = Path(directory) / "tables" / "summary_methods.csv"
    frame = pd.read_csv(path)
    frame["run"] = Path(directory).name
    frames.append(frame)
all_runs = pd.concat(frames, ignore_index=True)
summary = all_runs.groupby("method")["accuracy"].agg(["mean", "std", "count"]).reset_index()
out = Path("outputs") / "multiseed_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out, index=False)
print(summary.to_string(index=False))
print(out)
