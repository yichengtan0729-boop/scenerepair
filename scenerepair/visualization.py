from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .io import read_json


def create_all_figures(output_dir: str | Path) -> list[str]:
    output_dir = Path(output_dir); figure_dir = output_dir / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    method_path = output_dir / "tables" / "summary_methods.csv"
    if method_path.exists():
        frame = pd.read_csv(method_path).sort_values("accuracy", ascending=False)
        if not frame.empty:
            plt.figure(figsize=(9, 5)); plt.bar(frame["method"], frame["accuracy"]); plt.ylim(0, 1); plt.ylabel("Accuracy"); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
            path = figure_dir / "method_accuracy.png"; plt.savefig(path, dpi=200); plt.close(); created.append(str(path))
    repair_path = output_dir / "tables" / "per_example_repair.csv"
    if repair_path.exists():
        frame = pd.read_csv(repair_path)
        if not frame.empty:
            gains = frame["mean_selected_consistency"] - frame["mean_original_consistency"]
            plt.figure(figsize=(7, 5)); plt.hist(gains, bins=20); plt.xlabel("Consistency gain"); plt.ylabel("Examples"); plt.tight_layout()
            path = figure_dir / "consistency_gain_histogram.png"; plt.savefig(path, dpi=200); plt.close(); created.append(str(path))
    causal_rows = []
    for path in sorted((output_dir / "diagnoses").glob("*.diagnosis.json")):
        payload = read_json(path)
        for diagnosis in payload.get("diagnoses", []):
            for assessment in diagnosis.get("assessments", []):
                causal_rows.append({"step_idx": assessment["step_idx"], "consistency": assessment["consistency"]})
    if causal_rows:
        frame = pd.DataFrame(causal_rows); means = frame.groupby("step_idx")["consistency"].mean()
        plt.figure(figsize=(7, 5)); plt.plot(means.index, means.values, marker="o"); plt.ylim(0, 1); plt.xlabel("Reasoning step"); plt.ylabel("Mean transition consistency"); plt.tight_layout()
        path = figure_dir / "step_consistency.png"; plt.savefig(path, dpi=200); plt.close(); created.append(str(path))
    return created
