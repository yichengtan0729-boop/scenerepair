from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .io import read_json, write_json


def _bootstrap_ci(values: list[float], seed: int = 0, samples: int = 5000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(samples, len(arr)))
    estimates = arr[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _method_summary(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    subset = frame[frame["method"] == method]
    correct = subset["correct"].astype(float).tolist()
    low, high = _bootstrap_ci(correct)
    return {"method": method, "n": int(len(subset)), "accuracy": float(np.mean(correct)) if correct else 0.0, "ci95_low": low, "ci95_high": high}


def evaluate_outputs(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "repairs").glob("*.repair.json")):
        payload = read_json(path); example = payload["example"]
        answer = str(payload.get("answer", example.get("answer", "")))
        original = str(payload.get("original_prediction", "")); repaired = str(payload.get("repaired_prediction", ""))
        original_correct = bool(answer and original == answer); repaired_correct = bool(answer and repaired == answer)
        applied = int(payload.get("repair_applied_count", 0)); attempted = int(payload.get("repair_attempted_count", 0))
        rows.extend([
            {"example_id": example["example_id"], "task": example.get("task", "unknown"), "method": "structured_original", "prediction": original, "answer": answer, "correct": original_correct, "repair_applied_count": 0},
            {"example_id": example["example_id"], "task": example.get("task", "unknown"), "method": "scenerepair", "prediction": repaired, "answer": answer, "correct": repaired_correct, "repair_applied_count": applied},
        ])
        trace_records = payload.get("trace_records", [])
        pair_rows.append({
            "example_id": example["example_id"], "task": example.get("task", "unknown"), "answer": answer,
            "original_prediction": original, "repaired_prediction": repaired,
            "original_correct": original_correct, "repaired_correct": repaired_correct,
            "wrong_to_correct": (not original_correct) and repaired_correct,
            "correct_to_wrong": original_correct and (not repaired_correct),
            "repair_attempted": attempted > 0, "repair_applied": applied > 0,
            "attempted_traces": attempted, "applied_traces": applied,
            "mean_original_consistency": float(np.mean([item["diagnosis"]["global_consistency"] for item in trace_records])) if trace_records else 0.0,
            "mean_selected_consistency": float(np.mean([item["selected_score"].get("consistency", 0.0) for item in trace_records])) if trace_records else 0.0,
            "mean_causal_score": float(np.mean([item["diagnosis"]["causal_score"] for item in trace_records])) if trace_records else 0.0,
        })
    for path in sorted((output_dir / "baselines").glob("*.baselines.json")):
        payload = read_json(path); example = payload["example"]; answer = str(payload.get("answer", example.get("answer", "")))
        for method, method_payload in payload.get("methods", {}).items():
            prediction = str(method_payload.get("prediction", ""))
            rows.append({"example_id": example["example_id"], "task": example.get("task", "unknown"), "method": method, "prediction": prediction, "answer": answer, "correct": bool(answer and prediction == answer), "repair_applied_count": 0})
    table_dir = output_dir / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    method_frame = pd.DataFrame(rows); pair_frame = pd.DataFrame(pair_rows)
    method_frame.to_csv(table_dir / "per_example_methods.csv", index=False); pair_frame.to_csv(table_dir / "per_example_repair.csv", index=False)
    methods = sorted(method_frame["method"].unique()) if not method_frame.empty else []
    method_summaries = [_method_summary(method_frame, method) for method in methods]
    by_task: list[dict[str, Any]] = []
    if not method_frame.empty:
        for (task, method), subset in method_frame.groupby(["task", "method"], dropna=False):
            values = subset["correct"].astype(float).tolist(); low, high = _bootstrap_ci(values, seed=17)
            by_task.append({"task": task, "method": method, "n": int(len(subset)), "accuracy": float(np.mean(values)), "ci95_low": low, "ci95_high": high})
    pd.DataFrame(method_summaries).to_csv(table_dir / "summary_methods.csv", index=False); pd.DataFrame(by_task).to_csv(table_dir / "summary_by_task.csv", index=False)
    repair_summary: dict[str, Any] = {"n": int(len(pair_frame)), "original_accuracy": 0.0, "repaired_accuracy": 0.0, "net_accuracy_gain": 0.0, "wrong_to_correct": 0, "correct_to_wrong": 0, "no_harm_rate": 1.0, "repair_attempt_rate": 0.0, "repair_apply_rate": 0.0, "mcnemar_exact_p": 1.0}
    if not pair_frame.empty:
        original_accuracy = float(pair_frame["original_correct"].mean()); repaired_accuracy = float(pair_frame["repaired_correct"].mean())
        wrong_to_correct = int(pair_frame["wrong_to_correct"].sum()); correct_to_wrong = int(pair_frame["correct_to_wrong"].sum()); discordant = wrong_to_correct + correct_to_wrong
        p_value = float(binomtest(min(wrong_to_correct, correct_to_wrong), discordant, 0.5).pvalue) if discordant else 1.0
        originally_correct = int(pair_frame["original_correct"].sum())
        repair_summary.update({"original_accuracy": original_accuracy, "repaired_accuracy": repaired_accuracy, "net_accuracy_gain": repaired_accuracy - original_accuracy, "wrong_to_correct": wrong_to_correct, "correct_to_wrong": correct_to_wrong, "no_harm_rate": 1.0 - correct_to_wrong / originally_correct if originally_correct else 1.0, "repair_attempt_rate": float(pair_frame["repair_attempted"].mean()), "repair_apply_rate": float(pair_frame["repair_applied"].mean()), "mean_consistency_gain": float((pair_frame["mean_selected_consistency"] - pair_frame["mean_original_consistency"]).mean()), "mean_causal_score": float(pair_frame["mean_causal_score"].mean()), "mcnemar_exact_p": p_value})
    result = {"method_summaries": method_summaries, "by_task": by_task, "repair_summary": repair_summary}
    write_json(table_dir / "summary.json", result)
    return result
