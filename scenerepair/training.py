from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from .critics.learned import transition_to_text
from .io import read_json, write_json
from .schemas import ReasoningTrace, SpatialState


def train_transition_critic(output_dir: str | Path, min_positive_score: float = 0.72) -> dict:
    output_dir = Path(output_dir)
    texts: list[str] = []
    labels: list[int] = []
    for diagnosis_path in sorted((output_dir / "diagnoses").glob("*.diagnosis.json")):
        stem = diagnosis_path.name.replace(".diagnosis.json", "")
        trace_path = output_dir / "traces" / f"{stem}.traces.json"
        if not trace_path.exists():
            continue
        diagnosis_payload = read_json(diagnosis_path)
        trace_payload = read_json(trace_path)
        traces = [ReasoningTrace.from_dict(item, trace_id=idx) for idx, item in enumerate(trace_payload["traces"])]
        for trace, diagnosis in zip(traces, diagnosis_payload["diagnoses"]):
            for assessment in diagnosis["assessments"]:
                idx = int(assessment["step_idx"])
                if idx >= len(trace.states):
                    continue
                previous = trace.states[idx - 1] if idx > 0 else None
                current = trace.states[idx]
                if float(assessment["consistency"]) >= min_positive_score:
                    texts.append(transition_to_text(previous, current)); labels.append(1)
            for intervention in diagnosis.get("interventions", []):
                idx = int(intervention["step_idx"])
                if idx >= len(trace.states):
                    continue
                previous = trace.states[idx - 1] if idx > 0 else None
                corrupted = SpatialState.from_dict(intervention["intervened_state"], step_idx=idx)
                if float(intervention.get("intervened_consistency", 1.0)) < min_positive_score:
                    texts.append(transition_to_text(previous, corrupted)); labels.append(0)
    if len(texts) < 20 or len(set(labels)) < 2:
        raise ValueError("Not enough automatically labeled transitions. Run diagnosis on more examples before train_critic.")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True)
    features = vectorizer.fit_transform(texts)
    train_x, test_x, train_y, test_y = train_test_split(features, np.asarray(labels), test_size=0.2, random_state=42, stratify=labels)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)
    probabilities = classifier.predict_proba(test_x)[:, list(classifier.classes_).index(1)]
    metrics = {"n_total": len(texts), "n_positive": int(sum(labels)), "n_negative": int(len(labels) - sum(labels)), "accuracy": float(accuracy_score(test_y, predictions)), "roc_auc": float(roc_auc_score(test_y, probabilities)), "classification_report": classification_report(test_y, predictions, output_dict=True)}
    checkpoint_dir = output_dir / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "transition_critic.joblib"
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "metadata": metrics}, model_path)
    metrics["model_path"] = str(model_path)
    write_json(checkpoint_dir / "transition_critic_metrics.json", metrics)
    return metrics
