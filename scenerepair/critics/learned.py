from __future__ import annotations

from pathlib import Path

import joblib

from ..schemas import SpatialState


def transition_to_text(previous: SpatialState | None, current: SpatialState) -> str:
    prev = previous.to_dict() if previous else {"scene": "visual input"}
    cur = current.to_dict()
    return f"PREVIOUS={prev}\nCURRENT={cur}"


class LearnedTransitionCritic:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Learned critic not found: {path}")
        bundle = joblib.load(path)
        self.vectorizer = bundle["vectorizer"]
        self.classifier = bundle["classifier"]
        self.metadata = bundle.get("metadata", {})

    def score(self, previous: SpatialState | None, current: SpatialState) -> float:
        text = transition_to_text(previous, current)
        features = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(features)[0]
        classes = list(self.classifier.classes_)
        positive_index = classes.index(1) if 1 in classes else int(probabilities.argmax())
        return float(probabilities[positive_index])
