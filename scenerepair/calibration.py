from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def weighted_geometric_mean(values: dict[str, float], weights: dict[str, float], eps: float = 1e-8) -> float:
    """Stable weighted geometric mean for root-cause evidence."""
    active = [(key, max(eps, clamp01(values.get(key, 0.0))), max(0.0, weights.get(key, 0.0))) for key in values]
    total = sum(weight for _, _, weight in active)
    if total <= 0:
        return 0.0
    return float(math.exp(sum(weight * math.log(value) for _, value, weight in active) / total))


def lower_confidence_bound(samples: Iterable[float], confidence: float = 0.90) -> dict[str, float]:
    values = [float(item) for item in samples]
    if not values:
        return {"mean": 0.0, "std": 0.0, "stderr": 0.0, "z": 0.0, "lcb": float("-inf"), "n": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    stderr = std / math.sqrt(len(values))
    confidence = min(0.999, max(0.50, float(confidence)))
    z = NormalDist().inv_cdf(confidence)
    return {"mean": mean, "std": std, "stderr": stderr, "z": z, "lcb": mean - z * stderr, "n": len(values)}


def interventional_agreement(effects: Iterable[float], labels: Iterable[str] | None = None) -> float:
    """Agreement across semantically equivalent intervention realizations.

    The score combines dispersion of effect magnitudes and agreement of the
    induced answer label. It is label-free: labels here are model predictions.
    """
    values = [float(item) for item in effects]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / max(1, len(values) - 1)
    magnitude_agreement = 1.0 - min(1.0, math.sqrt(variance) / 0.25)
    label_agreement = 1.0
    if labels is not None:
        label_list = [str(item) for item in labels]
        if label_list:
            counts: dict[str, int] = {}
            for label in label_list:
                counts[label] = counts.get(label, 0) + 1
            label_agreement = max(counts.values()) / len(label_list)
    return clamp01(0.6 * magnitude_agreement + 0.4 * label_agreement)
