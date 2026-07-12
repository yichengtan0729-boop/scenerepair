from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def labels_for_choices(n: int) -> list[str]:
    if n < 1 or n > 26:
        raise ValueError(f"SceneRepair supports 1-26 choices, received {n}")
    return [chr(ord("A") + i) for i in range(n)]


def normalize_label(value: Any, n_choices: int | None = None) -> str:
    text = str(value or "").strip().upper()
    patterns = [r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([A-Z])\b", r"^\s*([A-Z])(?:[\).:\s]|$)"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            label = match.group(1)
            if n_choices is None or label in labels_for_choices(n_choices):
                return label
    if len(text) == 1 and text.isalpha():
        if n_choices is None or text in labels_for_choices(n_choices):
            return text
    return ""


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    arr = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-8)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    den = float(exp.sum())
    if not np.isfinite(den) or den <= 0:
        return [1.0 / len(arr)] * len(arr)
    return (exp / den).tolist()


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    p_arr = p_arr / max(float(p_arr.sum()), 1e-12)
    q_arr = q_arr / max(float(q_arr.sum()), 1e-12)
    m = 0.5 * (p_arr + q_arr)
    eps = 1e-12
    kl_pm = np.sum(p_arr * np.log((p_arr + eps) / (m + eps)))
    kl_qm = np.sum(q_arr * np.log((q_arr + eps) / (m + eps)))
    return float((0.5 * kl_pm + 0.5 * kl_qm) / np.log(2.0))


def weighted_vote(labels: Iterable[str], weights: Iterable[float]) -> tuple[str, dict[str, float]]:
    totals: dict[str, float] = {}
    for label, weight in zip(labels, weights):
        if not label:
            continue
        totals[label] = totals.get(label, 0.0) + max(float(weight), 0.0)
    if not totals:
        return "", {}
    winner = max(sorted(totals), key=lambda key: totals[key])
    total = sum(totals.values()) or 1.0
    return winner, {key: value / total for key, value in sorted(totals.items())}


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
