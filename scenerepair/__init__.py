"""SceneRepair package with lazy public entry points."""

from __future__ import annotations

from typing import Any

__version__ = "1.1.0"
__all__ = ["ExperimentConfig", "SceneRepairPipeline", "load_config"]


def __getattr__(name: str) -> Any:
    if name in {"ExperimentConfig", "load_config"}:
        from .config import ExperimentConfig, load_config
        return {"ExperimentConfig": ExperimentConfig, "load_config": load_config}[name]
    if name == "SceneRepairPipeline":
        from .pipeline import SceneRepairPipeline
        return SceneRepairPipeline
    raise AttributeError(name)
