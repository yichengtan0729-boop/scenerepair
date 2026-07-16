from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .baselines import run_baselines
from .config import ExperimentConfig
from .critics import TransitionCritic
from .data import load_examples
from .diagnosis import CausalDiagnoser
from .evaluation import evaluate_outputs
from .interventions import apply_intervention, generate_interventions
from .io import read_json, write_json
from .models import build_model
from .parsing import parse_reasoning_trace
from .prompts import structured_reasoning_prompt
from .repair import MinimalSuffixRepairer
from .schemas import DiagnosisResult, ReasoningTrace, SpatialExample
from .utils import set_seed, weighted_vote


class SceneRepairPipeline:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for directory in ("examples", "traces", "diagnoses", "repairs", "baselines", "tables", "figures", "synthetic"):
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "resolved_config.json", config.to_dict())
        self._model = None
        self._critic = None
        self._diagnoser = None
        self._repairer = None
        set_seed(config.run.seed)

    @property
    def model(self):
        if self._model is None:
            self._model = build_model(self.config.model)
        return self._model

    @property
    def critic(self) -> TransitionCritic:
        if self._critic is None:
            self._critic = TransitionCritic(self.config.critic, self.model)
        return self._critic

    @property
    def diagnoser(self) -> CausalDiagnoser:
        if self._diagnoser is None:
            self._diagnoser = CausalDiagnoser(self.model, self.critic, self.config.critic)
        return self._diagnoser

    @property
    def repairer(self) -> MinimalSuffixRepairer:
        if self._repairer is None:
            self._repairer = MinimalSuffixRepairer(
                self.model, self.critic, self.config.repair, self.config.run.max_reasoning_steps
            )
        return self._repairer

    @staticmethod
    def _safe_id(example_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(example_id))[:180]

    def _path(self, kind: str, example_id: str) -> Path:
        suffix = {
            "examples": ".json", "traces": ".traces.json", "diagnoses": ".diagnosis.json",
            "repairs": ".repair.json", "baselines": ".baselines.json", "synthetic": ".synthetic.json",
        }[kind]
        return self.output_dir / kind / f"{self._safe_id(example_id)}{suffix}"

    def load_examples(self) -> list[SpatialExample]:
        return load_examples(
            self.config.data,
            start_index=self.config.run.start_index,
            end_index=self.config.run.end_index,
            limit=self.config.run.limit,
        )

    def _generate_traces(self, example: SpatialExample) -> list[ReasoningTrace]:
        prompt = structured_reasoning_prompt(example, self.config.run.max_reasoning_steps)
        traces: list[ReasoningTrace] = []
        for trace_id in range(max(1, self.config.run.num_traces)):
            seed = self.config.run.seed + trace_id + 997 * sum(ord(ch) for ch in example.example_id)
            raw = self.model.generate(example.images, prompt, seed=seed)
            trace = parse_reasoning_trace(raw, trace_id, len(example.choices))
            trace.generation_seed = seed
            traces.append(trace)
        return traces

    def sample_example(self, example: SpatialExample) -> list[ReasoningTrace]:
        path = self._path("traces", example.example_id)
        if path.exists() and not self.config.run.overwrite:
            payload = read_json(path)
            return [
                ReasoningTrace.from_dict(item, trace_id=idx, n_choices=len(example.choices))
                for idx, item in enumerate(payload["traces"])
            ]
        traces = self._generate_traces(example)
        write_json(self._path("examples", example.example_id), example.to_public_dict())
        write_json(path, {
            "example": example.to_public_dict(),
            "traces": [trace.to_dict() for trace in traces],
            "model_stats": self.model.stats.to_dict(),
        })
        return traces

    def diagnose_example(self, example: SpatialExample, traces: list[ReasoningTrace] | None = None) -> list[DiagnosisResult]:
        path = self._path("diagnoses", example.example_id)
        if path.exists() and not self.config.run.overwrite:
            payload = read_json(path)
            return [DiagnosisResult.from_dict(item) for item in payload["diagnoses"]]
        traces = traces or self.sample_example(example)
        diagnoses = [self.diagnoser.diagnose(example, trace) for trace in traces]
        write_json(path, {
            "example": example.to_public_dict(),
            "diagnoses": [diagnosis.to_dict() for diagnosis in diagnoses],
            "model_stats": self.model.stats.to_dict(),
        })
        return diagnoses

    @staticmethod
    def _prediction_from_trace(trace: ReasoningTrace, diagnosis: DiagnosisResult | None = None) -> str:
        if trace.predicted_label:
            return trace.predicted_label
        distribution = diagnosis.original_distribution if diagnosis else trace.option_distribution
        return max(distribution, key=distribution.get) if distribution else ""

    def repair_example(
        self,
        example: SpatialExample,
        traces: list[ReasoningTrace] | None = None,
        diagnoses: list[DiagnosisResult] | None = None,
    ) -> dict:
        path = self._path("repairs", example.example_id)
        if path.exists() and not self.config.run.overwrite:
            return read_json(path)
        traces = traces or self.sample_example(example)
        diagnoses = diagnoses or self.diagnose_example(example, traces)
        records: list[dict] = []
        original_labels: list[str] = []
        original_weights: list[float] = []
        repaired_labels: list[str] = []
        repaired_weights: list[float] = []
        for trace, diagnosis in zip(traces, diagnoses):
            record = self.repairer.repair(example, trace, diagnosis)
            repaired_trace = ReasoningTrace.from_dict(
                record["repaired_trace"], trace_id=trace.trace_id, n_choices=len(example.choices)
            )
            original_label = self._prediction_from_trace(trace, diagnosis)
            repaired_label = self._prediction_from_trace(repaired_trace)
            if not repaired_label:
                distribution = record["selected_score"].get("distribution", {})
                repaired_label = max(distribution, key=distribution.get) if distribution else original_label
            original_weight = max(
                1e-6,
                diagnosis.global_consistency
                * max(diagnosis.original_distribution.values(), default=0.5)
                * max(0.1, diagnosis.interventional_consistency),
            )
            repaired_weight = max(
                1e-6,
                float(record["selected_score"].get("consistency", diagnosis.global_consistency))
                * float(record["selected_score"].get("confidence", 0.5)),
            )
            original_labels.append(original_label)
            original_weights.append(original_weight)
            repaired_labels.append(repaired_label)
            repaired_weights.append(repaired_weight)
            record["diagnosis"] = diagnosis.to_dict()
            record["original_prediction"] = original_label
            record["repaired_prediction"] = repaired_label
            records.append(record)
        original_prediction, original_vote = weighted_vote(original_labels, original_weights)
        repaired_prediction, repaired_vote = weighted_vote(repaired_labels, repaired_weights)
        payload = {
            "example": example.to_public_dict(), "answer": example.answer,
            "original_prediction": original_prediction, "repaired_prediction": repaired_prediction,
            "original_vote_distribution": original_vote, "repaired_vote_distribution": repaired_vote,
            "trace_records": records,
            "repair_attempted_count": sum(bool(item["repair_attempted"]) for item in records),
            "repair_applied_count": sum(bool(item["repair_applied"]) for item in records),
            "model_stats": self.model.stats.to_dict(),
        }
        write_json(path, payload)
        return payload

    def baselines_example(self, example: SpatialExample) -> dict:
        path = self._path("baselines", example.example_id)
        if path.exists() and not self.config.run.overwrite:
            return read_json(path)
        payload = {
            "example": example.to_public_dict(), "answer": example.answer,
            "methods": run_baselines(
                self.model, example, self.config.run.baseline_methods,
                self.config.run.num_traces,
                self.config.run.seed + sum(ord(ch) for ch in example.example_id),
            ),
            "model_stats": self.model.stats.to_dict(),
        }
        write_json(path, payload)
        return payload

    def _run_loop(self, stage: str) -> None:
        examples = self.load_examples()
        errors: list[dict[str, Any]] = []
        for example in tqdm(examples, desc=f"SceneRepair {stage}"):
            try:
                if stage == "sample":
                    self.sample_example(example)
                elif stage == "diagnose":
                    traces = self.sample_example(example)
                    self.diagnose_example(example, traces)
                elif stage == "repair":
                    traces = self.sample_example(example)
                    diagnoses = self.diagnose_example(example, traces)
                    self.repair_example(example, traces, diagnoses)
                elif stage == "baselines":
                    self.baselines_example(example)
                elif stage == "all":
                    traces = self.sample_example(example)
                    diagnoses = self.diagnose_example(example, traces)
                    self.repair_example(example, traces, diagnoses)
                    if self.config.run.run_baselines:
                        self.baselines_example(example)
                else:
                    raise ValueError(stage)
            except Exception as exc:
                errors.append({
                    "example_id": example.example_id, "stage": stage,
                    "error": str(exc), "traceback": traceback.format_exc(),
                })
                write_json(self.output_dir / "errors.json", errors)
                if self.config.run.fail_fast:
                    raise
        if errors:
            print(f"Completed with {len(errors)} failed examples. See {self.output_dir / 'errors.json'}")

    def synthetic_localization(self) -> dict:
        examples = self.load_examples()
        rows: list[dict] = []
        for example in tqdm(examples, desc="Synthetic localization"):
            traces = self.sample_example(example)
            if not traces or len(traces[0].states) < 2:
                continue
            trace = traces[0]
            injected_step = min(2, len(trace.states) - 1)
            candidates = generate_interventions(
                trace.states[injected_step],
                ["frame_swap", "operation_inverse", "relation_flip", "object_swap"],
                len(example.images),
            )
            if not candidates:
                continue
            corrupted = apply_intervention(trace, injected_step, candidates[0])
            diagnosis = self.diagnoser.diagnose(example, corrupted)
            row = {
                "example_id": example.example_id,
                "injected_step": injected_step,
                "injection": candidates[0].name,
                "localized_step": diagnosis.localized_step,
                "exact": diagnosis.localized_step == injected_step,
                "within_one": diagnosis.localized_step is not None and abs(diagnosis.localized_step - injected_step) <= 1,
                "root_score": diagnosis.root_score,
                "necessity": diagnosis.necessity_score,
                "sufficiency": diagnosis.sufficiency_score,
                "diagnosis": diagnosis.to_dict(),
            }
            rows.append(row)
            write_json(self._path("synthetic", example.example_id), row)
        summary = {
            "n": len(rows),
            "exact_accuracy": sum(item["exact"] for item in rows) / len(rows) if rows else 0.0,
            "within_one_accuracy": sum(item["within_one"] for item in rows) / len(rows) if rows else 0.0,
            "mean_root_score": sum(item["root_score"] for item in rows) / len(rows) if rows else 0.0,
            "rows": rows,
        }
        write_json(self.output_dir / "tables" / "synthetic_localization.json", summary)
        return summary

    def run(self, task: str) -> Any:
        task = task.lower()
        if task in {"sample", "diagnose", "repair", "baselines", "all"}:
            self._run_loop(task)
            return evaluate_outputs(self.output_dir) if task == "all" else None
        if task == "evaluate":
            return evaluate_outputs(self.output_dir)
        if task == "plot":
            from .visualization import create_all_figures
            return create_all_figures(self.output_dir)
        if task == "synthetic_localization":
            return self.synthetic_localization()
        if task == "train_critic":
            from .training import train_transition_critic
            return train_transition_critic(self.output_dir)
        raise ValueError(f"Unknown task: {task}")
