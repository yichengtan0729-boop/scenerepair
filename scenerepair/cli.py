from __future__ import annotations

import argparse
import json
from typing import Any

import yaml

from .config import load_config
from .pipeline import SceneRepairPipeline


def _parse_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Overrides must use key=value")
    key, raw = value.split("=", 1)
    return key, yaml.safe_load(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneRepair experiment runner")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument(
        "--task",
        default="all",
        choices=["sample", "diagnose", "repair", "baselines", "evaluate", "plot", "all", "synthetic_localization", "train_critic"],
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a dotted config key")
    parser.add_argument("--output-dir")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = dict(_parse_override(item) for item in args.set)
    if args.output_dir is not None:
        overrides["run.output_dir"] = args.output_dir
    if args.start_index is not None:
        overrides["run.start_index"] = args.start_index
    if args.end_index is not None:
        overrides["run.end_index"] = args.end_index
    if args.limit is not None:
        overrides["run.limit"] = args.limit
    if args.seed is not None:
        overrides["run.seed"] = args.seed
    if args.overwrite:
        overrides["run.overwrite"] = True
    config = load_config(args.config, overrides)
    result = SceneRepairPipeline(config).run(args.task)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
