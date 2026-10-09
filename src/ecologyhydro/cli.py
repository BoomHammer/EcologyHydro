"""Configuration, preprocessing and bounded official AWY engineering trials."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ecologyhydro.config import ConfigError, load_config
from ecologyhydro.logging_utils import configure_logging
from ecologyhydro.runtime import configure_threads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EcologyHydro preprocessing and AWY trials")
    parser.add_argument(
        "command", choices=("validate-config", "doctor", "preprocess", "simulate", "trial")
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--require-inputs", action="store_true")
    parser.add_argument("--recipe", type=Path, default=Path("config/m2.yaml"))
    parser.add_argument("--m2-index", type=Path)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--landcover", choices=("copernicus", "fine"), default="copernicus")
    parser.add_argument("--scope", choices=("tangnaihai", "full"), default="full")
    parser.add_argument("--z", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("static", "climate", "landcover", "routing", "quality"),
        default=["static", "climate", "landcover", "routing", "quality"],
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, root=args.project_root)
        if args.require_inputs:
            config.validate_inputs()
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    configure_threads(config.resources)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    logger = configure_logging(config.paths.logs / f"{args.command}_{run_id}.log", config.log_level)
    logger.info("Configuration valid: %s", args.config)
    if args.command == "validate-config":
        print(config.model_dump_json(indent=2))
        return 0
    if args.command == "preprocess":
        from ecologyhydro.supervisor import run_preprocessing

        return run_preprocessing(config, args.config, args.recipe, args.stages)
    if args.command in {"simulate", "trial"}:
        from ecologyhydro.trials import run_group, run_suite

        index = (args.m2_index or config.paths.cache / "m2/latest.json").resolve()
        try:
            if args.command == "trial":
                run_suite(config, args.config, index, benchmark=not args.skip_benchmark)
            else:
                job = {
                    "year": args.year,
                    "landcover": args.landcover,
                    "scope": args.scope,
                    "z": args.z,
                    "force": args.force,
                }
                run_group(config, args.config, index, [job], 1, "single")
        except Exception:
            logger.exception("M3 execution failed")
            return 1
        return 0
    try:
        from ecologyhydro.diagnostics import check_environment

        report = check_environment(config)
        report_path = config.paths.logs / f"environment_{run_id}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Environment checks passed; report: %s", report_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception:
        logger.exception("Environment check failed")
        return 1
    return 0
