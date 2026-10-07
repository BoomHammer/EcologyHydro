"""M0 configuration and environment checks; no simulation entrypoint yet."""

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
    parser = argparse.ArgumentParser(description="EcologyHydro engineering checks")
    parser.add_argument("command", choices=("validate-config", "doctor"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--require-inputs", action="store_true")
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
