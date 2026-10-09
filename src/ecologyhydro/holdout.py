"""Require frozen training decisions for independent-year model execution."""

import json
from pathlib import Path

from ecologyhydro.cache import fingerprint


def verify_holdout(path, config, year, product, z, scenario_path, index_path):
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        year not in config.study.validation_years
        or year in config.study.calibration_years
        or frozen["test_year"] != year
        or frozen["training_years"] != list(config.study.calibration_years)
        or frozen["parameters"][product]["z"] != z
        or frozen["index_sha256"] != fingerprint(Path(index_path))["sha256"]
    ):
        raise ValueError("Holdout year, parameters or index differ from frozen training decisions")
    selected = frozen["scenarios"][product]
    if (
        scenario_path is None
        or Path(selected["path"]).resolve() != Path(scenario_path).resolve()
        or selected["sha256"] != fingerprint(Path(scenario_path))["sha256"]
    ):
        raise ValueError("Holdout scenario differs from frozen decision")
    for entry in frozen["inputs"]:
        if fingerprint(Path(entry["path"]))["sha256"] != entry["sha256"]:
            raise ValueError("Frozen holdout input has changed")
    return frozen
