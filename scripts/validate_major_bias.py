"""Freeze the development fit, then execute the independent 2023 year exactly as frozen."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from calibrate_major_bias import metrics

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import ArtifactCache, fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.et0_trials import aligned_et
from ecologyhydro.model_inputs import prepare_inputs
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.trials import run_group
from ecologyhydro.water_balance import read_csv


def freeze(root):
    training = root / "project/calibration/major_bias"
    summary = read_json(training / "summary.json")
    if not summary["quadrature_accepted"]:
        raise ValueError("Official spatial verification has not passed")
    fitted = read_json(training / "parameters.json")
    if fitted["climate"] != "terraclimate_pair":
        raise ValueError("This holdout driver requires the selected paired climate")
    output = training / "holdout_2023"
    output.mkdir(exist_ok=True)
    frozen_path = output / "frozen_training.json"
    if frozen_path.exists():
        existing = read_json(frozen_path)
        if (
            existing["training_parameter_sha256"]
            != fingerprint(training / "parameters.json")["sha256"]
            or existing["index_sha256"]
            != fingerprint(root / "project/repairs/soil_routing/m2_index.json")["sha256"]
        ):
            raise ValueError("Training changed after freezing; no holdout-driven retuning")
        for entry in existing["tables"].values():
            if fingerprint(Path(entry["path"]))["sha256"] != entry["sha256"]:
                raise ValueError("Frozen table was modified")
        return output, existing
    decision = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "training_years": [2019, 2020, 2021, 2022],
        "test_year": 2023,
        "parameters": fitted["parameters"],
        "climate": fitted["climate"],
        "training_parameter_sha256": fingerprint(training / "parameters.json")["sha256"],
        "annual_kc_policy": "mean of four fitted training-year tables; no test-year tuning",
        "index_sha256": fingerprint(root / "project/repairs/soil_routing/m2_index.json")["sha256"],
        "tables": {},
    }
    for product in ("fine", "copernicus"):
        tables = [read_csv(training / f"biophysical_{product}_{y}.csv") for y in range(2019, 2023)]
        codes = [int(r["lucode"]) for r in tables[0]]
        if any([int(r["lucode"]) for r in t] != codes for t in tables):
            raise ValueError("Training-year table classes differ")
        rows = []
        for i, code in enumerate(codes):
            row = {"lucode": code}
            for field in ("lulc_veg", "root_depth", "kc"):
                values = [float(t[i][field]) for t in tables]
                if field != "kc" and len(set(values)) != 1:
                    raise ValueError("Only annual Kc may vary in the frozen extrapolation")
                row[field] = float(np.mean(values)) if field == "kc" else values[0]
            row["lulc_veg"] = int(row["lulc_veg"])
            rows.append(row)
        path = output / f"biophysical_{product}.csv"
        write_csv(path, rows)
        decision["tables"][product] = fingerprint(path)
    path = output / "frozen_training.json"
    if path.exists() and read_json(path) != decision:
        raise ValueError("Training decisions changed after freezing; do not retune on holdout")
    write_json(path, decision)
    return output, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    root, config = project_root(), load_config()
    configure_threads(config.resources)
    output, decision = freeze(root)
    if args.freeze_only:
        print(output / "frozen_training.json")
        return
    index_path = root / "project/repairs/soil_routing/m2_index.json"
    provider = root / "project/repairs/climate_reference_full_holdout"
    cache = ArtifactCache(config.paths.cache / "major_bias_holdout")
    jobs, scenarios, inputs = [], {}, [fingerprint(output / "frozen_training.json")]
    for product in ("fine", "copernicus"):
        prepared = prepare_inputs(index_path, config.paths.cache, 2023, product, "full")
        overrides = {"biophysical_table_path": decision["tables"][product]["path"]}
        for field, variable in (("precipitation_path", "ppt"), ("eto_path", "pet")):
            source = provider / f"provider_{variable}_2023_aligned.tif"
            artifact = cache.build(
                "aligned",
                [source, Path(prepared["zones"])],
                {"year": 2023, "field": field},
                lambda out, s=source, z=prepared["zones"]: aligned_et(s, z, out),
            )
            overrides[field] = str(artifact / "et0.tif")
        scenario_path = output / f"scenario_{product}.json"
        write_json(
            scenario_path,
            {
                "name": "frozen_major_bias_holdout",
                "year": 2023,
                "landcover": product,
                "scope": "full",
                "overrides": overrides,
                "et0_label": "terraclimate_pair_et0",
                "precipitation_label": "terraclimate_pair_p",
            },
        )
        scenarios[product] = fingerprint(scenario_path)
        inputs += [fingerprint(Path(p)) for p in overrides.values()]
        jobs.append(
            {
                "year": 2023,
                "landcover": product,
                "scope": "full",
                "z": decision["parameters"][product]["z"],
                "scenario": str(scenario_path),
                "holdout_manifest": str(output / "frozen_manifest.json"),
                "variant": "independent_holdout",
            }
        )
    write_json(
        output / "frozen_manifest.json", {**decision, "scenarios": scenarios, "inputs": inputs}
    )
    directory, batch = run_group(
        config,
        root / "config.yaml",
        index_path,
        jobs,
        2,
        "holdout_2023",
        batch_root=output / "runs",
    )
    if batch["status"] != "success":
        raise ValueError("Holdout official execution failed")
    rows = []
    for entry in batch["completed"]:
        run = Path(entry["directory"])
        state = read_json(run / "run.json")
        if state["status"] == "reused":
            run = Path(state["reused_run_dir"])
        for row in read_csv(run / "stations.csv"):
            rows.append(
                {
                    "landcover": row["landcover"],
                    "station": row["station"],
                    "year": 2023,
                    "predicted": float(row["natural_yield_1e8_m3"]),
                    "observed": float(row["observed_1e8_m3"]),
                }
            )
    scores = {}
    for product in ("fine", "copernicus"):
        selected = [r for r in rows if r["landcover"] == product]
        scores[product] = metrics(
            np.array([r["predicted"] for r in selected]),
            np.array([r["observed"] for r in selected]),
        )
    write_csv(output / "stations.csv", rows)
    write_json(
        output / "summary.json",
        {
            "batch": str(directory / "batch.json"),
            "metrics": scores,
            "parameters_retuned_on_holdout": False,
            "status": "independent_test_complete",
        },
    )
    print(scores)


if __name__ == "__main__":
    main()
