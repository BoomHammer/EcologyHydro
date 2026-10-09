"""Isolated official InVEST AWY worker, provenance, resume and station accounting."""

import argparse
import hashlib
import json
import logging
import os
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config
from ecologyhydro.runtime import configure_threads


def write_json(path, data):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def valid_completed(record):
    directory = Path(record["run_dir"])
    return (
        record.get("status") == "success"
        and all(
            (directory / item["path"]).is_file()
            and fingerprint(directory / item["path"])["sha256"] == item["sha256"]
            for item in record.get("outputs", [])
        )
        and bool(record.get("outputs"))
    )


def execute_trial(config, index_path, run_dir, year, landcover, scope, z, force=False):
    from osgeo import gdal, ogr

    from ecologyhydro.aggregation import (
        official_volume_check,
        read_observations,
        station_totals,
        write_csv,
        zone_volumes,
    )
    from ecologyhydro.model_inputs import prepare_inputs

    started = time.perf_counter()
    gdal.UseExceptions()
    ogr.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    if year not in config.study.calibration_years:
        raise ValueError("M3 engineering trials are restricted to calibration years")
    if not 1 <= z <= 30:
        raise ValueError("Engineering Z must be finite and between 1 and 30")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=run_dir / "model.log",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    state = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "status": "running",
        "year": year,
        "landcover": landcover,
        "scope": scope,
        "z": z,
        "parameter_status": "engineering_trial_not_calibrated",
        "timings_seconds": {},
    }
    write_json(run_dir / "run.json", state)
    try:
        prepared = prepare_inputs(index_path, config.paths.cache, year, landcover, scope)
        state["timings_seconds"]["input_verification_and_preparation"] = (
            time.perf_counter() - started
        )
        observation = config.paths.hydrology / "实测年径流量2018-2023.csv"
        source_files = [Path(path) for path in prepared["args"].values()]
        source_files += [Path(prepared["zones"]), Path(prepared["zone_station"]), observation]
        source_files += [
            Path(__file__),
            Path(__file__).with_name("aggregation.py"),
            Path(__file__).with_name("model_inputs.py"),
        ]
        sources = [fingerprint(path) for path in source_files]
        versions = {
            name: version(name)
            for name in ("natcap.invest", "GDAL", "numpy", "pygeoprocessing", "taskgraph")
        }
        recipe = {
            "sources": sources,
            "versions": versions,
            "z": z,
            "year": year,
            "scope": scope,
            "landcover": landcover,
            "model_workers": -1,
            "station_ids": config.study.station_ids,
            "threads": config.resources.threads_per_worker,
            "random_seed": config.random_seed,
        }
        key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        state["task_key"] = key
        index_dir = config.paths.cache / "m3/completed"
        index_dir.mkdir(parents=True, exist_ok=True)
        pointer = index_dir / f"{key}.json"
        args = {
            **prepared["args"],
            "workspace_dir": str(run_dir / "model"),
            "seasonality_constant": z,
            "n_workers": -1,
            "results_suffix": "",
            "demand_table_path": "",
            "valuation_table_path": "",
        }
        write_json(
            run_dir / "snapshot.json",
            {
                "config": config.model_dump(mode="json"),
                "recipe": recipe,
                "prepared": prepared,
                "model_args": args,
            },
        )
        if pointer.exists() and not force:
            previous = json.loads(pointer.read_text(encoding="utf-8"))
            if valid_completed(previous):
                state.update(
                    status="reused",
                    reused_run_id=previous["run_id"],
                    reused_run_dir=previous["run_dir"],
                )
                state["timings_seconds"]["total_worker"] = time.perf_counter() - started
                write_json(run_dir / "run.json", state)
                return state
        from natcap.invest.annual_water_yield import annual_water_yield

        validation_started = time.perf_counter()
        warnings = annual_water_yield.validate(args)
        write_json(run_dir / "validation.json", warnings)
        if warnings:
            raise ValueError(f"Official AWY validation failed: {warnings}")
        state["timings_seconds"]["official_validation"] = time.perf_counter() - validation_started
        model_started = time.perf_counter()
        registry = annual_water_yield.execute(args)
        state["timings_seconds"]["official_model"] = time.perf_counter() - model_started
        write_json(run_dir / "registry.json", registry)
        aggregate_started = time.perf_counter()
        records, checks = zone_volumes(
            registry["wyield"], prepared["zones"], args["precipitation_path"], registry["aet"]
        )
        comparisons = official_volume_check(records, registry["watershed_results_wyield"])
        observations = read_observations(observation, year)
        totals = station_totals(
            records,
            prepared["zone_station"],
            observations,
            run_dir.name,
            year,
            landcover,
            prepared["precipitation"],
            prepared["station_limit"],
        )
        if [row["station"] for row in totals] != config.study.station_ids[
            : prepared["station_limit"]
        ]:
            raise ValueError("Configured stations do not match M2 upstream station order")
        state["timings_seconds"]["aggregation_and_checks"] = time.perf_counter() - aggregate_started
        write_started = time.perf_counter()
        write_csv(run_dir / "zones.csv", records)
        write_csv(run_dir / "stations.csv", totals)
        write_json(
            run_dir / "checks.json",
            {
                **checks,
                "official_zone_comparisons": comparisons,
                "station_count": len(totals),
                "all_areas_complete": True,
            },
        )
        products = [
            run_dir / name
            for name in (
                "zones.csv",
                "stations.csv",
                "checks.json",
                "snapshot.json",
                "registry.json",
            )
        ]
        products.extend(
            Path(registry[key]) for key in ("wyield", "aet", "watershed_results_wyield_csv")
        )
        output_vector = Path(registry["watershed_results_wyield"])
        products.extend(
            output_vector.with_suffix(suffix) for suffix in (".shp", ".shx", ".dbf", ".prj")
        )
        state["outputs"] = [
            {"path": str(path.relative_to(run_dir)), "sha256": fingerprint(path)["sha256"]}
            for path in products
        ]
        if any(
            fingerprint(path)["sha256"] != before["sha256"]
            for path, before in zip(source_files, sources, strict=True)
        ):
            raise RuntimeError("An input changed during simulation")
        state["timings_seconds"]["report_and_integrity_checks"] = (
            time.perf_counter() - write_started
        )
        state["timings_seconds"]["total_worker"] = time.perf_counter() - started
        state["status"] = "success"
        write_json(run_dir / "run.json", state)
        write_json(pointer, state)
        return state
    except BaseException as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        state["timings_seconds"]["total_worker"] = time.perf_counter() - started
        write_json(run_dir / "run.json", state)
        logging.exception("M3 trial failed")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--landcover", choices=["copernicus", "fine"], required=True)
    parser.add_argument("--scope", choices=["tangnaihai", "full"], required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_threads(config.resources)
    execute_trial(
        config,
        args.index.resolve(),
        args.run_dir.resolve(),
        args.year,
        args.landcover,
        args.scope,
        args.z,
        args.force,
    )


if __name__ == "__main__":
    main()
