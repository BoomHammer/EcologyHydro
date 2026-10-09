"""Bounded official-AWY science scenarios; no fitting or holdout selection."""

import argparse
import csv
import json
import math
from pathlib import Path

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.config import UniqueKeyLoader, load_config, project_root
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import valid_completed, write_json


def root_values(profile):
    if "root_mm" in profile:
        values = profile["root_mm"]
    else:
        if "beta" in profile:
            depth = 10 * math.log(0.05) / math.log(profile["beta"])
        else:
            depth = 300 * math.log(0.05) / math.log(1 - profile["fraction_top_300mm"])
        values = [depth * 0.5, depth, depth * 1.5]
    if not all(math.isfinite(v) and v > 0 for v in values) or values != sorted(values):
        raise ValueError("Invalid root candidate bounds")
    return values


def candidate_rows(spec, landcover, variant):
    records = []
    for code, name in spec["mapping"][landcover].items():
        profile = spec["profiles"][name]
        kc = profile["kc"]
        roots = root_values(profile)
        if len(kc) != 3 or not all(math.isfinite(v) and v > 0 for v in kc) or kc != sorted(kc):
            raise ValueError("Invalid Kc bounds")
        records.append(
            {
                "lucode": int(code),
                "lulc_veg": profile["veg"],
                "kc": kc[{"kc_low": 0, "kc_high": 2}.get(variant, 1)],
                "root_depth": roots[{"root_low": 0, "root_high": 2}.get(variant, 1)],
                "profile": name,
                "kc_source": profile["kc_source"],
                "root_source": profile["root_source"],
                "evidence": profile["evidence"],
                "status": spec["status"],
            }
        )
    return records


def scale_depth(source_path, target_path, factor):
    import numpy as np
    from osgeo import gdal

    from ecologyhydro.spatial import OPTIONS, windows

    with (
        gdal.Open(str(source_path)) as source,
        gdal.Translate(str(target_path), source, creationOptions=OPTIONS) as target,
    ):
        for block in windows(source):
            values = source.ReadAsArray(*block)
            valid = source.GetRasterBand(1).GetMaskBand().ReadAsArray(*block) > 0
            values[valid] = np.minimum(values[valid] * factor, 2000)
            target.GetRasterBand(1).WriteArray(values, block[0], block[1])
    return {"factor": factor, "cap_mm": 2000, "status": "hypothetical_global_depth_perturbation"}


def prepare_scenarios(config, index_path, spec_path):
    import yaml

    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.model_inputs import prepare_inputs

    spec = yaml.load(spec_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    old = yaml.load(
        (project_root() / "config/biophysical_priors.yaml").read_text(encoding="utf-8"),
        Loader=UniqueKeyLoader,
    )
    for product in ("fine", "copernicus"):
        if set(spec["mapping"][product]) != set(old["mapping"][product]):
            raise ValueError("Candidate class mapping incomplete")
    cache = ArtifactCache(config.paths.cache / "m4")
    jobs = []
    variants = [
        "central",
        "kc_low",
        "kc_high",
        "root_low",
        "root_high",
        "soil_low",
        "soil_high",
        "z1",
        "z15",
        "z30",
    ]
    for product in ("copernicus", "fine"):
        prepared = prepare_inputs(index_path, config.paths.cache, 2019, product, "full")
        for variant in variants:
            table_variant = variant if variant in variants[1:5] else "central"

            def table_producer(out, p=product, v=table_variant):
                records = candidate_rows(spec, p, v)
                write_csv(out / "biophysical.csv", records)
                return {"landcover": p, "variant": v, "status": spec["status"]}

            table = cache.build(
                "tables",
                [spec_path, Path(__file__)],
                {"landcover": product, "variant": table_variant},
                table_producer,
            )
            overrides = {"biophysical_table_path": str(table / "biophysical.csv")}
            if variant.startswith("soil_"):
                factor = 0.5 if variant == "soil_low" else 4 / 3
                source = Path(prepared["args"]["depth_to_root_rest_layer_path"])
                soil = cache.build(
                    "soil_depth",
                    [source, Path(__file__)],
                    {"factor": factor},
                    lambda out, src=source, f=factor: scale_depth(src, out / "depth.tif", f),
                )
                overrides["depth_to_root_rest_layer_path"] = str(soil / "depth.tif")
            scenario = {
                "name": variant,
                "landcover": product,
                "scope": "full",
                "overrides": overrides,
                "status": spec["status"],
                "policy": "one-factor-at-a-time; no observed-flow parameter selection",
            }

            def scenario_producer(out, record=scenario):
                write_json(out / "scenario.json", record)
                return record

            artifact = cache.build(
                "scenarios",
                [spec_path, Path(__file__), *map(Path, overrides.values())],
                scenario,
                scenario_producer,
            )
            jobs.append(
                {
                    "year": 2019,
                    "landcover": product,
                    "scope": "full",
                    "z": float(variant[1:]) if variant.startswith("z") else 5,
                    "scenario": str(artifact / "scenario.json"),
                    "variant": variant,
                }
            )
    return jobs


def summarize(batch_dir, batch, output):
    from ecologyhydro.aggregation import write_csv

    results = []
    for entry in batch["completed"]:
        directory = Path(entry["directory"])
        record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if record["status"] == "reused":
            directory = Path(record["reused_run_dir"])
            record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if not valid_completed(record):
            raise ValueError("Invalid completed sensitivity result")
        with (directory / "stations.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                results.append(
                    {
                        "variant": entry["job"]["variant"],
                        "landcover": row["landcover"],
                        "station": row["station"],
                        "year": int(row["year"]),
                        "z": entry["job"]["z"],
                        "awy_yield_1e8_m3": float(row["natural_yield_1e8_m3"]),
                        "run_id": row["run_id"],
                        "scientific_status": "sensitivity_not_accuracy",
                    }
                )
    if len(results) != len(batch["jobs"]) * 11:
        raise ValueError("Missing sensitivity station rows")
    central = {
        (r["landcover"], r["station"]): r["awy_yield_1e8_m3"]
        for r in results
        if r["variant"] == "central"
    }
    for row in results:
        base = central[row["landcover"], row["station"]]
        row["change_from_central_percent"] = (
            100 * (row["awy_yield_1e8_m3"] / base - 1) if base else None
        )
    write_csv(output / "stations.csv", results)
    write_json(
        output / "summary.json",
        {
            "status": "success",
            "batch": str(batch_dir / "batch.json"),
            "tasks": len(batch["jobs"]),
            "station_rows": len(results),
            "elapsed_seconds": batch["elapsed_seconds"],
            "peak_tree_rss_bytes": batch["peak_tree_rss_bytes"],
            "calibrated": False,
            "validation_used": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = project_root()
    config = load_config()
    configure_threads(config.resources)
    index = root / "project/cache/m2/latest.json"
    jobs = prepare_scenarios(config, index, root / "config/biophysical_candidates.yaml")
    from ecologyhydro.trials import run_group

    output = root / "project/m4/sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "plan.json", {"jobs": jobs, "workers": 2, "status": "registered_before_execution"}
    )
    if not args.prepare_only:
        directory, batch = run_group(
            config, root / "config.yaml", index, jobs, 2, "sensitivity", batch_root=output / "runs"
        )
        summarize(directory, batch, output)


if __name__ == "__main__":
    main()
