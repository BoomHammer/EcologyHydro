"""ET0-weighted annual wheat-maize rotation hypotheses on unchanged fine LULC."""

import csv
import json
from contextlib import ExitStack
from pathlib import Path

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.config import UniqueKeyLoader, load_config, project_root
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import valid_completed, write_json


def weighted_kc(monthly_depth_sums, monthly_coefficients):
    import numpy as np

    weights = np.asarray(monthly_depth_sums, dtype=float)
    coefficients = np.asarray(monthly_coefficients, dtype=float)
    if weights.shape != (12,) or coefficients.shape != (12,):
        raise ValueError("Twelve monthly totals and coefficients are required")
    if (
        not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(coefficients))
        or np.any(weights < 0)
        or np.any(coefficients <= 0)
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid monthly Kc weights")
    return float(weights @ coefficients / weights.sum())


def monthly_statistics(index, sources, out):
    import numpy as np
    from osgeo import gdal

    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.spatial import NODATA, windows

    grid = index["grid"]
    with ExitStack() as stack:
        zones = stack.enter_context(
            gdal.Open(str(Path(index["routing"]["partitions"]) / "zones.tif"))
        )
        fine = stack.enter_context(gdal.Open(index["aligned"]["lulc_fine"]))
        if (
            fine.GetGeoTransform() != zones.GetGeoTransform()
            or fine.GetProjection() != zones.GetProjection()
            or (fine.RasterXSize, fine.RasterYSize) != (zones.RasterXSize, zones.RasterYSize)
        ):
            raise ValueError("Mismatched rotation grid")
        count = np.zeros(12, dtype=np.int64)
        crop_count = count.copy()
        total = np.zeros((12, 12))
        crop_total = total.copy()
        for month, path in enumerate(sources):
            with gdal.Warp(
                "",
                str(path),
                format="VRT",
                dstSRS=grid["crs"],
                outputBounds=grid["bounds"],
                width=grid["width"],
                height=grid["height"],
                resampleAlg="bilinear",
                dstNodata=NODATA,
                outputType=gdal.GDT_Float32,
            ) as pet:
                for block in windows(zones):
                    ids = zones.ReadAsArray(*block)
                    inside = ids > 0
                    if not inside.any():
                        continue
                    crop = inside & (fine.ReadAsArray(*block) == 2)
                    values = pet.ReadAsArray(*block).astype(np.float64) * 0.1
                    valid = pet.GetRasterBand(1).GetMaskBand().ReadAsArray(*block) > 0
                    if np.any(inside & (~valid | ~np.isfinite(values) | (values < 0))):
                        raise ValueError("Invalid monthly ET0 inside watershed")
                    if month == 0:
                        count += np.bincount(ids[inside], minlength=12)
                        crop_count += np.bincount(ids[crop], minlength=12)
                    total[month] += np.bincount(ids[inside], weights=values[inside], minlength=12)
                    crop_total[month] += np.bincount(ids[crop], weights=values[crop], minlength=12)
    records = [
        {
            "month": m + 1,
            "zone_id": z,
            "pixels": int(count[z]),
            "crop_pixels": int(crop_count[z]),
            "et0_sum_mm": total[m, z],
            "crop_et0_sum_mm": crop_total[m, z],
        }
        for m in range(12)
        for z in range(1, 12)
    ]
    write_csv(out / "monthly_zones.csv", records)
    return {
        "crop_monthly_sums": crop_total.sum(axis=1).tolist(),
        "crop_pixels": int(crop_count.sum()),
        "method": "bilinear monthly raw PET * 0.1",
    }


def prepare(config, index, index_path, crop_path, spec_path):
    import yaml

    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.model_inputs import prepare_inputs, verified_artifact
    from ecologyhydro.sensitivity import candidate_rows

    crop_spec = yaml.load(crop_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    system = crop_spec["systems"][crop_spec["mapping"]["fine"]["2"]]
    spec = yaml.load(spec_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    cache = ArtifactCache(config.paths.cache / "m4/rotation")
    jobs, diagnostics = [], []
    static = [
        Path(index["aligned"]["lulc_fine"]),
        Path(index["routing"]["partitions"]) / "zones.tif",
    ]
    for path in static:
        verified_artifact(path)
    for year in config.study.calibration_years:
        monthly = [
            project_root() / index["recipe"]["pet_directory"] / f"PET{year % 100:02d}{m:02d}.tif"
            for m in range(1, 13)
        ]
        artifact = cache.build(
            "monthly_stats",
            [*monthly, *static, index_path, Path(__file__)],
            {"year": year},
            lambda out, files=monthly: monthly_statistics(index, files, out),
        )
        details = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))["details"]
        kc = weighted_kc(details["crop_monthly_sums"], system["monthly_kc"])
        with (artifact / "monthly_zones.csv").open(encoding="utf-8-sig") as stream:
            monthly_rows = list(csv.DictReader(stream))
        for station_id in range(1, 12):
            weights = [
                sum(
                    float(r["crop_et0_sum_mm"])
                    for r in monthly_rows
                    if int(r["month"]) == m and int(r["zone_id"]) <= station_id
                )
                for m in range(1, 13)
            ]
            diagnostics.append(
                {
                    "year": year,
                    "station_id": station_id,
                    "whole_domain_class_kc": kc,
                    "upstream_crop_weighted_kc": weighted_kc(weights, system["monthly_kc"])
                    if sum(weights)
                    else None,
                    "monthly_statistics": str(artifact / "monthly_zones.csv"),
                }
            )
        prepared = prepare_inputs(index_path, config.paths.cache, year, "fine", "full")
        baseline_kc = next(
            r["kc"] for r in candidate_rows(spec, "fine", "central") if r["lucode"] == 2
        )
        cases = {"baseline": baseline_kc, "rotation": kc}
        if year == 2019:
            cases.update(
                rotation_low=kc * system["kc_sensitivity_factors"]["low"],
                rotation_high=kc * system["kc_sensitivity_factors"]["high"],
            )
        for variant, value in cases.items():

            def make_table(out, coefficient=value, case=variant):
                records = candidate_rows(spec, "fine", "central")
                for row in records:
                    if row["lucode"] == 2 and case != "baseline":
                        row.update(
                            kc=coefficient,
                            root_depth=system["annual_root_depth_mm"],
                            profile="winter_wheat_summer_maize_scenario",
                            evidence="ET0-weighted assumed monthly rotation; fixed root proxy",
                        )
                write_csv(out / "biophysical.csv", records)
                return {"kc": coefficient, "only_lucode_2_changed": True}

            table = cache.build(
                "tables",
                [crop_path, spec_path, artifact / "manifest.json", Path(__file__)],
                {"year": year, "variant": variant, "kc": value},
                make_table,
            )
            scenario = {
                "name": variant,
                "landcover": "fine",
                "scope": "full",
                "overrides": {"biophysical_table_path": str(table / "biophysical.csv")},
                "year": year,
                "annual_kc": value,
                "status": "rotation_hypothesis_not_calibrated",
            }

            def make_scenario(out, record=scenario):
                write_json(out / "scenario.json", record)
                return record

            scenario_dir = cache.build(
                "scenarios",
                [table / "biophysical.csv", crop_path, Path(__file__)],
                scenario,
                make_scenario,
            )
            jobs.append(
                {
                    "year": year,
                    "landcover": "fine",
                    "scope": "full",
                    "z": 5,
                    "variant": variant,
                    "scenario": str(scenario_dir / "scenario.json"),
                    "prepared": prepared,
                    "annual_kc": value,
                }
            )
    return jobs, diagnostics


def summarize(batch, output):
    from ecologyhydro.aggregation import write_csv

    records = []
    for entry in batch["completed"]:
        directory = Path(entry["directory"])
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if run["status"] == "reused":
            directory = Path(run["reused_run_dir"])
            run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if not valid_completed(run):
            raise ValueError("Invalid rotation result")
        with (directory / "stations.csv").open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                records.append(
                    {
                        "year": int(row["year"]),
                        "station": row["station"],
                        "station_id": int(row["station_id"]),
                        "variant": entry["job"]["variant"],
                        "annual_kc": entry["job"]["annual_kc"],
                        "awy_yield_1e8_m3": float(row["natural_yield_1e8_m3"]),
                        "run_id": row["run_id"],
                    }
                )
    baseline = {
        (r["year"], r["station"]): r["awy_yield_1e8_m3"]
        for r in records
        if r["variant"] == "baseline"
    }
    for row in records:
        value = baseline[row["year"], row["station"]]
        row["change_from_baseline_percent"] = (
            (row["awy_yield_1e8_m3"] / value - 1) * 100 if value else None
        )
    if len(records) != len(batch["jobs"]) * 11:
        raise ValueError("Missing station outputs")
    write_csv(output / "stations.csv", records)


def main():
    root = project_root()
    config = load_config()
    configure_threads(config.resources)
    from osgeo import gdal

    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.trials import run_group

    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    index_path = root / "project/cache/m2/latest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    jobs, diagnostics = prepare(
        config,
        index,
        index_path,
        root / "config/crop_systems.yaml",
        root / "config/biophysical_candidates.yaml",
    )
    output = root / "project/m4/rotation"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "annual_kc_diagnostics.csv", diagnostics)
    write_json(output / "plan.json", {"jobs": jobs, "workers": 2, "calibrated": False})
    directory, batch = run_group(
        config, root / "config.yaml", index_path, jobs, 2, "rotation", batch_root=output / "runs"
    )
    summarize(batch, output)
    write_json(
        output / "summary.json",
        {
            "status": batch["status"],
            "batch": str(directory / "batch.json"),
            "tasks": len(jobs),
            "elapsed_seconds": batch["elapsed_seconds"],
            "peak_tree_rss_bytes": batch["peak_tree_rss_bytes"],
            "validation_used": False,
            "calibrated": False,
        },
    )


if __name__ == "__main__":
    main()
