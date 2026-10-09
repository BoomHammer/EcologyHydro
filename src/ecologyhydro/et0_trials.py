"""Paired Tangnaihai AWY runs isolating reference ET hypotheses, without fitting."""

import csv
import json
from contextlib import ExitStack
from pathlib import Path

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.config import UniqueKeyLoader, load_config, project_root
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import valid_completed, write_json


def aligned_et(source, zones_path, output):
    import numpy as np
    from osgeo import gdal

    from ecologyhydro.spatial import NODATA, OPTIONS, windows

    with gdal.Open(str(zones_path)) as zones:
        gt = zones.GetGeoTransform()
        bounds = [
            gt[0],
            gt[3] + gt[5] * zones.RasterYSize,
            gt[0] + gt[1] * zones.RasterXSize,
            gt[3],
        ]
        with gdal.Warp(
            str(output / "et0.tif"),
            str(source),
            format="GTiff",
            dstSRS=zones.GetProjection(),
            outputBounds=bounds,
            width=zones.RasterXSize,
            height=zones.RasterYSize,
            resampleAlg="bilinear",
            errorThreshold=0,
            dstNodata=NODATA,
            outputType=gdal.GDT_Float32,
            creationOptions=OPTIONS,
        ) as target:
            for block in windows(zones):
                inside = zones.ReadAsArray(*block) > 0
                values = target.ReadAsArray(*block)
                valid = target.GetRasterBand(1).GetMaskBand().ReadAsArray(*block) > 0
                if np.any(inside & (~valid | ~np.isfinite(values) | (values < 0))):
                    raise ValueError("Invalid candidate ET0")
                values[~inside] = NODATA
                target.GetRasterBand(1).WriteArray(values, block[0], block[1])
    return {"method": "exact_transform_bilinear", "units": "mm/year"}


def yield_floor(args, zones_path):
    import numpy as np
    from osgeo import gdal

    from ecologyhydro.spatial import windows

    with Path(args["biophysical_table_path"]).open(encoding="utf-8-sig") as stream:
        lookup = {int(r["lucode"]): float(r["kc"]) for r in csv.DictReader(stream)}
    coefficients = np.full(max(lookup) + 1, np.nan)
    for code, value in lookup.items():
        coefficients[code] = value
    total, count, et_total, p_total = 0.0, 0, 0.0, 0.0
    with ExitStack() as stack:
        ds = {
            k: stack.enter_context(gdal.Open(str(p)))
            for k, p in {
                "zones": zones_path,
                "p": args["precipitation_path"],
                "et": args["eto_path"],
                "lulc": args["lulc_path"],
            }.items()
        }
        reference = ds["zones"]
        for dataset in ds.values():
            if (
                dataset.GetGeoTransform() != reference.GetGeoTransform()
                or not dataset.GetSpatialRef().IsSame(reference.GetSpatialRef())
                or (dataset.RasterXSize, dataset.RasterYSize)
                != (reference.RasterXSize, reference.RasterYSize)
            ):
                raise ValueError("Floor grid mismatch")
        for block in windows(reference):
            mask = reference.ReadAsArray(*block) > 0
            if not mask.any():
                continue
            arrays = {k: v.ReadAsArray(*block)[mask] for k, v in ds.items() if k != "zones"}
            for k in arrays:
                if not ds[k].GetRasterBand(1).GetMaskBand().ReadAsArray(*block)[mask].all():
                    raise ValueError("Incomplete floor input")
                if not np.isfinite(arrays[k]).all() or (arrays[k] < 0).any():
                    raise ValueError("Invalid negative/nonfinite floor input")
            codes = arrays["lulc"].astype(int)
            if (codes < 0).any() or (codes >= len(coefficients)).any():
                raise ValueError("Unknown floor class")
            bound = np.maximum(arrays["p"] - coefficients[codes] * arrays["et"], 0)
            if not np.isfinite(bound).all():
                raise ValueError("Invalid floor coefficients")
            total += bound.sum(dtype=np.float64)
            et_total += arrays["et"].sum(dtype=np.float64)
            p_total += arrays["p"].sum(dtype=np.float64)
            count += int(mask.sum())
        gt = reference.GetGeoTransform()
        area = abs(gt[1] * gt[5] - gt[2] * gt[4])
    return {
        "lower_bound_1e8_m3": total * area / 1e11,
        "et0_mean_mm": et_total / count,
        "precipitation_mm": p_total / count,
        "area_km2": count * area / 1e6,
    }


def main():
    root = project_root()
    config = load_config()
    configure_threads(config.resources)
    import yaml
    from osgeo import gdal

    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.model_inputs import prepare_inputs, verified_artifact
    from ecologyhydro.sensitivity import candidate_rows
    from ecologyhydro.trials import run_group

    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    index_path = root / "project/cache/m2/latest.json"
    review_path = root / "project/m4/climate_review/summary.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    spec_path = root / "config/biophysical_candidates.yaml"
    spec = yaml.load(spec_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    cache = ArtifactCache(config.paths.cache / "m4/et0_trials")
    output = root / "project/m4/et0_trials"
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for year in config.study.calibration_years:
        native = Path(next(r["native"] for r in review["artifacts"] if r["year"] == year))
        for landcover in ("fine", "copernicus"):
            prepared = prepare_inputs(index_path, config.paths.cache, year, landcover, "tangnaihai")
            rows = candidate_rows(spec, landcover, "central")
            table = cache.build(
                "tables",
                [spec_path, Path(__file__), Path(__file__).with_name("sensitivity.py")],
                {"landcover": landcover},
                lambda out, data=rows: write_csv(out / "biophysical.csv", data),
            )
            for variant in ("terraclimate", "pm_wind10m", "pm_wind2m"):
                args = {
                    **prepared["args"],
                    "biophysical_table_path": str(table / "biophysical.csv"),
                }
                if variant != "terraclimate":
                    source = native / f"{variant}.tif"
                    verified_artifact(source)
                    raster = cache.build(
                        "et0",
                        [source, Path(prepared["zones"]), Path(__file__)],
                        {"year": year, "variant": variant},
                        lambda out, src=source, z=prepared["zones"]: aligned_et(src, z, out),
                    )
                    args["eto_path"] = str(raster / "et0.tif")
                diagnostic = yield_floor(args, prepared["zones"])
                scenario = {
                    "name": variant,
                    "landcover": landcover,
                    "scope": "tangnaihai",
                    "year": year,
                    "et0_label": variant,
                    "overrides": {k: args[k] for k in ("biophysical_table_path", "eto_path")},
                    "status": "ET0_isolation_hypothesis_not_calibrated",
                }
                scenario_dir = cache.build(
                    "scenarios",
                    [Path(args["biophysical_table_path"]), Path(args["eto_path"]), Path(__file__)],
                    scenario,
                    lambda out, record=scenario: write_json(out / "scenario.json", record),
                )
                jobs.append(
                    {
                        "year": year,
                        "landcover": landcover,
                        "scope": "tangnaihai",
                        "z": 5,
                        "variant": variant,
                        "scenario": str(scenario_dir / "scenario.json"),
                        **diagnostic,
                    }
                )
    write_json(output / "plan.json", {"jobs": jobs, "workers": 2, "validation_used": False})
    directory, batch = run_group(
        config,
        root / "config.yaml",
        index_path,
        jobs,
        2,
        "et0_isolation",
        batch_root=output / "runs",
    )
    results = []
    for entry in batch["completed"]:
        run_dir = Path(entry["directory"])
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if run["status"] == "reused":
            run_dir = Path(run["reused_run_dir"])
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if not valid_completed(run):
            raise ValueError("Invalid completed climate trial")
        with (run_dir / "stations.csv").open(encoding="utf-8-sig") as stream:
            station_rows = list(csv.DictReader(stream))
        if len(station_rows) != 1 or station_rows[0]["station"] != "唐乃亥":
            raise ValueError("Isolation trial must contain Tangnaihai only")
        job, row = entry["job"], station_rows[0]
        volume = float(row["natural_yield_1e8_m3"])
        if volume + 0.01 < job["lower_bound_1e8_m3"]:
            raise ValueError("AWY output violates necessary lower bound")
        results.append(
            {
                "year": job["year"],
                "landcover": job["landcover"],
                "variant": job["variant"],
                "et0_mean_mm": job["et0_mean_mm"],
                "lower_bound_1e8_m3": job["lower_bound_1e8_m3"],
                "awy_yield_1e8_m3": volume,
                "observed_1e8_m3": row["observed_1e8_m3"],
                "observation_status": row["observation_status"],
                "run_id": row["run_id"],
            }
        )
    baseline = {
        (r["year"], r["landcover"]): r["awy_yield_1e8_m3"]
        for r in results
        if r["variant"] == "terraclimate"
    }
    for row in results:
        row["change_percent"] = (
            row["awy_yield_1e8_m3"] / baseline[row["year"], row["landcover"]] - 1
        ) * 100
    if len(results) != 24 or batch["status"] != "success":
        raise ValueError("Incomplete ET0 isolation suite")
    write_csv(output / "stations.csv", results)
    write_json(
        output / "summary.json",
        {
            "status": batch["status"],
            "batch": str(directory),
            "tasks": len(results),
            "elapsed_seconds": batch["elapsed_seconds"],
            "peak_tree_rss_bytes": batch["peak_tree_rss_bytes"],
            "validation_used": False,
            "calibrated": False,
        },
    )


if __name__ == "__main__":
    main()
