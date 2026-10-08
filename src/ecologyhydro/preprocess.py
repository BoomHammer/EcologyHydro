"""M2 orchestrator. Static, land-cover and climate artifacts have separate keys."""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from osgeo import gdal

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.climate import annual_pet, annual_precipitation, crop_netcdf
from ecologyhydro.config import UniqueKeyLoader, load_config, project_root
from ecologyhydro.logging_utils import configure_logging
from ecologyhydro.runtime import configure_threads
from ecologyhydro.spatial import ALBERS, NODATA, bounds, crop_soil, make_grid, warp

LOGGER = logging.getLogger(__name__)


def code_inputs():
    return sorted(Path(__file__).parent.glob("*.py"))


def soil_parameters(source, table, depth_lookup, output):
    from ecologyhydro.spatial import write_raster

    depth_lookup = {int(key): value for key, value in depth_lookup.items()}
    pawc = np.full(65536, NODATA, dtype=np.float32)
    depth = pawc.copy()
    seen = set()
    with table.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            code = int(row["HWSD2_SMU_ID"])
            if code in seen:
                raise ValueError(f"Duplicate SMU: {code}")
            seen.add(code)
            awc = float(row["AWC"] or "nan")
            root_class = int(row["ROOT_DEPTH"] or 0)
            if np.isfinite(awc) and 0 <= awc <= 1000 and root_class in depth_lookup:
                pawc[code] = awc / 1000
                depth[code] = depth_lookup[root_class]
    with gdal.Open(str(source)) as dataset:
        codes = dataset.ReadAsArray()
        for name, lookup in (("pawc", pawc), ("root_depth", depth)):
            write_raster(
                output / f"{name}.tif",
                lookup[codes],
                dataset.GetGeoTransform(),
                dataset.GetProjection(),
            )
    return {
        "pawc_method": "HWSD2_SMU representative AWC (mm/m) / 1000; no duplicate coarse correction",
        "depth_proxy_mm": depth_lookup,
        "depth_status": "provisional, not measured depth",
        "unmapped_codes": [int(c) for c in np.unique(codes) if pawc[c] == NODATA],
    }


def export_soil_table(database, output):
    import sys

    sys.path.insert(0, str(project_root() / ".tools/mdb-reader"))
    from access_parser import AccessParser

    table = AccessParser(str(database)).parse_table("HWSD2_SMU")
    with (output / "smu.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(table)
        writer.writerows(zip(*table.values(), strict=True))


def prepare(config, recipe_path, stages):
    started = time.perf_counter()
    root = project_root()
    recipe = yaml.load(recipe_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if recipe["projected_crs"] != ALBERS:
        raise ValueError("Only the documented equal-area Albers grid is currently supported")
    resolution = float(recipe["resolution_m"])
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution_m must be positive")
    cache = ArtifactCache(config.paths.cache / "m2")
    # Code identity is scoped by module, so climate edits do not invalidate DEM routing.
    directory = Path(__file__).parent
    spatial_code = [directory / "spatial.py", directory / "cache.py"]
    climate_code = [directory / "climate.py", directory / "cache.py"]
    reference = root / recipe["extent_reference"]
    extent = bounds(reference, "EPSG:4326")
    grid = make_grid(reference, resolution)
    grid_dir = cache.build(
        "grid",
        [*spatial_code],
        {"grid": grid, "extent": extent},
        lambda out: (out / "grid.json").write_text(json.dumps(grid, indent=2)) and {},
    )
    paths = {"grid": str(grid_dir / "grid.json")}
    if "climate" in stages:
        for source in sorted((root / recipe["climate_directory"]).glob("*.nc")):
            variable = source.name.split("_")[0]
            subset = cache.build(
                f"climate/crop/{variable}",
                [source, *climate_code],
                {"bounds": extent},
                lambda out, src=source: crop_netcdf(src, out / "monthly.nc", extent),
            )
            paths[f"monthly_{variable}"] = str(subset / "monthly.nc")
            if variable not in {"prec", "bcpr"}:
                continue
            for year in sorted(config.study.calibration_years + config.study.validation_years):
                native = cache.build(
                    f"climate/annual/{variable}/{year}",
                    [subset / "manifest.json", *climate_code],
                    {"year": year},
                    lambda out, yr=year, var=variable, src=subset / "monthly.nc": (
                        annual_precipitation(src, var, yr, out / "annual.tif")
                    ),
                )
                paths[f"{variable}_{year}"] = str(native / "annual.tif")
        for year in sorted(config.study.calibration_years + config.study.validation_years):
            monthly = [
                root / recipe["pet_directory"] / f"PET{year % 100:02d}{m:02d}.tif"
                for m in range(1, 13)
            ]
            native = cache.build(
                f"climate/annual/pet/{year}",
                [*monthly, *climate_code],
                {"year": year, "scale": 0.1},
                lambda out, files=monthly: annual_pet(files, out / "annual.tif"),
            )
            paths[f"pet_{year}"] = str(native / "annual.tif")
    if "static" in stages:
        source = root / recipe["soil"]
        soil = cache.build(
            "static/soil_crop",
            [source, source.with_suffix(".hdr"), source.with_suffix(".blw"), *spatial_code],
            {"bounds": extent, "soil_crs": recipe["soil_crs_assumption"]},
            lambda out: crop_soil(source, out / "codes.tif", extent),
        )
        table = cache.build(
            "static/soil_table",
            [root / recipe["soil_database"], directory / "preprocess.py"],
            {"table": "HWSD2_SMU"},
            lambda out: export_soil_table(root / recipe["soil_database"], out),
        )
        parameters = cache.build(
            "static/soil_parameters",
            [soil / "manifest.json", table / "manifest.json", directory / "preprocess.py"],
            {"depth": recipe["soil_depth_proxy_mm"]},
            lambda out: soil_parameters(
                soil / "codes.tif", table / "smu.csv", recipe["soil_depth_proxy_mm"], out
            ),
        )
        for name in ("pawc", "root_depth"):
            paths[name] = str(parameters / f"{name}.tif")
        paths["dem"] = str(root / recipe["dem"])
    if "landcover" in stages:
        from ecologyhydro.biophysical import export_tables

        tables = {name: root / value for name, value in recipe["class_tables"].items()}
        priors = root / "config/biophysical_priors.yaml"
        parameters = cache.build(
            "landcover/biophysical",
            [priors, *tables.values(), directory / "biophysical.py", directory / "cache.py"],
            {},
            lambda out: export_tables(priors, tables, out),
        )
        for name, path in recipe["landcover"].items():
            paths[f"lulc_{name}"] = str(root / path)
    aligned = {}
    for name, path in paths.copy().items():
        if name == "grid" or name.startswith("monthly_"):
            continue
        source = Path(path)
        category = name.startswith("lulc_")
        method = "near" if category else "bilinear"
        artifact = cache.build(
            f"aligned/{name}",
            [source, *spatial_code],
            {"grid": grid, "method": method},
            lambda out, src=source, alg=method: warp(src, out / "data.tif", grid, alg),
        )
        aligned[name] = str(artifact / "data.tif")
        LOGGER.info("Aligned %s", name)
    report = {
        "recipe": recipe,
        "grid": grid,
        "geographic_bounds": extent,
        "native": paths,
        "aligned": aligned,
        "cache_events": cache.events,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "preprocessed; scientific and watershed acceptance checked separately",
    }
    if "landcover" in stages:
        report["biophysical"] = str(parameters)
    report_path = cache.root / "latest.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous["grid"] == grid and previous["recipe"] == recipe:
            report["native"] = previous["native"] | paths
            report["aligned"] = previous["aligned"] | aligned
            if "biophysical" not in report and "biophysical" in previous:
                report["biophysical"] = previous["biophysical"]
            if not aligned and "routing" in previous:
                report["routing"] = previous["routing"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="M2 cached preprocessing")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--recipe", type=Path, default=Path("config/m2.yaml"))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["static", "climate", "landcover", "routing", "quality"],
        default=["static", "climate", "landcover", "routing", "quality"],
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    configure_threads(config.resources)
    configure_logging(config.paths.logs / "m2.log", config.log_level)
    gdal.UseExceptions()
    gdal.SetCacheMax(256 * 1024 * 1024)
    report = prepare(config, args.recipe, args.stages)
    if "routing" in args.stages:
        from ecologyhydro.watersheds import prepare_watersheds

        prepare_watersheds(config, report)
    if "quality" in args.stages:
        from ecologyhydro.quality import prepare_quality

        prepare_quality(config, report)
    runs = config.paths.cache / "m2/runs"
    runs.mkdir(exist_ok=True)
    (runs / f"{time.time_ns()}_{uuid4().hex[:8]}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("M2 stage execution complete")


if __name__ == "__main__":
    main()
