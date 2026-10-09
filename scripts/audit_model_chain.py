"""Independent numerical and source-to-model audit; no parameter fitting."""

import calendar
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date
from osgeo import gdal, ogr

from ecologyhydro.aggregation import write_csv
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.simulation import valid_completed, write_json
from ecologyhydro.spatial import crs, windows
from ecologyhydro.water_balance import read_csv


def independent_yield(p, et0, kc, root, soil, pawc, vegetated, z=5, omega_override=None):
    """Double-precision Fu equation, independently expressed in log space."""
    demand = et0 * kc
    omega = np.minimum(5, 1.25 + z * np.minimum(root, soil) * pawc / p)
    if omega_override is not None:
        omega = np.full_like(p, omega_override)
    ratio = demand / p
    log_ratio = np.log(np.maximum(ratio, np.finfo(float).tiny))
    aet_ratio = 1 + ratio - np.exp(np.logaddexp(0, omega * log_ratio) / omega)
    aet = np.where(vegetated == 1, np.minimum(ratio, aet_ratio), np.minimum(ratio, 1))
    return p * (1 - aet)


def raw_climate_check(root, index):
    checks = []
    for year in (2019, 2020, 2021, 2022):
        with gdal.Open(index["native"][f"prec_{year}"]) as target:
            gt = target.GetGeoTransform()
            lon = gt[0] + (np.arange(target.RasterXSize) + 0.5) * gt[1]
            lat = gt[3] + (np.arange(target.RasterYSize) + 0.5) * gt[5]
            source = next((root / index["recipe"]["climate_directory"]).glob("prec_*.nc"))
            with Dataset(source) as ds:
                xs = np.abs(ds["lon"][:][:, None] - lon).argmin(axis=0)
                ys = np.abs(ds["lat"][:][:, None] - lat).argmin(axis=0)
                t = ds["time"]
                dates = num2date(t[:], t.units, calendar=t.calendar)
                selected = [(i, date) for i, date in enumerate(dates) if date.year == year]
                if sorted(d.month for _, d in selected) != list(range(1, 13)):
                    raise ValueError("Missing or duplicate source month")
                total = np.zeros((len(ys), len(xs)), dtype=float)
                for i, date in selected:
                    raw = np.ma.asarray(ds["prec"][i, ys, xs], dtype=float).filled(np.nan)
                    total += raw * calendar.monthrange(year, date.month)[1] * 86400
            values = target.ReadAsArray()
            valid = target.GetRasterBand(1).GetMaskBand().ReadAsArray() > 0
            error = float(np.max(np.abs(values[valid] - total[valid])))
            if not np.isfinite(error) or error > 0.001:
                raise ValueError("Raw precipitation differs from cached annual raster")
            checks.append({"year": year, "variable": "prec", "max_error_mm": error})
        with gdal.Open(index["native"][f"pet_{year}"]) as target:
            total = np.zeros((target.RasterYSize, target.RasterXSize), dtype=float)
            for month in range(1, 13):
                source = (
                    root / index["recipe"]["pet_directory"] / f"PET{year % 100:02}{month:02}.tif"
                )
                with gdal.Open(str(source)) as ds:
                    if ds.GetGeoTransform() != target.GetGeoTransform():
                        raise ValueError("PET monthly/annual transform differs")
                    total += ds.ReadAsArray().astype(float) / 10
            values = target.ReadAsArray()
            valid = target.GetRasterBand(1).GetMaskBand().ReadAsArray() > 0
            error = float(np.max(np.abs(values[valid] - total[valid])))
            if not np.isfinite(error) or error > 0.001:
                raise ValueError("Raw PET differs from cached annual raster")
            checks.append({"year": year, "variable": "pet", "max_error_mm": error})
    return checks


def main():
    started = time.perf_counter()
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    root = project_root()
    output = root / "project/diagnostics/deep_review"
    output.mkdir(parents=True, exist_ok=True)
    index_path = root / "project/cache/m2/latest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    climate_checks = raw_climate_check(root, index)
    summary = json.loads((root / "project/m4/rotation/summary.json").read_text(encoding="utf-8"))
    batch = json.loads(Path(summary["batch"]).read_text(encoding="utf-8"))
    entry = next(
        e
        for e in batch["completed"]
        if e["job"]["year"] == 2019 and e["job"]["variant"] == "rotation"
    )
    run_dir = Path(entry["directory"])
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] == "reused":
        run_dir = Path(run["reused_run_dir"])
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if not valid_completed(run):
        raise ValueError("Existing official results failed checksum verification")
    snapshot = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    registry = json.loads((run_dir / "registry.json").read_text(encoding="utf-8"))
    keys = (
        "precip",
        "eto",
        "kc_raster",
        "root_depth",
        "depth_to_root_rest_layer",
        "pawc",
        "veg",
        "wyield",
    )
    paths = {k: registry[k] for k in keys}
    paths["zones"] = snapshot["prepared"]["zones"]
    variants = (
        "baseline_independent",
        "soil_total_mm_interpretation",
        "kc1_diagnostic",
        "omega5_envelope",
        "soil_total_mm_and_kc1",
        "kc1_and_omega5_envelope",
        "exact_precipitation_warp",
    )
    counts, sums = np.zeros(12), {k: np.zeros(12) for k in variants}
    max_formula_error = 0.0
    input_differences = {}
    spatial_samples = []
    with ExitStack() as stack:
        datasets = {k: stack.enter_context(gdal.Open(str(p))) for k, p in paths.items()}
        zones = datasets["zones"]
        gt = zones.GetGeoTransform()
        exact_precipitation = stack.enter_context(
            gdal.Warp(
                "",
                index["native"]["prec_2019"],
                format="VRT",
                dstSRS=zones.GetProjection(),
                outputBounds=[
                    gt[0],
                    gt[3] + zones.RasterYSize * gt[5],
                    gt[0] + zones.RasterXSize * gt[1],
                    gt[3],
                ],
                width=zones.RasterXSize,
                height=zones.RasterYSize,
                resampleAlg="bilinear",
                errorThreshold=0,
                outputType=gdal.GDT_Float32,
            )
        )
        pixel_area = abs(zones.GetGeoTransform()[1] * zones.GetGeoTransform()[5])
        for ds in datasets.values():
            if ds.GetGeoTransform() != zones.GetGeoTransform():
                raise ValueError("Official intermediate grids do not align")
        for block in windows(zones):
            ids = zones.ReadAsArray(*block)
            valid = ids > 0
            if not valid.any():
                continue
            data = {
                k: ds.ReadAsArray(*block)[valid].astype(float)
                for k, ds in datasets.items()
                if k != "zones"
            }
            if any(not np.isfinite(v).all() for v in data.values()):
                raise ValueError("Non-finite official input")
            args = [data[k] for k in keys[:7]]
            base = independent_yield(*args)
            max_formula_error = max(max_formula_error, float(np.max(np.abs(base - data["wyield"]))))
            # Conditional diagnostic only: assume SMU AWC is profile total and
            # a uniform profile using the SAME existing depth proxy. This does
            # not establish that midpoint depth or uniform PAWC is correct.
            soil_args = args.copy()
            soil_args[5] = args[5] * 1000 / args[4]
            kc_args = args.copy()
            kc_args[2] = np.ones_like(args[2])
            both = soil_args.copy()
            both[2] = kc_args[2]
            exact_args = args.copy()
            exact_args[0] = exact_precipitation.ReadAsArray(*block)[valid].astype(float)
            values = (
                base,
                independent_yield(*soil_args),
                independent_yield(*kc_args),
                independent_yield(*args, omega_override=5),
                independent_yield(*both),
                independent_yield(*kc_args, omega_override=5),
                independent_yield(*exact_args),
            )
            counts += np.bincount(ids[valid], minlength=12)
            for name, value in zip(variants, values, strict=True):
                sums[name] += np.bincount(ids[valid], weights=value, minlength=12)
        if max_formula_error > 0.003:
            raise ValueError(f"Independent AWY formula mismatch: {max_formula_error}")
        # Check exact values passed to official model, beyond matched filenames.
        for key, argument in (
            ("precip", "precipitation_path"),
            ("eto", "eto_path"),
            ("pawc", "pawc_path"),
            ("depth_to_root_rest_layer", "depth_to_root_rest_layer_path"),
        ):
            original = stack.enter_context(gdal.Open(snapshot["model_args"][argument]))
            maximum = 0.0
            for block in windows(zones):
                valid = zones.ReadAsArray(*block) > 0
                if valid.any():
                    difference = (
                        original.ReadAsArray(*block)[valid]
                        - datasets[key].ReadAsArray(*block)[valid]
                    )
                    maximum = max(maximum, float(np.max(np.abs(difference))))
            input_differences[key] = maximum
        # Independent spatial check: transform pixel centers back to lon/lat,
        # bilinearly sample the native precipitation raster without gdal.Warp.
        native = stack.enter_context(gdal.Open(index["native"]["prec_2019"]))
        native_values = native.ReadAsArray()
        from osgeo import osr

        transform = osr.CoordinateTransformation(crs(zones.GetProjection()), crs("EPSG:4326"))
        inverse = gdal.InvGeoTransform(native.GetGeoTransform())
        for row in range(0, zones.RasterYSize, 71):
            for col in range(0, zones.RasterXSize, 83):
                if zones.ReadAsArray(col, row, 1, 1)[0, 0] == 0:
                    continue
                x, y = gdal.ApplyGeoTransform(zones.GetGeoTransform(), col + 0.5, row + 0.5)
                lon, lat, _ = transform.TransformPoint(x, y)
                px, py = gdal.ApplyGeoTransform(inverse, lon, lat)
                px, py = px - 0.5, py - 0.5
                ix, iy = int(np.floor(px)), int(np.floor(py))
                dx, dy = px - ix, py - iy
                patch = native_values[iy : iy + 2, ix : ix + 2].astype(float)
                if patch.shape != (2, 2) or (patch < 0).any():
                    continue
                sampled = (patch[0, 0] * (1 - dx) + patch[0, 1] * dx) * (1 - dy) + (
                    patch[1, 0] * (1 - dx) + patch[1, 1] * dx
                ) * dy
                actual = float(datasets["precip"].ReadAsArray(col, row, 1, 1)[0, 0])
                spatial_samples.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "raw_bilinear_mm": sampled,
                        "official_mm": actual,
                        "difference_mm": actual - sampled,
                    }
                )
    stations = read_csv(run_dir / "stations.csv")
    results = []
    for row in stations:
        identifier = int(row["station_id"])
        for name in variants:
            volume = sums[name][1 : identifier + 1].sum() * pixel_area / 1e11
            results.append(
                {
                    "station": row["station"],
                    "variant": name,
                    "year": 2019,
                    "yield_1e8_m3": volume,
                    "observed_1e8_m3": row["observed_1e8_m3"],
                    "change_percent": (volume / float(row["natural_yield_1e8_m3"]) - 1) * 100,
                }
            )
    write_csv(output / "counterfactuals.csv", results)
    write_csv(output / "spatial_samples.csv", spatial_samples)
    matches = read_csv(Path(index["routing"]["mainstem"]) / "stations.csv")
    matched = {int(row["river_id"]): row for row in matches}
    network_rows = []
    with ogr.Open(str(root / index["recipe"]["river_network"])) as ds:
        for feature in ds.GetLayer():
            identifier = int(feature["HYRIV_ID"])
            if identifier in matched:
                match = matched[identifier]
                i = int(match["ws_id"])
                network_rows.append(
                    {
                        "station": match["station"],
                        "model_km2": counts[1 : i + 1].sum() * pixel_area / 1e6,
                        "hydrorivers_upland_km2": feature["UPLAND_SKM"],
                        "user_reference_km2": match["reference_km2"],
                    }
                )
    write_csv(output / "network_area_check.csv", network_rows)
    record = {
        "official_run_dir": str(run_dir),
        "official_outputs_checksums_valid": True,
        "climate_recomputed_from_raw": climate_checks,
        "independent_formula_max_error_mm": max_formula_error,
        "official_input_max_absolute_differences": input_differences,
        "spatial_samples": len(spatial_samples),
        "spatial_max_error_mm": max(abs(r["difference_mm"]) for r in spatial_samples),
        "scope": "2019 fine rotation; controlled diagnostic interventions, not calibration",
        "soil_intervention": (
            "Post-warp PAWC/depth ratio; provisional uniform-profile approximation"
        ),
        "validation_observations_used": False,
        "duration_seconds": time.perf_counter() - started,
        "sources": [
            fingerprint(Path(__file__)),
            fingerprint(index_path),
            fingerprint(run_dir / "snapshot.json"),
            fingerprint(run_dir / "registry.json"),
        ],
    }
    write_json(output / "model_chain.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("Tangnaihai", [r for r in results if r["station"] == "唐乃亥"])
    print("Areas", network_rows)


if __name__ == "__main__":
    main()
