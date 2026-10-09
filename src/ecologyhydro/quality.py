"""Spatial coverage, common masks and calibration-only precipitation diagnostics."""

import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.config import project_root
from ecologyhydro.spatial import OPTIONS, windows


def boundary_mask(boundary, template, output):
    with gdal.Open(str(template)) as reference, ogr.Open(str(boundary)) as original:
        source_layer = original.GetLayer()
        source_crs = source_layer.GetSpatialRef()
        source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_crs = reference.GetSpatialRef()
        target_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        options = osr.CoordinateTransformationOptions()
        options.SetBallparkAllowed(True)
        transform = osr.CreateCoordinateTransformation(source_crs, target_crs, options)
        memory = ogr.GetDriverByName("Memory").CreateDataSource("")
        layer = memory.CreateLayer("boundary", target_crs, ogr.wkbMultiPolygon)
        repaired = 0
        area = 0.0
        for feature in source_layer:
            geometry = feature.GetGeometryRef().Clone()
            if not geometry.IsValid():
                geometry = geometry.MakeValid()
                repaired += 1
            geometry.Transform(transform)
            area += geometry.GetArea()
            copied = ogr.Feature(layer.GetLayerDefn())
            copied.SetGeometry(geometry)
            layer.CreateFeature(copied)
        with gdal.GetDriverByName("GTiff").Create(
            str(output),
            reference.RasterXSize,
            reference.RasterYSize,
            1,
            gdal.GDT_Byte,
            options=OPTIONS,
        ) as mask:
            mask.SetGeoTransform(reference.GetGeoTransform())
            mask.SetProjection(reference.GetProjection())
            mask.GetRasterBand(1).SetNoDataValue(0)
            gdal.RasterizeLayer(mask, [1], layer, burn_values=[1])
    return {
        "area_km2": area / 1e6,
        "repaired_features": repaired,
        "datum": "unspecified Krassowsky datum; ballpark transformation, diagnostic only",
    }


def coverage(aligned, scope_path, class_tables, output):
    categories = {}
    for name, table in class_tables.items():
        with table.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            categories[f"lulc_{name}"] = {int(row[0]) for row in reader} - {0, 255}
    metrics = {name: {"invalid": 0, "zero": 0, "total": 0.0, "valid": 0} for name in aligned}
    class_counts = {name: Counter() for name in categories}
    scope_count = common_count = edge_count = 0
    with ExitStack() as stack:
        scope = stack.enter_context(gdal.Open(str(scope_path)))
        datasets = {
            name: stack.enter_context(gdal.Open(str(path))) for name, path in aligned.items()
        }
        for name, dataset in datasets.items():
            if (
                dataset.GetGeoTransform() != scope.GetGeoTransform()
                or dataset.GetProjection() != scope.GetProjection()
                or dataset.RasterXSize != scope.RasterXSize
                or dataset.RasterYSize != scope.RasterYSize
            ):
                raise ValueError(f"Misaligned raster: {name}")
        mask = stack.enter_context(
            gdal.GetDriverByName("GTiff").Create(
                str(output / "common_valid.tif"),
                scope.RasterXSize,
                scope.RasterYSize,
                1,
                gdal.GDT_Byte,
                options=OPTIONS,
            )
        )
        mask.SetGeoTransform(scope.GetGeoTransform())
        mask.SetProjection(scope.GetProjection())
        mask.GetRasterBand(1).SetNoDataValue(0)
        for window in windows(scope):
            inside = scope.ReadAsArray(*window) > 0
            x, y, width, height = window
            top, bottom = max(0, y - 1), min(scope.RasterYSize, y + height + 1)
            padded = np.zeros((height + 2, width + 2), dtype=bool)
            raw_mask = (
                datasets["dem"]
                .GetRasterBand(1)
                .GetMaskBand()
                .ReadAsArray(0, top, width, bottom - top)
                != 0
            )
            offset = 1 - (y - top)
            padded[offset : offset + len(raw_mask), 1:-1] = raw_mask
            touches_edge = np.zeros(inside.shape, dtype=bool)
            for dy in range(3):
                for dx in range(3):
                    touches_edge |= ~padded[dy : dy + height, dx : dx + width]
            edge_count += int((inside & touches_edge).sum())
            scope_count += int(inside.sum())
            common = inside.copy()
            for name, dataset in datasets.items():
                values = dataset.ReadAsArray(*window)
                valid = dataset.GetRasterBand(1).GetMaskBand().ReadAsArray(*window) != 0
                valid &= np.isfinite(values)
                if name in categories:
                    codes, counts = np.unique(values[inside & valid], return_counts=True)
                    class_counts[name].update(
                        dict(zip(map(int, codes), map(int, counts), strict=True))
                    )
                    valid &= np.isin(values, list(categories[name]))
                elif name == "pawc":
                    valid &= (values >= 0) & (values <= 1)
                elif name == "root_depth":
                    valid &= values > 0
                elif name != "dem":
                    valid &= values >= 0
                common &= valid
                selected = values[inside & valid]
                metrics[name]["valid"] += int(selected.size)
                metrics[name]["invalid"] += int((inside & ~valid).sum())
                metrics[name]["zero"] += int((selected == 0).sum())
                metrics[name]["total"] += float(selected.sum(dtype=np.float64))
            common_count += int(common.sum())
            mask.GetRasterBand(1).WriteArray(common.astype(np.uint8), window[0], window[1])
        pixel_area = abs(scope.GetGeoTransform()[1] * scope.GetGeoTransform()[5])
    if not scope_count:
        raise ValueError("Empty quality-check region")
    for name, metric in metrics.items():
        metric["invalid_fraction"] = metric["invalid"] / scope_count
        metric["mean"] = metric["total"] / metric["valid"] if metric["valid"] else None
        if name in class_counts:
            metric["class_counts"] = dict(class_counts[name])
            metric["undefined_codes"] = sorted(set(class_counts[name]) - categories[name])
            metric["mapped_classes_absent"] = sorted(categories[name] - set(class_counts[name]))
    return {
        "scope_pixels": scope_count,
        "scope_area_km2": scope_count * pixel_area / 1e6,
        "common_pixels": common_count,
        "common_area_km2": common_count * pixel_area / 1e6,
        "common_missing_fraction": 1 - common_count / scope_count,
        "dem_edge_touch_pixels": edge_count,
        "rasters": metrics,
        "mask_policy": "diagnostic common valid region; never substitutes complete watershed area",
    }


def precipitation_comparison(report, zones_path, observed_path, calibration_years, output):
    """Compare only calibration years; lower reach extent mismatch stays excluded."""
    from ecologyhydro.reference_checks import REGIONS

    with observed_path.open(encoding="utf-8-sig", newline="") as stream:
        observed = {row["水资源二级区"]: row for row in csv.DictReader(stream)}
    results = []
    # Station indices: Guide=2, Lanzhou=3, Toudaoguai=6, Longmen=7, Sanmenxia=8, Huayuankou=9.
    groups = REGIONS
    with gdal.Open(str(zones_path)) as zones:
        pixel_area = abs(zones.GetGeoTransform()[1] * zones.GetGeoTransform()[5])
        for variable in ("prec", "bcpr"):
            for year in calibration_years:
                totals = np.zeros(12)
                valid_counts = np.zeros(12, dtype=np.int64)
                counts = valid_counts.copy()
                with gdal.Open(report["aligned"][f"{variable}_{year}"]) as source:
                    for window in windows(zones):
                        ids = zones.ReadAsArray(*window)
                        data = source.ReadAsArray(*window)
                        valid = (ids > 0) & (
                            source.GetRasterBand(1).GetMaskBand().ReadAsArray(*window) != 0
                        )
                        valid &= np.isfinite(data) & (data >= 0)
                        counts += np.bincount(ids.ravel(), minlength=12)
                        valid_counts += np.bincount(ids[valid], minlength=12)
                        totals += np.bincount(ids[valid], weights=data[valid], minlength=12)
                for index, (region, start, end) in enumerate(groups):
                    expected = counts[start + 1 : end + 1].sum()
                    actual = valid_counts[start + 1 : end + 1].sum()
                    volume = totals[start + 1 : end + 1].sum() * pixel_area / 1e11
                    record = observed[region]
                    observation = float(record[str(year)])
                    results.append(
                        {
                            "region": region,
                            "year": year,
                            "variable": variable,
                            "volume_1e8_m3": volume,
                            "observed_1e8_m3": observation,
                            "relative_difference": (volume - observation) / observation,
                            "valid_fraction": float(actual / expected) if expected else 0,
                            "area_km2": float(expected * pixel_area / 1e6),
                            "comparable": index < 6 and actual == expected,
                            "limitation": "unverified statistical boundaries; no bias correction",
                        }
                    )
    with (output / "precipitation_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    scores = {
        variable: float(
            np.mean(
                [
                    abs(row["relative_difference"])
                    for row in results
                    if row["variable"] == variable and row["comparable"]
                ]
            )
        )
        for variable in ("prec", "bcpr")
    }
    if not all(np.isfinite(score) for score in scores.values()):
        raise ValueError("No usable calibration precipitation comparison")
    selected = report["recipe"].get("precipitation_baseline", "prec")
    if selected not in scores:
        raise ValueError("Unknown explicit precipitation baseline")
    return {
        "years_used": calibration_years,
        "trial_selection": selected,
        "mean_absolute_relative_volume_difference": scores,
        "selection_rule": "explicit recipe baseline; table differences are diagnostic only",
        "limitation": "provisional comparison; statistical boundaries differ; no rescaling applied",
        "validation_observations_used": False,
    }


def prepare_quality(config, report):
    root = project_root()
    cache = ArtifactCache(config.paths.cache / "m2")
    code = [
        Path(__file__),
        Path(__file__).with_name("spatial.py"),
        Path(__file__).with_name("cache.py"),
        Path(__file__).with_name("reference_checks.py"),
    ]
    boundary = root / report["recipe"]["boundary"]
    template = Path(report["aligned"]["dem"])
    mask = cache.build(
        "quality/boundary_mask",
        [boundary.with_suffix(s) for s in (".shp", ".shx", ".dbf", ".prj")] + [template, *code],
        {},
        lambda out: boundary_mask(boundary, template, out / "mask.tif"),
    )
    classes = {name: root / path for name, path in report["recipe"]["class_tables"].items()}
    scopes = {"approximate_boundary": mask / "mask.tif"}
    if "routing" in report:
        zones = Path(report["routing"]["partitions"]) / "zones.tif"
        if zones.is_file():
            scopes["watersheds"] = zones
    results = {}
    for name, scope in scopes.items():
        artifact = cache.build(
            f"quality/coverage/{name}",
            [scope, *map(Path, report["aligned"].values()), *classes.values(), *code],
            {},
            lambda out, region=scope: coverage(report["aligned"], region, classes, out),
        )
        results[name] = str(artifact)
    if "watersheds" in scopes:
        from ecologyhydro.resampling import compare_resampling

        native_climate = [
            Path(path)
            for name, path in report["native"].items()
            if name.startswith(("prec_", "bcpr_", "pet_"))
        ]
        resampled = cache.build(
            "quality/resampling",
            [
                scopes["watersheds"],
                *native_climate,
                *map(Path, report["aligned"].values()),
                Path(__file__).with_name("resampling.py"),
                *code,
            ],
            {},
            lambda out: compare_resampling(report, scopes["watersheds"], out),
        )
        results["resampling"] = str(resampled)
        observation = config.paths.hydrology / "降水量2018-2023.csv"
        artifact = cache.build(
            "quality/precipitation",
            [scopes["watersheds"], observation, *map(Path, report["aligned"].values()), *code],
            {
                "calibration_years": config.study.calibration_years,
                "precipitation_baseline": report["recipe"].get("precipitation_baseline", "prec"),
            },
            lambda out: precipitation_comparison(
                report, scopes["watersheds"], observation, config.study.calibration_years, out
            ),
        )
        results["precipitation"] = str(artifact)
        from ecologyhydro.reference_checks import reference_checks

        station_table = root / report["recipe"]["stations"]
        area_table = config.paths.hydrology / "水资源二级区面积.csv"
        thresholds = report["recipe"].get(
            "reference_review",
            {
                "area_relative_difference": 0.15,
                "precipitation_volume_relative_difference": 0.30,
                "precipitation_depth_relative_difference": 0.25,
            },
        )
        references = cache.build(
            "quality/reference_checks",
            [scopes["watersheds"], station_table, area_table, artifact / "manifest.json", *code],
            thresholds,
            lambda out: reference_checks(
                scopes["watersheds"],
                station_table,
                area_table,
                artifact / "precipitation_comparison.csv",
                thresholds,
                out,
            ),
        )
        results["references"] = str(references)
    readiness = {
        "engineering_ready": False,
        "formal_experiment_ready": False,
        "limitations": [
            "Biophysical coefficients are explicit engineering trial priors",
            "Root-restricting depth uses class-midpoint proxies",
            "Hydrography-conditioned catchments require sensitivity assessment",
            "Approximate boundary datum and precipitation-region mismatch retained",
        ],
    }
    if "watersheds" in results:
        coverage_report = json.loads(
            (Path(results["watersheds"]) / "manifest.json").read_text(encoding="utf-8")
        )["details"]
        readiness["engineering_ready"] = (
            coverage_report["common_missing_fraction"] == 0
            and coverage_report["dem_edge_touch_pixels"] == 0
            and "biophysical" in report
        )
        if report["recipe"].get("closed_lakes"):
            readiness["routing_constraints_passed"] = all(
                report.get(key, {}).get("passed", False)
                for key in ("closed_lake_acceptance", "inland_channel_acceptance")
            )
            readiness["engineering_ready"] &= readiness["routing_constraints_passed"]
        reference_report = json.loads(
            (Path(results["references"]) / "manifest.json").read_text(encoding="utf-8")
        )["details"]
        readiness["reference_review_required"] = reference_report["review_required"]
        readiness["reference_alerts"] = {
            key: reference_report[key]
            for key in ("station_alerts", "region_alerts", "precipitation_alert_rows")
        }
    report["readiness"] = readiness
    report["quality"] = results
    report["cache_events"].extend(cache.events)
    (cache.root / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results
