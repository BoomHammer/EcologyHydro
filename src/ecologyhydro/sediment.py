"""Reproducible SDR preparation and hillslope-to-stream sediment accounting."""

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from osgeo import gdal, ogr

from ecologyhydro.aggregation import write_csv
from ecologyhydro.cache import ArtifactCache, fingerprint
from ecologyhydro.config import Resources
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import OPTIONS, warp, windows

LOGGER = logging.getLogger(__name__)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def c_parameters(settings, legend_path, crop_path, noncrop_path):
    """Map source-table IDs to verified fine labels; never interpret No data as zero."""
    crops = {int(row["n"]): float(row["Ccrop"]) for row in read_csv(crop_path)}
    noncrops = {int(row["n"]): row for row in read_csv(noncrop_path)}
    position = settings["policy"]["noncrop_range_position"]
    if not 0 <= position <= 1:
        raise ValueError("Noncrop range position must be in [0, 1]")
    legend = {int(row["数字"]): row for row in read_csv(legend_path)}
    mapping = settings["fine_c"]
    if set(legend) != set(mapping):
        raise ValueError("SDR crosswalk and current fine legend have different codes")
    result = []
    for code, spec in mapping.items():
        row = legend[code]
        label = next(row[key] for key in ("四级类", "三级类", "二级类", "一级类") if row[key])
        if label != spec["label"]:
            raise ValueError(f"Stale SDR mapping for code {code}: {label}")
        lower = upper = None
        if "crop" in spec:
            value = crops[spec["crop"]]
            source = f"table2:n={spec['crop']}"
        elif "crop_mix" in spec:
            if not np.isclose(sum(spec["crop_mix"].values()), 1):
                raise ValueError("Crop mixture weights must sum to one")
            value = sum(crops[key] * weight for key, weight in spec["crop_mix"].items())
            source = f"table2:mixture={spec['crop_mix']}"
        elif "noncrop" in spec:
            text = next(v for k, v in noncrops[spec["noncrop"]].items() if k.startswith("CNAn"))
            if text == "No data":
                raise ValueError("No data requires an explicit, documented proxy")
            lower, upper = sorted(float(x) for x in text.replace("–", "-").split("-"))
            value = lower + position * (upper - lower)
            source = f"table3:n={spec['noncrop']}; range_position={position}"
        else:
            value = spec["value"]
            source = "explicit_model_assumption"
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid C for {code}")
        result.append(
            {
                "lucode": code,
                "label": label,
                "usle_c": value,
                "c_lower": lower,
                "c_upper": upper,
                "c_source": source,
                "assumption": spec.get("assumption", ""),
            }
        )
    return result


def class_p_statistics(lulc_path, p_path, zones_path, codes):
    """Equal-area zonal mean P, with missing P explicitly replaced by no practice."""
    size = max(codes) + 1
    count = np.zeros(size, dtype=np.int64)
    valid_count = count.copy()
    total = np.zeros(size, dtype=np.float64)
    with ExitStack() as stack:
        lulc, practice, zones = [
            stack.enter_context(gdal.Open(str(path))) for path in (lulc_path, p_path, zones_path)
        ]
        for window in windows(lulc):
            inside = zones.ReadAsArray(*window) > 0
            values = lulc.ReadAsArray(*window)
            if not np.isin(values[inside], codes).all():
                raise ValueError("Missing or unmapped LULC inside station watersheds")
            values = values[inside].astype(int)
            p = practice.ReadAsArray(*window)[inside]
            valid = practice.GetRasterBand(1).GetMaskBand().ReadAsArray(*window)[inside] != 0
            valid &= np.isfinite(p)
            if np.any(valid & ((p < 0) | (p > 1))):
                raise ValueError("P outside [0, 1]; verify source scaling")
            count += np.bincount(values, minlength=size)
            valid_count += np.bincount(values[valid], minlength=size)
            total += np.bincount(values, weights=np.where(valid, p, 1), minlength=size)
    return {
        code: {
            "usle_p": float(total[code] / count[code]) if count[code] else 1.0,
            "class_pixels": int(count[code]),
            "p_valid_pixels": int(valid_count[code]),
            "p_missing_pixels": int(count[code] - valid_count[code]),
            "p_method": "equal_area_class_mean_missing_p1",
        }
        for code in codes
    }


def prepare_inputs(root, settings):
    protocol = yaml.safe_load((root / settings["comparison_config"]).read_text(encoding="utf-8"))
    index_path = root / protocol["paths"]["grid_index"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    grid = index["grid"]
    fine = protocol["products"]["fine"]
    lulc = root / fine["aligned"]
    zones = root / protocol["paths"]["zones"]
    # Unburned terrain: artificial river incision must not inflate RUSLE slopes.
    dem = Path(index["aligned"]["dem"])
    cache = ArtifactCache(root / settings["cache"])

    def align(key):
        source = root / settings[key]
        folder = cache.build(
            key,
            [source],
            {"grid": grid, "resampling": "average", "multiplier": 1.0},
            lambda out: warp(source, out / "data.tif", grid, method="average"),
        )
        return key, folder / "data.tif"

    # GDAL performs independent block-based warps; two concurrent jobs cap RAM/I/O.
    with ThreadPoolExecutor(max_workers=2) as pool:
        aligned = dict(pool.map(align, ["erosivity", "erodibility", "support_practice"]))
    rows = c_parameters(
        settings,
        root / fine["classes"],
        root / settings["crop_c"],
        root / settings["noncrop_c"],
    )
    p_stats = class_p_statistics(
        lulc, aligned["support_practice"], zones, [r["lucode"] for r in rows]
    )
    rows = [{**row, **p_stats[row["lucode"]]} for row in rows]
    zero_codes = [row["lucode"] for row in rows if row["usle_c"] == 0]

    def repair_and_audit(out):
        audit = {}
        with ExitStack() as stack:
            zone_ds = stack.enter_context(gdal.Open(str(zones)))
            lulc_ds = stack.enter_context(gdal.Open(str(lulc)))
            dem_ds = stack.enter_context(gdal.Open(str(dem)))
            for key in ("erosivity", "erodibility"):
                source = stack.enter_context(gdal.Open(str(aligned[key])))
                target = stack.enter_context(
                    gdal.GetDriverByName("GTiff").CreateCopy(
                        str(out / f"{key}.tif"),
                        source,
                        options=OPTIONS,
                    )
                )
                nearest_k = distances = None
                if key == "erodibility":
                    from scipy.ndimage import distance_transform_edt

                    original = source.ReadAsArray()
                    source_valid = source.GetRasterBand(1).GetMaskBand().ReadAsArray() != 0
                    source_valid &= np.isfinite(original) & (original >= 0)
                    distances, indices = distance_transform_edt(
                        ~source_valid,
                        sampling=grid["resolution"],
                        return_indices=True,
                    )
                    nearest_k = original[tuple(indices)]
                    del original, indices, source_valid
                count = missing = repaired = 0
                interpolated = 0
                max_distance = 0.0
                minimum, maximum, total = np.inf, -np.inf, 0.0
                for window in windows(source):
                    inside = zone_ds.ReadAsArray(*window) > 0
                    terrain_valid = dem_ds.GetRasterBand(1).GetMaskBand().ReadAsArray(*window) != 0
                    if np.any(inside & ~terrain_valid):
                        raise ValueError("DEM missing inside watershed")
                    data = source.ReadAsArray(*window)
                    valid = source.GetRasterBand(1).GetMaskBand().ReadAsArray(*window) != 0
                    valid &= np.isfinite(data) & (data >= 0)
                    gap = inside & ~valid
                    allowed = gap & np.isin(lulc_ds.ReadAsArray(*window), zero_codes)
                    missing += int(gap.sum())
                    repaired += int(allowed.sum())
                    soil_gap = gap & ~allowed
                    if key == "erodibility" and soil_gap.any():
                        y, height = window[1], window[3]
                        local_distance = distances[y : y + height]
                        distance = float(local_distance[soil_gap].max())
                        if distance > settings["policy"]["k_gap_max_distance_m"]:
                            raise ValueError(f"K gap exceeds permitted interpolation: {distance} m")
                        data[soil_gap] = nearest_k[y : y + height][soil_gap]
                        interpolated += int(soil_gap.sum())
                        max_distance = max(max_distance, distance)
                    elif soil_gap.any():
                        raise ValueError(
                            f"{key}: missing/negative input on erodible land: "
                            f"{int((gap & ~allowed).sum())} pixels in {window}"
                        )
                    # Only non-eroding surfaces may receive neutral zeros for missing R/K.
                    data[allowed] = 0
                    target.GetRasterBand(1).WriteArray(data, window[0], window[1])
                    selected = data[inside]
                    if selected.size:
                        count += selected.size
                        total += float(selected.sum(dtype=np.float64))
                        minimum = min(minimum, float(selected.min()))
                        maximum = max(maximum, float(selected.max()))
                audit[key] = {
                    "pixels": count,
                    "missing_source": missing,
                    "zero_on_non_eroding": repaired,
                    "nearest_valid_soil_pixels": interpolated,
                    "max_imputation_distance_m": max_distance,
                    "min": minimum,
                    "max": maximum,
                    "mean": total / count,
                    "multiplier": 1.0,
                }
        write_json(out / "audit.json", audit)
        return audit

    cleaned = cache.build(
        "validated_rk",
        [aligned["erosivity"], aligned["erodibility"], lulc, zones, dem],
        {
            "zero_c_codes": zero_codes,
            "policy": "bounded_nearest_soil_v2",
            "k_gap_max_distance_m": settings["policy"]["k_gap_max_distance_m"],
        },
        repair_and_audit,
    )
    sources = [
        root / settings["comparison_config"],
        index_path,
        lulc,
        dem,
        zones,
        root / fine["classes"],
        root / settings["crop_c"],
        root / settings["noncrop_c"],
        root / settings["watersheds"],
        root / settings["zone_station"],
    ]
    sources += [root / settings[key] for key in ("erosivity", "erodibility", "support_practice")]
    return {
        "rows": rows,
        "dem": dem,
        "lulc": lulc,
        "zones": zones,
        "r": cleaned / "erosivity.tif",
        "k": cleaned / "erodibility.tif",
        "audit": json.loads((cleaned / "audit.json").read_text(encoding="utf-8")),
        "sources": [fingerprint(p) for p in sources],
        "cache_events": cache.events,
        "grid": grid,
    }


def observation_rows(path, unit, station_totals):
    factors = {"t/yr": 1.0, "10000_t/yr": 1e4, "100million_t/yr": 1e8}
    if unit not in {*factors, "unknown"}:
        raise ValueError(f"Unsupported observation unit: {unit}")
    result = []
    for record in read_csv(path):
        station = record["测站"]
        for year, raw in record.items():
            if not year.isdigit() or len(year) != 4:
                continue
            value = float(raw) if raw.strip() else None
            valid = value is not None and np.isfinite(value) and value >= 0
            result.append(
                {
                    "station": station,
                    "year": int(year),
                    "observed_raw": raw,
                    "observed_unit": unit,
                    "observed_t_yr": value * factors[unit] if valid and unit in factors else None,
                    "observation_status": "valid" if valid else "missing_or_invalid",
                    "climatological_hillslope_delivery_t_yr": station_totals.get(station),
                    "comparison_status": "reference_only_different_processes_and_period"
                    if station in station_totals
                    else "no_existing_station_catchment",
                }
            )
    return result


def sediment_coverage(workspace, root, settings):
    """Audit missing SDR output against the entire water-model reporting domain."""
    protocol = yaml.safe_load((root / settings["comparison_config"]).read_text(encoding="utf-8"))
    grid = json.loads((root / protocol["paths"]["grid_index"]).read_text(encoding="utf-8"))["grid"]
    counts = np.zeros(256, dtype=np.int64)
    valid_counts = counts.copy()
    stream_counts = counts.copy()
    totals = np.zeros(256, dtype=np.float64)
    with (
        gdal.Open(str(root / protocol["paths"]["zones"])) as zones,
        gdal.Warp(
            "",
            str(workspace / "sed_export.tif"),
            format="VRT",
            dstSRS=grid["crs"],
            outputBounds=grid["bounds"],
            width=grid["width"],
            height=grid["height"],
            resampleAlg="near",
            dstNodata=-9999,
        ) as sediment,
        gdal.Warp(
            "",
            str(workspace / "stream.tif"),
            format="VRT",
            dstSRS=grid["crs"],
            outputBounds=grid["bounds"],
            width=grid["width"],
            height=grid["height"],
            resampleAlg="near",
            outputType=gdal.GDT_Int16,
            dstNodata=-9999,
        ) as streams,
    ):
        for window in windows(zones):
            ids = zones.ReadAsArray(*window).astype(np.int32)
            inside = ids > 0
            data = sediment.ReadAsArray(*window)
            valid = inside & (sediment.GetRasterBand(1).GetMaskBand().ReadAsArray(*window) != 0)
            valid &= np.isfinite(data)
            if np.any(data[valid] < 0):
                raise ValueError("Negative sediment export")
            counts += np.bincount(ids[inside], minlength=256)
            valid_counts += np.bincount(ids[valid], minlength=256)
            on_stream = inside & (streams.ReadAsArray(*window) == 1)
            stream_counts += np.bincount(ids[on_stream], minlength=256)
            totals += np.bincount(ids[valid], weights=data[valid], minlength=256)
    membership = read_csv(root / settings["zone_station"])
    result = {}
    for station_id in {int(row["station_id"]) for row in membership}:
        zone_ids = [
            int(row["zone_id"]) for row in membership if int(row["station_id"]) == station_id
        ]
        count = int(counts[zone_ids].sum())
        valid = int(valid_counts[zone_ids].sum())
        stream_count = int(stream_counts[zone_ids].sum())
        result[station_id] = {
            "area_km2": count * grid["resolution"] ** 2 / 1e6,
            "missing_output_pixels": count - valid,
            "coverage_fraction": valid / count,
            "stream_excluded_pixels": stream_count,
            "unresolved_nonstream_pixels": count - valid - stream_count,
            "hillslope_coverage_fraction": valid / (count - stream_count),
            "independent_raster_sum_t_yr": float(totals[zone_ids].sum())
            * grid["resolution"] ** 2
            / 10000,
        }
    return result


def summarize(workspace, root, settings):
    station_map = {
        int(r["station_id"]): r["station"] for r in read_csv(root / settings["zone_station"])
    }
    rows = []
    coverage = sediment_coverage(workspace, root, settings)
    # Use stable numeric IDs: some legacy GPKG station-name strings have encoding damage.
    with ogr.Open(str(workspace / "watershed_results_sdr.shp")) as source:
        for feature in source.GetLayer():
            record = feature.items()
            rows.append(
                {
                    "station": station_map[int(record["ws_id"])],
                    "ws_id": int(record["ws_id"]),
                    "hillslope_delivery_t_yr": float(record["sed_export"]),
                    "hillslope_delivery_100million_t_yr": float(record["sed_export"]) / 1e8,
                    "usle_t_yr": float(record["usle_tot"]),
                    "period": "R_1991_2020_climatology_current_landcover",
                    "status": "uncalibrated_hillslope_delivery_not_gauged_load",
                    **coverage[int(record["ws_id"])],
                }
            )
    rows.sort(key=lambda row: row["ws_id"])
    for row in rows:
        if not np.isclose(
            row["hillslope_delivery_t_yr"], row["independent_raster_sum_t_yr"], rtol=1e-5, atol=1
        ):
            raise ValueError("Raster/vector sediment totals disagree; check hectare conversion")
    if len(rows) != len(station_map) or any(
        not np.isfinite(r["hillslope_delivery_t_yr"])
        or r["hillslope_delivery_t_yr"] < 0
        or r["hillslope_delivery_t_yr"] > r["usle_t_yr"] + 1
        for r in rows
    ):
        raise ValueError("Invalid SDR station sediment budget")
    write_csv(workspace.parent / "station_sediment.csv", rows)
    references = observation_rows(
        root / settings["observations"],
        settings["observation_unit"],
        {row["station"]: row["hillslope_delivery_t_yr"] for row in rows},
    )
    write_csv(workspace.parent / "observations_reference.csv", references)
    lines = [
        "# 年均坡面入河泥沙：InVEST SDR 初始结果",
        "",
        "R：1991—2020 多年平均；当前精细图；未率定。单位为吨/年。",
        "K 未重复乘 0.1317；C/P 映射和假设见 biophysical_table.csv。",
        "测站输沙量≈坡面入河泥沙+其他侵蚀来沙−水库及河道净淤积。",
        "本表不包含沟蚀、河岸侵蚀、库坝和河道净淤积过程，不代表测站实测输沙量。",
        "",
        "| 测站 | 坡面入河泥沙（亿吨/年） | 坡面流失（亿吨/年） | 非河流区输出覆盖率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['station']} | {row['hillslope_delivery_t_yr'] / 1e8:.6f} | "
            f"{row['usle_t_yr'] / 1e8:.6f} | {row['hillslope_coverage_fraction']:.6%} |"
        )
    lines += [
        "",
        "嵌套测站结果不能相加；官方模型不在河流像元上计算坡面侵蚀，覆盖率排除河流区。",
        "剩余未连通坡面保留为缺失，不补零；原始全域覆盖率和缺失数量同时保留在 CSV。",
        "本地 InVEST 3.20.2 栅格单位为吨/(公顷·年)，核验求和时乘 6.25 公顷/像元；"
        "官方流域总量已经是吨/年，不再乘面积。",
        "实测参考及未匹配站见 observations_reference.csv；完整方法见 "
        "[SEDIMENT_SDR.md](../../../docs/SEDIMENT_SDR.md)。",
    ]
    (workspace.parent / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def run_sediment(config_path, prepare_only=False, resume_run=None):
    """Run official SDR; water experiments can call this once per distinct SDR input set."""
    configure_threads(Resources(max_workers=2, threads_per_worker=1))
    gdal.UseExceptions()
    gdal.SetCacheMax(256 * 1024 * 1024)
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if settings["policy"]["k_multiplier"] != 1:
        raise ValueError("K is already metric: a second 0.1317 conversion is prohibited")
    if settings["policy"]["p_missing"] != "no_practice_p1":
        raise ValueError("Unsupported missing P policy")
    started = time.perf_counter()
    if resume_run is None:
        output = root / settings["outputs"] / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        output.mkdir(parents=True)
    else:
        output = Path(resume_run).resolve()
        previous = json.loads((output / "settings.json").read_text(encoding="utf-8"))
        current = json.loads(json.dumps(settings))
        if {k: v for k, v in previous.items() if k != "observation_unit"} != {
            k: v for k, v in current.items() if k != "observation_unit"
        }:
            raise ValueError("Resume requires identical settings")
        if previous["observation_unit"] != current["observation_unit"]:
            write_json(
                output / "reference_unit_update.json",
                {
                    "previous": previous["observation_unit"],
                    "current": current["observation_unit"],
                    "effect": "reference_display_only_no_model_change",
                },
            )
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    write_json(output / "settings.json", settings)
    print(f"SDR output: {output}", flush=True)
    try:
        prepared = prepare_inputs(root, settings)
        if resume_run is not None:
            old = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            if old["sources"] != prepared["sources"]:
                raise ValueError("Resume requires unchanged source fingerprints")
        write_csv(output / "biophysical_table.csv", prepared["rows"])
        manifest = {
            **{key: prepared[key] for key in ("audit", "sources", "cache_events", "grid")},
            "config": fingerprint(config_path),
            "code": fingerprint(Path(__file__)),
            "versions": {key: version(key) for key in ("natcap.invest", "GDAL", "pygeoprocessing")},
            "observation_source": fingerprint(root / settings["observations"]),
            "calibrated": False,
            "k_conversion_multiplier": 1.0,
        }
        write_json(output / "manifest.json", manifest)
        if prepare_only:
            return output
        from natcap.invest.sdr import sdr

        workspace = output / "invest_sdr"
        args = {
            **settings["model"],
            "workspace_dir": str(workspace),
            "results_suffix": "",
            "dem_path": str(prepared["dem"]),
            "lulc_path": str(prepared["lulc"]),
            "erosivity_path": str(prepared["r"]),
            "erodibility_path": str(prepared["k"]),
            "watersheds_path": str(root / settings["watersheds"]),
            "biophysical_table_path": str(output / "biophysical_table.csv"),
            "drainage_path": "",
        }
        write_json(output / "invest_args.json", args)
        problems = sdr.validate(args)
        if problems:
            raise ValueError(f"InVEST SDR validation failed: {problems}")
        print("Inputs checked; running InVEST SDR with two workers", flush=True)
        sdr.execute(args)
        rows = summarize(workspace, root, settings)
        if (output / "failure.json").exists():
            (output / "failure.json").replace(output / "previous_failure.json")
        write_json(
            output / "summary.json",
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "stations": rows,
                "calibrated": False,
            },
        )
        return output
    except Exception as error:
        LOGGER.exception("SDR run failed")
        write_json(output / "failure.json", {"error": str(error)})
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
