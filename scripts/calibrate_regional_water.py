"""Compare regional Z and managed water accounts without changing frozen experiments."""

import argparse
import multiprocessing
import time
from collections import defaultdict
from contextlib import ExitStack, suppress
from pathlib import Path

import numpy as np
import psutil
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import current_models, read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.regional_calibration import (
    ENDPOINTS,
    ZONE_REGION,
    ZONE_REGION_FOUR,
    fit,
    predict,
    route,
)
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows
from ecologyhydro.water_balance import read_csv

PRODUCTS = ("fine", "copernicus")
YEARS = list(range(2019, 2024))
REGIONS = [
    "龙羊峡以上",
    "龙羊峡至兰州",
    "兰州至头道拐",
    "头道拐至龙门",
    "龙门至三门峡",
    "三门峡至花园口",
]


def metrics(predicted, observed):
    error = np.asarray(predicted) - observed
    relative = error / observed
    return {
        "mape_pct": float(np.mean(np.abs(relative)) * 100),
        "relative_rmse_pct": float(np.sqrt(np.mean(relative**2)) * 100),
        "bias_pct": float(error.sum() / np.sum(observed) * 100),
    }


def water_accounts(root, version):
    folder = root / "data/Hydrology"
    sw = {r["水资源二级区"]: r for r in read_csv(folder / "地表水耗水2018-2023.csv")}
    storage = {
        r["水资源二级区"]: r for r in read_csv(folder / "大中型水库年蓄水变化量2018-2023.csv")
    }
    consumption = np.array([[float(sw[r][str(y)]) for r in REGIONS] for y in YEARS])
    if version == "sector_2022":
        rows = read_csv(folder / "用取水耗水量/2022年黄河流域_地表水耗水量.csv")
        alternate = {r["水资源二级区"]: float(r["合计（亿立方米）"]) for r in rows}
        consumption[3] = [alternate[r] for r in REGIONS]
    elif version != "legacy":
        raise ValueError("Unknown water-account version")
    changes = np.array([[float(storage[r][str(y)]) for r in REGIONS] for y in YEARS])
    # Validate with route; this must reject missing accounts.
    route(np.zeros_like(consumption), consumption, changes)
    return consumption, changes


def source_paths(root, index, product, year):
    provider = (
        root
        / "project/repairs"
        / ("climate_reference_full_holdout" if year == 2023 else "climate_reference_full")
    )
    return {
        "zone": str(Path(index["routing"]["partitions"]) / "zones.tif"),
        "code": index["aligned"][f"lulc_{product}"],
        "pawc": index["aligned"]["pawc"],
        "soil": index["aligned"]["root_depth"],
        "p": str(provider / f"provider_ppt_{year}_aligned.tif"),
        "et": str(provider / f"provider_pet_{year}_aligned.tif"),
    }


def open_inputs(stack, paths):
    datasets = {k: stack.enter_context(gdal.Open(str(v))) for k, v in paths.items()}
    reference = datasets["zone"]
    for dataset in datasets.values():
        if (
            dataset.RasterXSize != reference.RasterXSize
            or dataset.RasterYSize != reference.RasterYSize
            or dataset.GetGeoTransform() != reference.GetGeoTransform()
            or not dataset.GetSpatialRef().IsSame(reference.GetSpatialRef())
        ):
            raise ValueError("Input grids differ")
    return datasets


def block_data(datasets, window, table):
    raw = {k: ds.ReadAsArray(*window).ravel() for k, ds in datasets.items()}
    inside = raw["zone"] > 0
    values = {k: v[inside].astype(float) for k, v in raw.items()}
    for key, array in values.items():
        nodata = datasets[key].GetRasterBand(1).GetNoDataValue()
        if not np.isfinite(array).all() or (nodata is not None and (array == nodata).any()):
            raise ValueError(f"Invalid coverage in {key}; cannot discard pixels")
    if (values["p"] <= 0).any() or (values["et"] < 0).any():
        raise ValueError("Invalid climate")
    if (values["pawc"] < 0).any() or (values["pawc"] > 1).any() or (values["soil"] < 0).any():
        raise ValueError("Invalid soil")
    lookup = {int(r["lucode"]): r for r in table}
    codes = values["code"].astype(int)
    if set(np.unique(codes)) - set(lookup):
        raise ValueError("Undefined landcover class")
    for field in ("kc", "root_depth", "lulc_veg"):
        values[field] = np.zeros(len(codes))
        for code in np.unique(codes):
            values[field][codes == code] = float(lookup[code][field])
    values["veg"] = values.pop("lulc_veg")
    values["awc"] = np.minimum(values["root_depth"], values["soil"]) * values["pawc"]
    values["zone"] = values["zone"].astype(int) - 1
    values["groups"] = values["zone"].copy()
    return values, inside


def sampling_positions(paths, budget=8192):
    """Read in blocks; retain only indices, then systematic sampling in each stratum."""
    positions = defaultdict(list)
    with gdal.Open(paths["zone"]) as zones, gdal.Open(paths["code"]) as codes:
        width = zones.RasterXSize
        gt = zones.GetGeoTransform()
        volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
        for window in windows(zones):
            z = zones.ReadAsArray(*window).ravel()
            c = codes.ReadAsArray(*window).ravel()
            for zone in np.unique(z[z > 0]):
                for code in np.unique(c[z == zone]):
                    local = np.flatnonzero((z == zone) & (c == code))
                    positions[int(zone), int(code)].append(local + window[1] * width)
    sizes = {k: sum(len(v) for v in chunks) for k, chunks in positions.items()}
    totals = {z: sum(n for (zone, _), n in sizes.items() if zone == z) for z in range(1, 12)}
    sampled, weights = [], []
    for key, chunks in positions.items():
        members = np.concatenate(chunks)
        n = min(len(members), max(32, round(budget * len(members) / totals[key[0]])))
        sampled.extend(members[np.linspace(0, len(members) - 1, n).astype(int)])
        weights.extend([len(members) / n * volume] * n)
    order = np.argsort(sampled)
    return np.asarray(sampled)[order], np.asarray(weights)[order], volume


def sample_year(paths, table, positions, weights):
    chunks = defaultdict(list)
    with ExitStack() as stack:
        datasets = open_inputs(stack, paths)
        width = datasets["zone"].RasterXSize
        for window in windows(datasets["zone"]):
            offset = window[1] * width
            size = window[3] * width
            lo, hi = np.searchsorted(positions, [offset, offset + size])
            if lo == hi:
                continue
            values, inside = block_data(datasets, window, table)
            # Map full-window indices to compact valid-pixel indices.
            selected = np.searchsorted(np.flatnonzero(inside), positions[lo:hi] - offset)
            for key, array in values.items():
                chunks[key].append(array[selected])
    data = {key: np.concatenate(arrays) for key, arrays in chunks.items()}
    if len(data["p"]) != len(weights):
        raise ValueError("Sampling lost pixels")
    data["weights"] = weights
    return data


def full_year(paths, table, parameter_sets, volume):
    """Every valid pixel, official InVEST kernel with a scalar Z per disjoint region.

    This is official-kernel verification, not a full InVEST execute workspace.
    No substitution of soil or vegetation parameters is used to implement Z.
    """
    from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

    results = {key: np.zeros(11) for key in parameter_sets}
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    with ExitStack() as stack:
        datasets = open_inputs(stack, paths)
        for window in windows(datasets["zone"]):
            data, _ = block_data(datasets, window, table)
            for key, parameters in parameter_sets.items():
                region_map = ZONE_REGION_FOUR if len(parameters) == 5 else ZONE_REGION
                region_ids = region_map[data["zone"]]
                kc = np.where(
                    data["veg"] == 1, np.minimum(data["kc"] * parameters[-1], 1.3), data["kc"]
                )
                depth = np.zeros(len(kc))
                for region in range(int(region_map.max()) + 1):
                    mask = region_ids == region
                    if not mask.any():
                        continue
                    z = parameters[0] if len(parameters) == 2 else parameters[region]
                    fraction = fractp_op(
                        kc[mask],
                        data["et"][mask],
                        data["p"][mask],
                        data["root_depth"][mask],
                        data["soil"][mask],
                        data["pawc"][mask],
                        data["veg"][mask],
                        nodata,
                        z,
                    )
                    depth[mask] = (1 - fraction) * data["p"][mask]
                if not np.isfinite(depth).all() or (depth < -0.0005).any():
                    raise ValueError("Invalid official yield")
                results[key] += np.bincount(
                    data["zone"], weights=np.maximum(0, depth) * volume, minlength=11
                )
    return {k: v.cumsum() for k, v in results.items()}


def worker(root_string, output_string, product):
    root, output = Path(root_string), Path(output_string)
    started = time.perf_counter()
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    config = load_config()
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    tables = read_json(output / "base_tables.json")[product]
    paths = {y: source_paths(root, index, product, y) for y in YEARS}
    positions, weights, volume = sampling_positions(paths[2019])
    samples = {y: sample_year(paths[y], tables[str(y)], positions, weights) for y in YEARS[:4]}
    data = {k: np.concatenate([samples[y][k] for y in YEARS[:4]]) for k in samples[2019]}
    data["groups"] = np.concatenate([samples[y]["zone"] + i * 11 for i, y in enumerate(YEARS[:4])])
    observations = read_csv(root / "data/Hydrology/实测年径流量2018-2023.csv")
    obs = {r["测站"]: r for r in observations}
    stations = config.study.station_ids
    targets = np.array([[float(obs[stations[i]][str(y)]) for i in ENDPOINTS] for y in YEARS])
    cases, cv_rows, cv_predictions = {}, [], []
    for managed in (False, True):
        for regional in (False, True):
            for version in ("legacy", "sector_2022") if managed else ("legacy",):
                name = "_".join(
                    (
                        "regional" if regional else "uniform",
                        "managed" if managed else "raw",
                        version,
                    )
                )
                consumption, storage = water_accounts(root, version)
                adjustment = (consumption + storage).cumsum(axis=1) if managed else np.zeros((5, 6))
                fitted = fit(data, targets[:4], adjustment[:4], regional)
                print(f"{product} {name}: {fitted['parameters']}", flush=True)
                predictions = []
                for withheld in range(4):
                    keep = np.arange(4) != withheld
                    fold = fit(data, targets[:4], adjustment[:4], regional, keep)
                    estimate = (
                        predict(fold["parameters"], data)[withheld, ENDPOINTS]
                        - adjustment[withheld]
                    )
                    predictions.append(estimate)
                    cv_rows.append(
                        {
                            "landcover": product,
                            "variant": name,
                            "year": YEARS[withheld],
                            "parameters": str(fold["parameters"]),
                            **metrics(estimate, targets[withheld]),
                        }
                    )
                    for i, station in enumerate(ENDPOINTS):
                        cv_predictions.append(
                            {
                                "landcover": product,
                                "variant": name,
                                "year": YEARS[withheld],
                                "station": stations[station],
                                "predicted": estimate[i],
                                "observed": targets[withheld, i],
                            }
                        )
                cases[name] = {
                    **fitted,
                    "managed": managed,
                    "water_version": version,
                    "development_leave_year_out": metrics(np.array(predictions), targets[:4]),
                }
    # Save all decisions before test-year prediction; its observations never enter fit().
    write_json(output / f"parameters_{product}.json", cases)
    write_csv(output / f"cv_{product}.csv", cv_rows)
    write_csv(output / f"cv_stations_{product}.csv", cv_predictions)
    samples[2023] = sample_year(paths[2023], tables["2023"], positions, weights)
    parameter_sets = {k: v["parameters"] for k, v in cases.items()}
    old = read_json(root / "project/calibration/major_bias/parameters.json")["parameters"][product]
    parameter_sets["old_frozen_kernel_check"] = [old["z"], old["kc_multiplier"]]
    all_full = {}
    for year in YEARS:
        all_full[year] = full_year(paths[year], tables[str(year)], parameter_sets, volume)
        print(f"{product}: full raster official kernel {year} complete", flush=True)
    rows, scores, accounts, sensitivities = [], {}, [], []
    for name, case in cases.items():
        consumption, storage = water_accounts(root, case["water_version"])
        natural = np.array([all_full[y][name][ENDPOINTS] for y in YEARS])
        estimate = route(natural, consumption, storage) if case["managed"] else natural
        approximate = np.vstack(
            [
                predict(case["parameters"], data)[:, ENDPOINTS],
                predict(case["parameters"], samples[2023], 1)[:, ENDPOINTS],
            ]
        )
        sampled_error = float(np.max(np.abs(approximate - natural) / targets))
        if sampled_error > 0.01:
            raise ValueError(f"{name}: quadrature error exceeds 1% of observed runoff")
        scores[name] = {
            "training": metrics(estimate[:4], targets[:4]),
            "reused_2023": metrics(estimate[4], targets[4]),
            "development_leave_year_out": case["development_leave_year_out"],
            "quadrature_max_fraction_observed": sampled_error,
        }
        for yi, year in enumerate(YEARS):
            for si, station in enumerate(ENDPOINTS):
                rows.append(
                    {
                        "landcover": product,
                        "variant": name,
                        "year": year,
                        "station": stations[station],
                        "awy_yield": natural[yi, si],
                        "consumption_cumulative": float(consumption[yi, : si + 1].sum())
                        if case["managed"]
                        else 0,
                        "storage_cumulative": float(storage[yi, : si + 1].sum())
                        if case["managed"]
                        else 0,
                        "predicted": estimate[yi, si],
                        "observed": targets[yi, si],
                        "relative_error": estimate[yi, si] / targets[yi, si] - 1,
                    }
                )
        # Quantify where high Z has stopped changing omega, including each proxy region.
        z_values = case["parameters"][:-1]
        for r in range(3):
            mask = (ZONE_REGION[data["zone"]] == r) & (data["veg"] == 1)
            z = z_values[0] if len(z_values) == 1 else z_values[r]
            capped = 1.25 + z * data["awc"][mask] / data["p"][mask] >= 5
            sensitivities.append(
                {
                    "landcover": product,
                    "variant": name,
                    "region": r,
                    "z": z,
                    "vegetated_area_fraction_omega_capped": float(
                        np.average(capped, weights=data["weights"][mask])
                    ),
                }
            )
    for version in ("legacy", "sector_2022"):
        consumption, storage = water_accounts(root, version)
        incremental = np.diff(targets, axis=1, prepend=0)
        for yi, year in enumerate(YEARS):
            for ri, region in enumerate(REGIONS):
                accounts.append(
                    {
                        "version": version,
                        "year": year,
                        "region": region,
                        "observed_increment": incremental[yi, ri],
                        "surface_consumption": consumption[yi, ri],
                        "storage_change": storage[yi, ri],
                        "partial_restored_increment": incremental[yi, ri]
                        + consumption[yi, ri]
                        + storage[yi, ri],
                    }
                )
    # Regression against the old official execute, which used the same uncapped base tables.
    old_rows = read_csv(root / "project/calibration/major_bias/official_comparison.csv")
    old_errors = [
        abs(
            all_full[int(r["year"])]["old_frozen_kernel_check"][stations.index(r["station"])]
            / float(r["after"])
            - 1
        )
        for r in old_rows
        if r["landcover"] == product
    ]
    if max(old_errors) > 0.001:
        raise ValueError("Official-kernel aggregation differs from old execute by >0.1%")
    write_csv(output / f"stations_{product}.csv", rows)
    write_csv(output / f"accounts_{product}.csv", accounts)
    write_csv(output / f"z_sensitivity_{product}.csv", sensitivities)
    write_json(
        output / f"summary_{product}.json",
        {
            "metrics": scores,
            "old_official_max_relative_difference": max(old_errors),
            "elapsed_seconds": time.perf_counter() - started,
            "status": "conditional_process_diagnostic_not_final_acceptance",
            "full_recompute": "official fractp_op on every pixel; not full execute workspaces",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/regional_water_v1")
    args = parser.parse_args()
    root, config = project_root(), load_config()
    configure_threads(config.resources)
    output = (root / args.output).resolve()
    if not output.is_relative_to(root.resolve() / "project/calibration"):
        raise ValueError("Output must be a new directory inside project/calibration")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing experiment; choose --output")
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "training_years": YEARS[:4],
            "reused_test_year": 2023,
            "endpoints": [config.study.station_ids[i] for i in ENDPOINTS],
            "excluded_station_reason": "Unresolved within-region allocation or downstream extent",
            "z_regions": ["above_Lanzhou", "Lanzhou_to_Longmen", "below_Longmen"],
            "region_basis": "fixed hydrological proxies, not validated climate zones",
            "bounds": {"z": [1, 30], "kc_multiplier": [0.7, 2], "kc_cap": 1.3},
            "managed_operator": "cumulative (AWY - surface consumption - reservoir change)",
            "unknown_terms": [
                "groundwater storage and stream exchange",
                "interbasin transfers",
                "irrigation ET overlap",
                "regional boundary mismatch",
            ],
            "versions": ["legacy", "sector_2022"],
            "version_selected_by_error": False,
            "kc_2023": "mean unfitted development Kc, then multiplier and cap",
            "climate": "previously selected TerraClimate pair; fixed across candidates and folds",
            "cv_interpretation": "parameter-only leave-year-out, not independent climate selection",
            "evaluation": "same six endpoints; report all variants, no test-year selection",
            "quadrature_tolerance": "max sample/full difference / observed runoff <= 1%",
        },
    )
    _, sources, snapshots = current_models()
    tables = {}
    for product in PRODUCTS:
        tables[product] = {}
        for year in YEARS[:4]:
            snapshot = next(
                s
                for _, s in snapshots
                if s["prepared"]["year"] == year and s["prepared"]["landcover"] == product
            )
            path = Path(snapshot["model_args"]["biophysical_table_path"])
            tables[product][str(year)] = read_csv(path)
            sources.append(path)
        annual = [tables[product][str(y)] for y in YEARS[:4]]
        if any([r["lucode"] for r in t] != [r["lucode"] for r in annual[0]] for t in annual):
            raise ValueError("Annual class tables differ")
        tables[product]["2023"] = [
            {**r, "kc": float(np.mean([float(t[i]["kc"]) for t in annual]))}
            for i, r in enumerate(annual[0])
        ]
    write_json(output / "base_tables.json", tables)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    for product in PRODUCTS:
        for year in YEARS:
            sources.extend(Path(p) for p in source_paths(root, index, product, year).values())
    sources.extend((root / "data/Hydrology").glob("*.csv"))
    sources += [
        root / "data/Hydrology/用取水耗水量/2022年黄河流域_地表水耗水量.csv",
        Path(__file__),
        root / "src/ecologyhydro/regional_calibration.py",
        output / "protocol.json",
    ]
    write_json(
        output / "manifest.json", {"sources": [fingerprint(p) for p in dict.fromkeys(sources)]}
    )
    started, peak = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, args=(str(root), str(output), p)) for p in PRODUCTS]
    for process in processes:
        process.start()
    try:
        while any(p.is_alive() for p in processes):
            memory = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    memory += child.memory_info().rss
            peak = max(peak, memory)
            if peak > 24 * 1024**3 or time.perf_counter() - started > 12 * 3600:
                raise RuntimeError("Experiment resource limit exceeded")
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError("A calibration worker failed")
            time.sleep(1)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError("A calibration worker failed")
    write_json(
        output / "summary.json",
        {
            "products": {p: read_json(output / f"summary_{p}.json") for p in PRODUCTS},
            "elapsed_seconds_workers": time.perf_counter() - started,
            "peak_process_tree_mib": peak / 1024**2,
            "workers": 2,
            "protocol": str(output / "protocol.json"),
        },
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
