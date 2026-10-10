"""Past-only validation of defensible three/seven station-proxy regions."""

import argparse
import multiprocessing
import time
from contextlib import ExitStack, suppress
from pathlib import Path

import numpy as np
import psutil
from calibrate_regional_water import block_data, open_inputs, sample_year, sampling_positions
from long_inputs import paths_for, tables_for
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.long_record import ENDPOINTS, REACH_MAP, STATIONS, YEARS, load_accounts, scores
from ecologyhydro.regional_transfer import evaluate, fit_transfer, regional_map
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows

MODELS = ("macro3", "reach7", "pooled7")
TEST_YEARS = (2019, 2020, 2021, 2022)


def full_models(paths, table, cases, volume, all_stations=False):
    """Official kernel on every pixel, with unchanged scalar Z per proxy region."""
    from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

    results = {name: np.zeros(11) for name in cases}
    climate = {name: np.zeros(7) for name in ("p", "et", "area_km2")}
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    with ExitStack() as stack:
        datasets = open_inputs(stack, paths)
        for window in windows(datasets["zone"]):
            data, _ = block_data(datasets, window, table)
            ids = REACH_MAP[data["zone"]]
            climate["area_km2"] += np.bincount(ids, minlength=7) * volume * 1e5
            for variable in ("p", "et"):
                climate[variable] += np.bincount(ids, weights=data[variable] * volume, minlength=7)
            for name, (model, fit) in cases.items():
                x = fit["parameters"]
                region = regional_map(model)[data["zone"]]
                kc = np.where(data["veg"] == 1, np.minimum(data["kc"] * x[-2], 1.3), data["kc"])
                depth = np.zeros(len(kc))
                for group in np.unique(region):
                    selected = region == group
                    fraction = fractp_op(
                        kc[selected],
                        data["et"][selected],
                        data["p"][selected],
                        data["root_depth"][selected],
                        data["soil"][selected],
                        data["pawc"][selected],
                        data["veg"][selected],
                        nodata,
                        x[group],
                    )
                    depth[selected] = (1 - fraction) * data["p"][selected]
                if not np.isfinite(depth).all() or (depth < -0.0005).any():
                    raise ValueError("Invalid official pixel yield")
                results[name] += np.bincount(
                    data["zone"], weights=np.maximum(0, depth) * volume, minlength=11
                )
    selected = slice(None) if all_stations else ENDPOINTS
    return {name: values.cumsum()[selected] for name, values in results.items()}, climate


def worker(product, output_string):
    root, output = project_root(), Path(output_string)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    paths = {y: paths_for(root, index, product, y) for y in YEARS}
    positions, weights, volume = sampling_positions(paths[2013])
    tables, rotation = tables_for(root, index, product, YEARS, positions, weights)
    write_json(output / f"tables_{product}.json", tables)
    if rotation:
        write_csv(output / "annual_rotation_kc.csv", rotation)
    samples = {y: sample_year(paths[y], tables[y], positions, weights) for y in YEARS}
    data = {key: np.concatenate([samples[y][key] for y in YEARS]) for key in samples[2013]}
    data["groups"] = np.concatenate([samples[y]["zone"] + i * 11 for i, y in enumerate(YEARS)])
    plans = {
        "early": np.array(YEARS) <= 2017,
        **{f"before_{y}": np.array(YEARS) < y for y in TEST_YEARS[1:]},
        "all_development": np.ones(len(YEARS), dtype=bool),
    }
    # Accountless 2018 cannot enter any supervised calibration.
    for mask in plans.values():
        mask[YEARS.index(2018)] = False
    fits, predictions, accounts, fit_cache = {}, {}, {}, {}
    for version in ("legacy", "sector_2022"):
        target, _, _, adjustment, eligible = load_accounts(root, version)
        accounts[version] = (target, adjustment, eligible)
        for model in MODELS:
            for window, keep in plans.items():
                name = f"{model}__{version}__{window}"
                cache_key = (model, tuple(keep), target[keep].tobytes(), adjustment[keep].tobytes())
                if cache_key not in fit_cache:
                    fit_cache[cache_key] = fit_transfer(
                        data, target, adjustment, eligible, keep, model
                    )
                fits[name] = fit_cache[cache_key]
                predictions[name] = (
                    evaluate(fits[name]["parameters"], data, model, len(YEARS)) - adjustment
                )
                print(
                    product,
                    name,
                    "n=",
                    fits[name]["observations"],
                    "bounds=",
                    fits[name]["active_bounds"],
                    flush=True,
                )
    write_json(output / f"parameters_{product}.json", fits)
    # Report all rolling fits and all early-fit predictions, never optimize on test years.
    rows, official_by_year, climate_rows = [], {}, []
    for yi, year in enumerate(YEARS):
        cases = {}
        if year in TEST_YEARS:
            for version in accounts:
                for model in MODELS:
                    for window in ("early", "early" if year == 2019 else f"before_{year}"):
                        name = f"{model}__{version}__{window}"
                        cases[name] = (model, fits[name])
        official, climate = full_models(paths[year], tables[year], cases, volume)
        official_by_year[year] = official
        for region in range(7):
            climate_rows.append(
                {
                    "year": year,
                    "region_id": region + 1,
                    "area_km2": climate["area_km2"][region],
                    "precipitation_mm": climate["p"][region] * 1e5 / climate["area_km2"][region],
                    "pet_mm": climate["et"][region] * 1e5 / climate["area_km2"][region],
                }
            )
        for version, (target, adjustment, eligible) in accounts.items():
            for model in MODELS:
                for phase, window in (
                    ("fixed_early", "early"),
                    ("rolling", "early" if year == 2019 else f"before_{year}"),
                    ("development_fit", "all_development"),
                ):
                    if phase == "rolling" and year not in TEST_YEARS:
                        continue
                    name = f"{model}__{version}__{window}"
                    approximate = predictions[name][yi]
                    predicted = approximate.copy()
                    official_used = phase != "development_fit" and year in TEST_YEARS
                    if official_used:
                        x = fits[name]["parameters"]
                        predicted = (
                            official[name]
                            - np.array([0, 0, 1, 1, 1, 1, 1]) * x[-1]
                            - adjustment[yi]
                        )
                        error = np.max(
                            np.abs(predicted[eligible[yi]] - approximate[eligible[yi]])
                            / target[yi, eligible[yi]]
                        )
                        if error > 0.01:
                            raise ValueError(
                                "Official temporal prediction differs by over 1% of observations"
                            )
                    for si, station in enumerate(STATIONS):
                        rows.append(
                            {
                                "landcover": product,
                                "model": model,
                                "account": version,
                                "phase": phase,
                                "year": year,
                                "station": station,
                                "fit": name,
                                "training_years": ";".join(
                                    str(YEARS[i]) for i in fits[name]["training_indices"]
                                ),
                                "eligible": bool(eligible[yi, si]),
                                "predicted": float(predicted[si])
                                if np.isfinite(predicted[si])
                                else None,
                                "sampled_prediction": float(approximate[si])
                                if np.isfinite(approximate[si])
                                else None,
                                "observed": float(target[yi, si])
                                if np.isfinite(target[yi, si])
                                else None,
                                "known_adjustment": float(adjustment[yi, si])
                                if np.isfinite(adjustment[yi, si])
                                else None,
                                "official_full_pixel": official_used,
                                "validation": year in TEST_YEARS and phase != "development_fit",
                            }
                        )
        print(product, "pixel verification and climate audit", year, flush=True)
    write_csv(output / f"stations_{product}.csv", rows)
    write_csv(output / f"climate_regions_{product}.csv", climate_rows)
    summary = {}
    for version in accounts:
        for model in MODELS:
            for phase in ("fixed_early", "rolling"):
                selected = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["account"] == version
                    and r["phase"] == phase
                    and r["validation"]
                ]
                pred = np.array([r["predicted"] for r in selected], dtype=float).reshape(4, 7)
                obs = np.array([r["observed"] for r in selected], dtype=float).reshape(4, 7)
                valid = np.array([r["eligible"] for r in selected]).reshape(4, 7)
                metrics = scores(pred, obs, valid)
                yearly = [scores(pred[i], obs[i], valid[i]) for i in range(4)]
                summary[f"{model}__{version}__{phase}"] = {
                    **metrics,
                    "years": dict(zip(map(str, TEST_YEARS), yearly, strict=True)),
                    "worst_year_relative_rmse_pct": max(r["relative_rmse_pct"] for r in yearly),
                    "common_six_stations": scores(pred[:, :6], obs[:, :6], valid[:, :6]),
                    "lower_reach_conditional": scores(pred[:, 6:], obs[:, 6:], valid[:, 6:]),
                }
    write_json(output / f"summary_{product}.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/long_record_v1")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root.resolve() / "project/calibration"):
        raise ValueError("Output must be under project/calibration")
    read_json(root / "project/repairs/long_climate_v1/summary.json")
    output.mkdir(exist_ok=False)
    configure_threads(load_config().resources)
    write_json(
        output / "protocol.json",
        {
            "forcing_years": YEARS,
            "initial_training_years": list(range(2013, 2018)),
            "forward_validation_years": TEST_YEARS,
            "evaluated_2023": False,
            "accountless_2018": "Climate audit only; no zero fill, interpolation or flow score",
            "2014_missing_storage": "Retain Guide/Lanzhou; missing account propagates downstream",
            "models": MODELS,
            "parameter_counts": {"macro3": 5, "reach7": 9, "pooled7": 9},
            "pooling": ("Fixed lambda=0.01; mean relative SSE plus within-macro log-Z penalty"),
            "partition_source": "User-authorized gauge proxies; no official polygons",
            "macro_boundaries": ["Toudaoguai", "Huayuankou"],
            "seven_boundaries": STATIONS,
            "mapping_limits": ["Guide != Longyangxia", "Lijin != mouth; last account conditional"],
            "target": "Observed regulated Q; conditional consumption/reservoir accounts",
            "constant_net_loss": "One unidentified Lanzhou-Toudaoguai L; historical bounds",
            "accounts": ["legacy", "sector_2022"],
            "model_selection": "Report all; no winner selected by validation or 2023",
            "production_accepted": False,
            "interpretation": (
                "Past-only parameter fits; previously selected climate and static LULC"
            ),
            "rotation": "Same-year V1.1 monthly PET and fixed crop calendar for all years",
            "acceleration": "Analytic derivatives, fit cache, two single-thread workers",
            "references": [
                "https://slsy.nhri.cn/cn/article/doi/10.12170/20210923001",
                "https://www.climatologylab.org/terraclimate.html",
                "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
            ],
        },
    )
    sources = [
        Path(__file__),
        root / "scripts/long_inputs.py",
        root / "src/ecologyhydro/long_record.py",
        root / "src/ecologyhydro/regional_transfer.py",
        root / "config/crop_systems.yaml",
        output / "protocol.json",
        root / "project/repairs/closed_routing_v3/zones.tif",
        root / "project/repairs/long_climate_v1/summary.json",
        root / "project/calibration/regional_water_v1/base_tables.json",
        *sorted((root / "data/Hydrology").glob("*2013-2023.csv")),
    ]
    write_json(output / "manifest.json", {"sources": [fingerprint(p) for p in sources]})
    started, peak = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=worker, args=(p, str(output))) for p in ("fine", "copernicus")
    ]
    for process in workers:
        process.start()
    try:
        while any(p.is_alive() for p in workers):
            memory = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    memory += child.memory_info().rss
            peak = max(peak, memory)
            if memory > 24 * 1024**3 or time.perf_counter() - started > 12 * 3600:
                raise RuntimeError("Resource budget exceeded")
            if any(p.exitcode not in (None, 0) for p in workers):
                raise RuntimeError("Calibration worker failed")
            time.sleep(1)
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join()
    if any(p.exitcode != 0 for p in workers):
        raise RuntimeError("Calibration worker failed")
    write_json(
        output / "summary.json",
        {
            "products": {
                p: read_json(output / f"summary_{p}.json") for p in ("fine", "copernicus")
            },
            "elapsed_seconds_workers": time.perf_counter() - started,
            "peak_process_tree_mib": peak / 1024**2,
            "production_accepted": False,
        },
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
