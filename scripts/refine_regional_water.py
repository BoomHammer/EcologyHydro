"""Test a fourth Z upstream of Guide; retain both account versions and all results."""

import multiprocessing
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import psutil
from calibrate_regional_water import (
    YEARS,
    full_year,
    metrics,
    sample_year,
    sampling_positions,
    source_paths,
    water_accounts,
)
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.regional_calibration import ENDPOINTS, fit, predict, route
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import read_csv


def worker(product):
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    root = project_root()
    output = root / "project/calibration/regional_water_v2"
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    tables = read_json(root / "project/calibration/regional_water_v1/base_tables.json")[product]
    paths = {y: source_paths(root, index, product, y) for y in YEARS}
    positions, weights, volume = sampling_positions(paths[2019])
    samples = {y: sample_year(paths[y], tables[str(y)], positions, weights) for y in YEARS[:4]}
    data = {k: np.concatenate([samples[y][k] for y in YEARS[:4]]) for k in samples[2019]}
    data["groups"] = np.concatenate([samples[y]["zone"] + i * 11 for i, y in enumerate(YEARS[:4])])
    obs = {r["测站"]: r for r in read_csv(root / "data/Hydrology/实测年径流量2018-2023.csv")}
    stations = load_config().study.station_ids
    target = np.array([[float(obs[stations[s]][str(y)]) for s in ENDPOINTS] for y in YEARS])
    cases, cv_rows = {}, []
    for version in ("legacy", "sector_2022"):
        consumption, storage = water_accounts(root, version)
        adjustment = (consumption + storage).cumsum(axis=1)
        fitted = fit(data, target[:4], adjustment[:4], regional=True, regions=4)
        predicted = []
        for withheld in range(4):
            fold = fit(
                data,
                target[:4],
                adjustment[:4],
                regional=True,
                keep=np.arange(4) != withheld,
                regions=4,
            )
            estimate = predict(fold["parameters"], data)[withheld, ENDPOINTS] - adjustment[withheld]
            predicted.append(estimate)
            for i, station in enumerate(ENDPOINTS):
                cv_rows.append(
                    {
                        "landcover": product,
                        "variant": version,
                        "year": YEARS[withheld],
                        "station": stations[station],
                        "parameters": str(fold["parameters"]),
                        "predicted": estimate[i],
                        "observed": target[withheld, i],
                    }
                )
        cases[version] = {
            **fitted,
            "development_leave_year_out": metrics(np.array(predicted), target[:4]),
        }
        print(product, version, cases[version], flush=True)
    write_json(output / f"parameters_{product}.json", cases)
    write_csv(output / f"cv_stations_{product}.csv", cv_rows)
    samples[2023] = sample_year(paths[2023], tables["2023"], positions, weights)
    complete = {
        y: full_year(
            paths[y], tables[str(y)], {k: c["parameters"] for k, c in cases.items()}, volume
        )
        for y in YEARS
    }
    rows, scores = [], {}
    for version, case in cases.items():
        consumption, storage = water_accounts(root, version)
        natural = np.array([complete[y][version][ENDPOINTS] for y in YEARS])
        estimates = route(natural, consumption, storage)
        approximate = np.vstack(
            [
                predict(case["parameters"], data)[:, ENDPOINTS],
                predict(case["parameters"], samples[2023], 1)[:, ENDPOINTS],
            ]
        )
        discrepancy = float(np.max(np.abs(approximate - natural) / target))
        if discrepancy > 0.01:
            raise ValueError("Spatial quadrature discrepancy exceeds 1% of observation")
        scores[version] = {
            "training": metrics(estimates[:4], target[:4]),
            "development_leave_year_out": case["development_leave_year_out"],
            "reused_2023": metrics(estimates[4], target[4]),
            "quadrature_max_fraction_observed": discrepancy,
        }
        for yi, year in enumerate(YEARS):
            for si, station in enumerate(ENDPOINTS):
                rows.append(
                    {
                        "landcover": product,
                        "variant": version,
                        "year": year,
                        "station": stations[station],
                        "awy_yield": natural[yi, si],
                        "consumption_cumulative": consumption[yi, : si + 1].sum(),
                        "storage_cumulative": storage[yi, : si + 1].sum(),
                        "predicted": estimates[yi, si],
                        "observed": target[yi, si],
                        "relative_error": estimates[yi, si] / target[yi, si] - 1,
                    }
                )
    write_csv(output / f"stations_{product}.csv", rows)
    write_json(output / f"summary_{product}.json", scores)


def main():
    root = project_root()
    configure_threads(load_config().resources)
    output = root / "project/calibration/regional_water_v2"
    output.mkdir(exist_ok=False)
    write_json(
        output / "protocol.json",
        {
            "training_years": YEARS[:4],
            "reused_test_year": 2023,
            "status": "followup_hypothesis_after_inspecting_v1_residuals_including_2023",
            "z_regions": ["above_Guide", "Guide_to_Lanzhou", "Lanzhou_to_Longmen", "below_Longmen"],
            "hypothesis": "v1 combines positive Guide and negative Guide-Lanzhou residuals",
            "selection_rule": "compare development leave-year-out errors; report both versions",
            "parameter_bounds": {"four_z": [1, 30], "kc_multiplier": [0.7, 2], "kc_cap": 1.3},
            "other_assumptions": "identical to regional_water_v1, including six-endpoint scope",
            "independent_validation": False,
        },
    )
    from natcap.invest.annual_water_yield import annual_water_yield

    sources = [
        root / "scripts/refine_regional_water.py",
        root / "scripts/calibrate_regional_water.py",
        root / "src/ecologyhydro/regional_calibration.py",
        output / "protocol.json",
        root / "project/calibration/regional_water_v1/base_tables.json",
        root / "project/calibration/regional_water_v1/manifest.json",
    ]
    sources.append(Path(annual_water_yield.__file__))
    write_json(output / "manifest.json", {"sources": [fingerprint(p) for p in sources]})
    start, peak = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, args=(p,)) for p in ("fine", "copernicus")]
    for p in processes:
        p.start()
    try:
        while any(p.is_alive() for p in processes):
            memory = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    memory += child.memory_info().rss
            peak = max(peak, memory)
            if memory > 24 * 1024**3 or time.perf_counter() - start > 12 * 3600:
                raise RuntimeError("Resource budget exceeded")
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError("Worker failed")
            time.sleep(1)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError("Worker failed")
    write_json(
        output / "summary.json",
        {
            "products": {
                p: read_json(output / f"summary_{p}.json") for p in ("fine", "copernicus")
            },
            "elapsed_seconds_workers": time.perf_counter() - start,
            "peak_process_tree_mib": peak / 1024**2,
            "workers": 2,
        },
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
