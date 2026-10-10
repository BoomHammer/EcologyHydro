"""Recompute all held-out predictions on every pixel using the official AWY kernel."""

import argparse
import multiprocessing
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import psutil
from calibrate_regional_water import full_year, source_paths, water_accounts
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.regional_calibration import ENDPOINTS
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import read_csv


def worker(product, output_string):
    root, output = project_root(), Path(output_string)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    parent = output.parent
    protocol = read_json(parent / "protocol.json")
    parameters = read_json(parent / f"parameters_{product}.json")
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    tables = read_json(root / "project/calibration/regional_water_v1/base_tables.json")[product]
    with gdal.Open(protocol["domain"]) as raster:
        gt = raster.GetGeoTransform()
        volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
    targets = {
        (r["variant"], int(r["year"]), r["station"]): r
        for r in read_csv(parent / f"cv_stations_{product}.csv")
    }
    stations = [load_config().study.station_ids[i] for i in ENDPOINTS]
    adjustment = {}
    for account in ("legacy", "sector_2022"):
        c, s = water_accounts(root, account)
        adjustment[account] = (c[:4] + s[:4]).cumsum(axis=1)
    rows = []
    for yi, year in enumerate(protocol["years"]):
        names = {name for item in parameters["members"].values() for name in item["folds"][yi]}
        paths = {**source_paths(root, index, product, year), "zone": protocol["domain"]}
        natural = full_year(
            paths,
            tables[str(year)],
            {n: parameters["fits"][n]["parameters"] for n in sorted(names)},
            volume,
        )
        for variant, members in parameters["members"].items():
            account = variant.split("__")[1]
            modeled = (
                np.mean(
                    [
                        natural[name][ENDPOINTS]
                        - np.array([0, 0, 1, 1, 1, 1])
                        * parameters["fits"][name]["annual_loss_1e8_m3"]
                        for name in members["folds"][yi]
                    ],
                    axis=0,
                )
                - adjustment[account][yi]
            )
            for si, station in enumerate(stations):
                original = targets[variant, year, station]
                rows.append(
                    {
                        "landcover": product,
                        "variant": variant,
                        "year": year,
                        "station": station,
                        "predicted": modeled[si],
                        "observed": float(original["observed"]),
                        "quadrature_prediction": float(original["predicted"]),
                    }
                )
        print(product, "full held-out pixel verification", year, flush=True)
    write_csv(output / f"cv_stations_{product}.csv", rows)
    scores = {}
    original_summary = read_json(parent / "summary.json")["products"][product]
    for variant in parameters["members"]:
        selected = [r for r in rows if r["variant"] == variant]
        prediction = np.array([r["predicted"] for r in selected]).reshape(4, 6)
        observation = np.array([r["observed"] for r in selected]).reshape(4, 6)
        approximate = np.array([r["quadrature_prediction"] for r in selected]).reshape(4, 6)
        error = (prediction - observation) / observation
        discrepancy = float(np.max(np.abs(prediction - approximate) / observation))
        if discrepancy > 0.01:
            raise ValueError("Full-pixel outer fold discrepancy exceeds one percent")
        yearly_rmse = np.sqrt(np.mean(error**2, axis=1)) * 100
        yearly_mape = np.mean(np.abs(error), axis=1) * 100
        scores[variant] = {
            **original_summary[variant],
            "mape_pct": float(np.mean(np.abs(error)) * 100),
            "relative_rmse_pct": float(np.sqrt(np.mean(error**2)) * 100),
            "worst_year_relative_rmse_pct": float(yearly_rmse.max()),
            "worst_year_mape_pct": float(yearly_mape.max()),
            "year_mape_std_pct": float(yearly_mape.std()),
            "worst_station_year_absolute_error_pct": float(np.abs(error).max() * 100),
            "year_mape_pct": yearly_mape.tolist(),
            "year_relative_rmse_pct": yearly_rmse.tolist(),
            "negative_heldout_predictions": int((prediction < 0).sum()),
            "official_outer_fold_quadrature_max_fraction_observed": discrepancy,
            "delete_year_sensitivity_remains_quadrature_based": True,
        }
    write_json(output / f"summary_{product}.json", scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="project/calibration/stability_v1")
    args = parser.parse_args()
    root = project_root()
    parent = (root / args.experiment).resolve()
    if not parent.is_relative_to(root.resolve() / "project/calibration"):
        raise ValueError("Experiment must be inside project/calibration")
    protocol = read_json(parent / "protocol.json")
    read_json(parent / "summary.json")  # Never consume an unfinished calibration.
    output = parent / "full_pixel_cv"
    output.mkdir(exist_ok=False)
    configure_threads(load_config().resources)
    write_json(
        output / "protocol.json",
        {
            "purpose": (
                "Resolve small CV score differences by all-pixel verification of every model/fold"
            ),
            "refit_or_model_change": False,
            "acceptance_threshold_change": False,
            "acceptance": protocol["acceptance"],
            "evaluated_2023": False,
            "delete_year_sensitivity": "Retain original four-year-forcing quadrature sensitivity",
            "sources": [
                fingerprint(p)
                for p in (
                    Path(__file__),
                    parent / "protocol.json",
                    parent / "parameters_fine.json",
                    parent / "parameters_copernicus.json",
                )
            ],
        },
    )
    started, peak = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(p, str(output))) for p in ("fine", "copernicus")
    ]
    for process in processes:
        process.start()
    try:
        while any(p.is_alive() for p in processes):
            memory = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    memory += child.memory_info().rss
            peak = max(peak, memory)
            if memory > 24 * 1024**3 or time.perf_counter() - started > 12 * 3600:
                raise RuntimeError("Resource budget exceeded")
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError("Worker failed")
            time.sleep(1)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError("Worker failed")
    products = {p: read_json(output / f"summary_{p}.json") for p in ("fine", "copernicus")}
    decisions = {}
    for model in ("three_region", "subset_mean_four_region"):
        checks = []
        for product, cases in products.items():
            for account in ("legacy", "sector_2022"):
                candidate, baseline = cases[f"{model}__{account}"], cases[f"four_region__{account}"]
                failed = [
                    m
                    for m in protocol["acceptance"]["must_not_worsen"]
                    if candidate[m] > baseline[m] + 0.01
                ]
                checks.append(
                    {
                        "landcover": product,
                        "account": account,
                        "failed_metrics": failed,
                        "passes": not failed and candidate["negative_heldout_predictions"] == 0,
                    }
                )
        decisions[model] = {
            "checks": checks,
            "passes_stability_gate": all(c["passes"] for c in checks),
        }
    write_json(
        output / "summary.json",
        {
            "products": products,
            "decisions": decisions,
            "production_accepted": False,
            "elapsed_seconds_workers": time.perf_counter() - started,
            "peak_process_tree_mib": peak / 1024**2,
        },
    )
    print(decisions, flush=True)


if __name__ == "__main__":
    main()
