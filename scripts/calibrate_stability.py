"""Compare fixed stability hypotheses using development years only, two workers."""

import argparse
import multiprocessing
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import psutil
from calibrate_regional_water import (
    full_year,
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
from ecologyhydro.regional_calibration import ENDPOINTS
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.stability import fit_stable, runoff, stability_metrics, training_subsets
from ecologyhydro.water_balance import read_csv

YEARS = [2019, 2020, 2021, 2022]
MODELS = ("four_region", "three_region", "subset_mean_four_region")


def worker(product, output_string, zones):
    root, output = project_root(), Path(output_string)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    tables = read_json(root / "project/calibration/regional_water_v1/base_tables.json")[product]
    paths = {y: {**source_paths(root, index, product, y), "zone": zones} for y in YEARS}
    positions, weights, volume = sampling_positions(paths[2019])
    samples = {y: sample_year(paths[y], tables[str(y)], positions, weights) for y in YEARS}
    data = {key: np.concatenate([samples[y][key] for y in YEARS]) for key in samples[2019]}
    data["groups"] = np.concatenate([samples[y]["zone"] + i * 11 for i, y in enumerate(YEARS)])
    obs = {r["测站"]: r for r in read_csv(root / "data/Hydrology/实测年径流量2018-2023.csv")}
    stations = load_config().study.station_ids
    target = np.array([[float(obs[stations[s]][str(y)]) for s in ENDPOINTS] for y in YEARS])
    all_fits, member_sets, adjustments, scores = {}, {}, {}, {}
    cv_rows, refit_rows, sensitivity_rows = [], [], []
    for version in ("legacy", "sector_2022"):
        consumption, reservoir = water_accounts(root, version)
        adjustment = (consumption[:4] + reservoir[:4]).cumsum(axis=1)
        adjustments[version] = adjustment

        def get_fit(indices, regions, version=version, adjustment=adjustment):
            name = f"{version}__z{regions}__" + "_".join(str(YEARS[i]) for i in indices)
            if name not in all_fits:
                all_fits[name] = fit_stable(
                    data, target, adjustment, np.isin(np.arange(4), indices), regions
                )
            return name

        def members(model, keep):
            indices = tuple(np.flatnonzero(keep).tolist())
            if model == "subset_mean_four_region":
                return [get_fit(subset, 4) for subset in training_subsets(keep)]
            return [get_fit(indices, 3 if model == "three_region" else 4)]

        def evaluate(names, forcing=data, adjustment=adjustment):
            return np.mean([runoff(all_fits[name], forcing) for name in names], axis=0) - adjustment

        for model in MODELS:
            key = f"{model}__{version}"
            full_members = members(model, np.ones(4, dtype=bool))
            full = evaluate(full_members)
            folds, estimates, fold_members = [], [], []
            for withheld in range(4):
                names = members(model, np.arange(4) != withheld)
                estimate = evaluate(names)
                folds.append(estimate)
                estimates.append(estimate[withheld])
                fold_members.append(names)
                for yi, year in enumerate(YEARS):
                    for si, station in enumerate(ENDPOINTS):
                        row = {
                            "landcover": product,
                            "variant": key,
                            "withheld_year": YEARS[withheld],
                            "year": year,
                            "station": stations[station],
                            "predicted": estimate[yi, si],
                            "observed": target[yi, si],
                            "full_fit_prediction": full[yi, si],
                        }
                        refit_rows.append(row)
                        if yi == withheld:
                            cv_rows.append(row)
            member_sets[key] = {"full": full_members, "folds": fold_members}
            scores[key] = stability_metrics(estimates, target, folds, full)
            for variable in ("p", "et"):
                for factor in (0.95, 1.05):
                    altered = evaluate(full_members, {**data, variable: data[variable] * factor})
                    delta = altered - full
                    expected_sign = (1 if factor > 1 else -1) * (1 if variable == "p" else -1)
                    if np.any(delta * expected_sign < -1e-7):
                        raise ValueError("Frozen-parameter climate response violates monotonicity")
                    sensitivity_rows.append(
                        {
                            "landcover": product,
                            "variant": key,
                            "variable": variable,
                            "factor": factor,
                            "response_rms_fraction_observed_pct": float(
                                np.sqrt(np.mean((delta / target) ** 2)) * 100
                            ),
                            "response_max_fraction_observed_pct": float(
                                np.max(np.abs(delta / target)) * 100
                            ),
                        }
                    )
            print(product, key, scores[key], flush=True)
    write_json(output / f"parameters_{product}.json", {"fits": all_fits, "members": member_sets})
    write_csv(output / f"cv_stations_{product}.csv", cv_rows)
    write_csv(output / f"refit_predictions_{product}.csv", refit_rows)
    write_csv(output / f"climate_sensitivity_{product}.csv", sensitivity_rows)
    # Verify final fits and the ensemble's three-year members with the official
    # kernel. Other outer-fold fits are checked against denser quadrature below.
    verified_names = {name for item in member_sets.values() for name in item["full"]}
    complete = {
        y: full_year(
            paths[y],
            tables[str(y)],
            {n: all_fits[n]["parameters"] for n in sorted(verified_names)},
            volume,
        )
        for y in YEARS
    }
    discrepancies = {}
    for name in sorted(verified_names):
        natural = np.array([complete[y][name][ENDPOINTS] for y in YEARS])
        approximate = runoff(all_fits[name], data)
        approximate[:, 2:] += all_fits[name]["annual_loss_1e8_m3"]
        error = float(np.max(np.abs(natural - approximate) / target))
        if error > 0.01:
            raise ValueError("Official-kernel quadrature discrepancy exceeds 1% of observations")
        discrepancies[name] = error
    dense_positions, dense_weights, _ = sampling_positions(paths[2019], budget=16384)
    dense_samples = {
        y: sample_year(paths[y], tables[str(y)], dense_positions, dense_weights) for y in YEARS
    }
    dense = {k: np.concatenate([dense_samples[y][k] for y in YEARS]) for k in dense_samples[2019]}
    dense["groups"] = np.concatenate(
        [dense_samples[y]["zone"] + i * 11 for i, y in enumerate(YEARS)]
    )
    dense_errors = {
        n: float(np.max(np.abs(runoff(f, dense) - runoff(f, data)) / target))
        for n, f in all_fits.items()
    }
    if max(dense_errors.values()) > 0.01:
        raise ValueError("Denser quadrature changes fitted predictions by more than 1%")
    write_json(
        output / f"quadrature_{product}.json",
        {
            "official_full_pixel_final_members": discrepancies,
            "double_density_all_fits": dense_errors,
            "double_density_is_not_full_pixel_verification": True,
        },
    )
    rows = []
    for key, item in member_sets.items():
        version = key.split("__")[1]
        final = (
            np.mean(
                [
                    np.array([complete[y][name][ENDPOINTS] for y in YEARS])
                    - np.array([0, 0, 1, 1, 1, 1]) * all_fits[name]["annual_loss_1e8_m3"]
                    for name in item["full"]
                ],
                axis=0,
            )
            - adjustments[version]
        )
        scores[key]["quadrature_max_fraction_observed"] = max(
            discrepancies[n] for n in item["full"]
        )
        for yi, year in enumerate(YEARS):
            for si, station in enumerate(ENDPOINTS):
                rows.append(
                    {
                        "landcover": product,
                        "variant": key,
                        "year": year,
                        "station": stations[station],
                        "predicted": final[yi, si],
                        "observed": target[yi, si],
                    }
                )
    write_csv(output / f"stations_{product}.csv", rows)
    write_json(output / f"summary_{product}.json", scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/stability_v1")
    parser.add_argument("--zones", default="project/repairs/closed_routing_v3/zones.tif")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.output).resolve()
    zones = (root / args.zones).resolve()
    if not output.is_relative_to(root.resolve() / "project/calibration") or not zones.is_file():
        raise ValueError("Require a new calibration output and existing fixed domain")
    output.mkdir(exist_ok=False)
    configure_threads(load_config().resources)
    protocol = {
        "years": YEARS,
        "evaluated_2023": False,
        "models_fixed_before_run": MODELS,
        "domain": str(zones),
        "soil_snow_external_adjustment": False,
        "accounts": ["legacy", "sector_2022"],
        "subset_average": (
            "Average predictions, not parameters; inner deletion uses training years only"
        ),
        "ensemble_outer_fold": "Three two-year fits inside each three-year training fold",
        "ensemble_final": "Four three-year fits, equal prediction weights; no tuned weights",
        "acceptance": {
            "comparisons": "each product and account against four_region on the SAME domain",
            "must_not_worsen": [
                "relative_rmse_pct",
                "worst_year_relative_rmse_pct",
                "delete_year_prediction_rms_change_pct",
            ],
            "negative_predictions": 0,
            "all_four_product_account_combinations_must_pass": True,
            "numerical_tolerance_percentage_points": 0.01,
            "production_acceptance": False,
        },
        "climate_stress": (
            "Fixed P or PET times 0.95/1.05, no refit; "
            "hypothetical, not input uncertainty estimates"
        ),
        "limitations": [
            "Only four development years",
            "Previously chosen climate source and model families",
            "Conditional CV, not independent validation",
            "Nested stations are correlated",
            "No formal confidence intervals or future-climate validation",
        ],
        "references": [
            "https://doi.org/10.1007/BF00058655",
            "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html",
            "https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/annual_water_yield.html",
        ],
    }
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "manifest.json",
        {
            "sources": [
                fingerprint(p)
                for p in (
                    Path(__file__),
                    root / "src/ecologyhydro/stability.py",
                    zones,
                    root / "scripts/calibrate_regional_water.py",
                    root / "src/ecologyhydro/regional_calibration.py",
                    root / "project/calibration/regional_water_v1/base_tables.json",
                    output / "protocol.json",
                    root / "project/repairs/soil_routing/m2_index.json",
                )
            ]
        },
    )
    started, peak = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(p, str(output), str(zones)))
        for p in ("fine", "copernicus")
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
    for model in MODELS[1:]:
        checks = []
        for product, cases in products.items():
            for account in ("legacy", "sector_2022"):
                candidate, baseline = cases[f"{model}__{account}"], cases[f"four_region__{account}"]
                checks.append(
                    {
                        "landcover": product,
                        "account": account,
                        "passes": all(
                            candidate[m] <= baseline[m] + 0.01
                            for m in protocol["acceptance"]["must_not_worsen"]
                        )
                        and candidate["negative_heldout_predictions"] == 0,
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
            "elapsed_seconds_workers": time.perf_counter() - started,
            "peak_process_tree_mib": peak / 1024**2,
            "production_accepted": False,
        },
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
