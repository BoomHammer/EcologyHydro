"""Shared-parameter, fixed-forcing control of map identity and supported class detail."""

import argparse
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml
from calibrate_long_record import full_models
from osgeo import gdal
from run_landcover_comparison import input_paths, run_pool, validate_prepared, worker_setup

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import (
    accounts,
    load_protocol,
    prediction_rows,
    resolve,
    summarize_rows,
    training_mask,
)
from ecologyhydro.config import project_root
from ecologyhydro.long_record import ENDPOINTS
from ecologyhydro.regional_transfer import evaluate, fit_transfer, training_data
from ecologyhydro.shared_biophysical import (
    load_registry,
    parameter_table,
    replace_sample_parameters,
)
from ecologyhydro.simulation import write_json


def load_samples(baseline, product):
    with np.load(baseline / f"samples_{product}.npz") as archive:
        return {key: archive[key] for key in archive.files}


def sample_directory(config, recipe, product):
    return resolve(config, recipe.get("sample_sources", {}).get(product, recipe["baseline_run"]))


def fit_task(config_path, output_string, name, held):
    config, output = worker_setup(config_path), Path(output_string)
    recipe = read_json(output / "experiment.json")
    registry = load_registry(output / "shared_biophysical.yaml")
    keep = training_mask(config) & ~np.isin(config["forcing_years"], held)
    target, adjustment, eligible, _, _ = accounts(config)
    parts = []
    for pi, product in enumerate(config["products"]):
        # Exclude validation and held-year forcing before the optimizer sees any samples.
        raw = training_data(load_samples(sample_directory(config, recipe, product), product), keep)
        data = replace_sample_parameters(
            raw, parameter_table(registry, product, recipe["fit_mode"])
        )
        data["groups"] = data["groups"] + pi * int(keep.sum()) * 11
        parts.append(data)
    joint = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    del parts, data, raw
    products = len(config["products"])
    fit = fit_transfer(
        joint,
        np.tile(target[keep], (products, 1)),
        np.tile(adjustment[keep], (products, 1)),
        np.tile(eligible[keep], (products, 1)),
        np.ones(int(keep.sum()) * products, dtype=bool),
        "reach7",
    )
    fit.update(
        training_years=np.asarray(config["forcing_years"])[keep].tolist(),
        held_years=held,
        unique_observations=int(eligible[keep].sum()),
        repeated_product_residuals=int(eligible[keep].sum()) * products,
        interpretation="Equal-weight model losses, NOT three independent copies of observations",
        fit_mode=recipe["fit_mode"],
        used_for="all_three_products_and_both_detail_modes",
    )
    write_json(output / f"fit_{name}.json", fit)
    print("Shared fit frozen", name, fit["parameters"], flush=True)


def full_worker(config_path, output_string, product):
    config, output = worker_setup(config_path), Path(output_string)
    recipe = read_json(output / "experiment.json")
    registry = load_registry(output / "shared_biophysical.yaml")
    years = config["forcing_years"]
    target, adjustment, eligible, _, _ = accounts(config)
    fit = read_json(output / "fit_final.json")
    raw = load_samples(sample_directory(config, recipe, product), product)
    rows, cv_rows, checks, audits = [], [], [], []
    forward_cache = {}
    with gdal.Open(input_paths(config, product, years[0])["zone"]) as raster:
        gt = raster.GetGeoTransform()
        volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
    for mode in recipe["modes"]:
        table = parameter_table(registry, product, mode)
        data = replace_sample_parameters(raw, table)
        pd.DataFrame(table).to_csv(output / f"table_{product}_{mode}.csv", index=False)
        first = raw["groups"] // 11 == 0
        for row in table:
            mask = first & (raw["code"] == row["lucode"])
            audits.append(
                dict(
                    product=product,
                    mode=mode,
                    **row,
                    area_km2=float(raw["weights"][mask].sum() * 1e5),
                    soil_limited_area_km2=float(
                        raw["weights"][mask & (raw["soil"] < row["root_depth"])].sum() * 1e5
                    )
                    if row["lulc_veg"]
                    else None,
                )
            )
        for fi, held in enumerate(recipe["folds"]):
            fold_fit = read_json(output / f"fit_fold{fi}.json")
            before = evaluate(fold_fit["parameters"], data, "reach7", len(years))
            natural = before.copy()
            natural[:, 2:] += fold_fit["parameters"][-1]
            predicted = before - adjustment
            pair = prediction_rows(
                config,
                f"shared_{recipe['fit_mode']}_fit",
                product,
                predicted,
                natural,
                target,
                adjustment,
                eligible,
            )
            cv_rows.extend({**row, "mode": mode, "fold": fi} for row in pair if row["year"] in held)
        approximate = evaluate(fit["parameters"], data, "reach7", len(years))
        signature = tuple(
            (row["lucode"], row["lulc_veg"], row["kc"], row["root_depth"]) for row in table
        )
        if signature not in forward_cache:
            natural = []
            for year in years:
                official, _ = full_models(
                    input_paths(config, product, year),
                    table,
                    {"shared": ("reach7", fit)},
                    volume,
                    all_stations=True,
                )
                natural.append(official["shared"][ENDPOINTS])
            forward_cache[signature] = np.asarray(natural)
        else:
            print("Reuse identical full-pixel response", product, mode, flush=True)
        natural = forward_cache[signature]
        before = natural.copy()
        before[:, 2:] -= fit["parameters"][-1]
        discrepancy = float(
            np.max(np.abs(before[eligible] - approximate[eligible]) / target[eligible])
        )
        checks.append(dict(product=product, mode=mode, max_fraction_observed=discrepancy))
        pair = prediction_rows(
            config,
            f"shared_{recipe['fit_mode']}_fit",
            product,
            before - adjustment,
            natural,
            target,
            adjustment,
            eligible,
        )
        rows.extend({**row, "mode": mode} for row in pair)
        print("Full pixels", product, mode, "max discrepancy", discrepancy, flush=True)
    pd.DataFrame(rows).to_csv(output / f"predictions_{product}.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output / f"cv_{product}.csv", index=False)
    pd.DataFrame(audits).to_csv(output / f"audit_{product}.csv", index=False)
    write_json(output / f"quadrature_{product}.json", checks)
    if (
        max(r["max_fraction_observed"] for r in checks)
        > config["evaluation"]["quadrature_max_fraction_observed"]
    ):
        raise ValueError("Sampling quadrature tolerance exceeded")


def main():
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="config/shared_parameter_control.yaml")
    recipe_path = root / parser.parse_args().recipe
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if recipe["fit_mode"] not in {"parent", "detail"}:
        raise ValueError("Shared fit mode must be parent or detail")
    config = load_protocol(recipe["baseline_config"])
    validate_prepared(config)
    if recipe.get("fine_sample_manifest"):
        sample_manifest = read_json(resolve(config, recipe["fine_sample_manifest"]))
        for item in sample_manifest["sources"] + [sample_manifest["sample"]]:
            if fingerprint(Path(item["path"])) != item:
                raise ValueError("Fine sample cache does not match its source manifest")
        if sample_manifest["aligned"] != fingerprint(
            resolve(config, config["products"]["fine"]["aligned"])
        ):
            raise ValueError("Fine samples belong to a different landcover version")
    output = resolve(config, recipe["output_root"]) / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", recipe)
    shutil.copyfile(config["config_path"], output / "execution_protocol.yaml")
    registry_path = resolve(config, recipe["registry"])
    shutil.copyfile(registry_path, output / "shared_biophysical.yaml")
    shutil.copyfile(__file__, output / "executed_experiment_shared_parameters.py")
    module_path = root / "src/ecologyhydro/shared_biophysical.py"
    shutil.copyfile(module_path, output / "shared_biophysical.py")
    sources = [recipe_path, registry_path, Path(__file__), module_path, config["config_path"]]
    sources += [root / "src/ecologyhydro/regional_transfer.py"]
    sources += [resolve(config, path) for path in config["hydrology"].values()]
    sources += [
        sample_directory(config, recipe, p) / f"samples_{p}.npz" for p in config["products"]
    ]
    sources += [
        resolve(config, config["products"]["fine"][key]) for key in ("source", "classes", "aligned")
    ]
    if recipe.get("fine_sample_manifest"):
        sources.append(resolve(config, recipe["fine_sample_manifest"]))
    manifest = [fingerprint(path) for path in sources]
    write_json(output / "manifest.json", {"sources": manifest})
    print("Run directory:", output, flush=True)
    started, peak = time.perf_counter(), 0
    tasks = [(f"fold{i}", held) for i, held in enumerate(recipe["folds"])] + [("final", [])]
    with ProcessPoolExecutor(
        max_workers=config["resources"]["max_workers"],
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(fit_task, str(config["config_path"]), str(output), name, held)
            for name, held in tasks
        ]
        while not all(f.done() for f in futures):
            processes = [psutil.Process(), *psutil.Process().children(recursive=True)]
            rss = sum(p.memory_info().rss for p in processes if p.is_running())
            peak = max(peak, rss)
            if rss > config["resources"]["memory_budget_gb"] * 1024**3:
                for process in psutil.Process().children(recursive=True):
                    process.terminate()
                raise MemoryError("Shared calibration exceeded memory budget")
            if time.perf_counter() - started > config["resources"]["timeout_hours"] * 3600:
                for process in psutil.Process().children(recursive=True):
                    process.terminate()
                raise TimeoutError("Shared calibration timeout")
            time.sleep(2)
        for future in futures:
            future.result()
    performance = {"fits": {"seconds": time.perf_counter() - started, "peak_gib": peak / 1024**3}}
    write_json(output / "freeze.json", dict(status="all_four_fits_frozen", selection="none"))
    performance["full_pixels"] = run_pool(full_worker, config, output)
    products = config["products"]
    for prefix in ("predictions", "cv", "audit"):
        combined = pd.concat([pd.read_csv(output / f"{prefix}_{p}.csv") for p in products])
        combined.to_csv(output / f"{prefix}.csv", index=False)
        if prefix == "audit":
            continue
        summaries = []
        for (mode, _), group in combined.groupby(["mode", "landcover"]):
            summaries.extend(
                {**row, "mode": mode} for row in summarize_rows(group.to_dict("records"))
            )
        pd.DataFrame(summaries).to_csv(output / f"{prefix}_metrics.csv", index=False)
    if manifest != [fingerprint(path) for path in sources]:
        raise ValueError("Experiment sources changed during execution")
    write_json(
        output / "summary.json",
        dict(
            status="complete",
            performance=performance,
            parameter_sets_per_fold=1,
            product_specific_parameters=False,
            selection="none",
            active_model_replaced=False,
            interpretation="Fairness control with explicitly unresolved annual Kc priors",
            original_sources_unchanged=True,
        ),
    )
    print("Complete:", output, flush=True)


if __name__ == "__main__":
    main()
