"""Calibrate four landcover representations independently and verify the main table."""

import argparse
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, wait
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml
from calibrate_regional_water import block_data, open_inputs
from experiment_shared_carryover import delivered
from experiment_shared_parameters import load_samples, sample_directory
from run_landcover_comparison import input_paths, worker_setup
from verify_class_response import official_depth

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.class_response import (
    collapse_hierarchy,
    evaluate,
    group_for,
    hierarchy_components,
    prepare_response,
    thin_samples,
)
from ecologyhydro.comparison import accounts, metrics, training_mask
from ecologyhydro.independent_calibration import expand_regional, fit_independent
from ecologyhydro.long_record import ENDPOINTS, STATIONS
from ecologyhydro.regional_transfer import fit_transfer
from ecologyhydro.shared_biophysical import load_registry, parameter_table
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows

COUNTS = {"reach7": 9, "class": 14, "class_storage": 15}


def context(output):
    output = Path(output)
    recipe = read_json(output / "experiment.json")
    config = worker_setup(output / "execution_protocol.yaml")
    return output, recipe, config


def sample_response(output, product, compressed=True):
    output, recipe, config = context(output)
    if compressed:
        return load_samples(output / "samples", product)
    original = "fine" if product == "fine_level1" else product
    source = read_json(output / "sample_recipe.json")
    registry = load_registry(output / "biophysical.yaml")
    raw = load_samples(sample_directory(config, source, original), original)
    data = prepare_response(raw, registry, original)
    if product == "fine_level1":
        hierarchy = read_json(output / "hierarchy.json")
        labels = {int(k): v for k, v in hierarchy["labels"].items()}
        data = collapse_hierarchy(data, labels, hierarchy["components"])
    return data


def rows_for(config, product, mode, fit_name, prediction, target, eligible, selected):
    rows = []
    for yi in np.flatnonzero(selected):
        for si, station in enumerate(STATIONS):
            rows.append(
                dict(
                    product=product,
                    mode=mode,
                    fit=fit_name,
                    year=config["forcing_years"][yi],
                    station=station,
                    predicted=prediction[yi, si],
                    observed=target[yi, si],
                    eligible=bool(eligible[yi, si]),
                )
            )
    return rows


def calibrate(output, product, name, held, final_mode=None):
    output, recipe, config = context(output)
    data = sample_response(output, product)
    target, adjustment, eligible, _, _ = accounts(config)
    keep = training_mask(config) & ~np.isin(config["forcing_years"], held)
    modes = (
        recipe["candidates"] if final_mode is None else list(dict.fromkeys(["reach7", final_mode]))
    )
    cv_rows = []
    for mode in modes:
        path = output / f"fit_{product}_{mode}_{name}.json"
        started = time.monotonic()
        if path.exists():
            fit = read_json(path)
        elif mode == "reach7":
            fit = fit_transfer(data, target, adjustment, eligible, keep, "reach7")
            fit["regional_parameters"] = fit["parameters"]
            fit["parameters"] = expand_regional(fit["parameters"]).tolist()
            fit.update(independent=True, mode=mode)
        else:
            seed = read_json(output / f"fit_{product}_reach7_{name}.json")["regional_parameters"]
            fit = fit_independent(data, target, adjustment, eligible, keep, mode, seed, recipe)
        fit.update(product=product, held_years=held, seconds=time.monotonic() - started)
        if not path.exists():
            write_json(path, fit)
        if not fit.get("success", True):
            print("FAILED CANDIDATE", product, name, mode, flush=True)
            if name == "final":
                raise RuntimeError(f"Selected final candidate did not converge: {product} {mode}")
            continue
        if name != "final":
            predicted = evaluate(fit["parameters"], data, len(keep)) - adjustment
            cv_rows.extend(
                rows_for(
                    config,
                    product,
                    mode,
                    name,
                    predicted,
                    target,
                    eligible,
                    np.isin(config["forcing_years"], held),
                )
            )
        print("FIT", product, name, mode, round(fit["seconds"], 1), flush=True)
    if name != "final":
        pd.DataFrame(cv_rows).to_csv(output / f"cv_{product}_{name}.csv", index=False)
        return
    fit = read_json(output / f"fit_{product}_{final_mode}_final.json")
    x = fit["parameters"]
    quick = evaluate(x, data, len(keep))
    del data
    full = sample_response(output, product, compressed=False)
    approximate = evaluate(x, full, len(keep))
    check = dict(
        product=product,
        max_fraction_observed=float(
            np.max(np.abs(quick - approximate)[eligible] / target[eligible])
        ),
        training_max_fraction_observed=float(
            np.max(
                np.abs(quick - approximate)[eligible & keep[:, None]]
                / target[eligible & keep[:, None]]
            )
        ),
    )
    write_json(output / f"compression_{product}.json", check)
    rows = rows_for(
        config,
        product,
        final_mode,
        "final",
        approximate - adjustment,
        target,
        eligible,
        np.ones(len(keep), dtype=bool),
    )
    pd.DataFrame(rows).to_csv(output / f"sample_predictions_{product}.csv", index=False)


def full_pixels(output, product):
    output, recipe, config = context(output)
    mode = read_json(output / "selection.json")[product]["mode"]
    x = np.asarray(read_json(output / f"fit_{product}_{mode}_final.json")["parameters"])
    registry = load_registry(output / "biophysical.yaml")
    hierarchy = read_json(output / "hierarchy.json")
    labels = {int(k): v for k, v in hierarchy["labels"].items()}
    original = "fine" if product == "fine_level1" else product
    table = parameter_table(registry, original, "detail")
    lut = np.full(256, -1, dtype=int)
    for row in table:
        lut[row["lucode"]] = group_for(registry, row["profile"])
    # Only stateful candidates need full forcing before the three report years.
    years = config["forcing_years"] if x[14] else recipe["validation_years"]
    natural = []
    for year in years:
        checkpoint = output / f"natural_{product}_{year}.json"
        if checkpoint.exists():
            natural.append(read_json(checkpoint)["natural"])
            continue
        started = time.monotonic()
        total = np.zeros(11)
        with ExitStack() as stack:
            datasets = open_inputs(stack, input_paths(config, original, year))
            gt = datasets["zone"].GetGeoTransform()
            volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
            for window in windows(datasets["zone"]):
                data, _ = block_data(datasets, window, table)
                data["response_group"] = lut[data["code"].astype(int)]
                data["weights"] = np.full(len(data["p"]), volume)
                if product == "fine_level1":
                    data = collapse_hierarchy(data, labels, hierarchy["components"])
                depth = official_depth(x, data)
                total += np.bincount(data["zone"], weights=depth * data["weights"], minlength=11)
        value = total.cumsum()[ENDPOINTS]
        natural.append(value)
        write_json(checkpoint, dict(natural=value.tolist(), parameters=x.tolist(), year=year))
        print("FULL", product, year, round(time.monotonic() - started, 1), flush=True)
    before = delivered(np.asarray(natural), x[14])
    before[:, 2:] -= x[13]
    target, adjustment, eligible, _, _ = accounts(config)
    lookup = [config["forcing_years"].index(y) for y in years]
    predicted = np.full_like(target, np.nan)
    predicted[lookup] = before - adjustment[lookup]
    rows = rows_for(
        config,
        product,
        mode,
        "final",
        predicted,
        target,
        eligible,
        np.isin(config["forcing_years"], recipe["validation_years"]),
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"full_{product}.csv", index=False)
    sample = pd.read_csv(output / f"sample_predictions_{product}.csv")
    pair = frame.merge(sample, on=["product", "year", "station"], suffixes=("", "_sample"))
    valid = pair.eligible
    error = float(((pair.predicted - pair.predicted_sample).abs() / pair.observed)[valid].max())
    write_json(
        output / f"quadrature_{product}.json",
        dict(max_fraction_observed=error, passed=error <= 0.01, full_pixel_table=True),
    )
    if error > config["evaluation"]["quadrature_max_fraction_observed"]:
        raise ValueError(f"Full-pixel quadrature tolerance exceeded for {product}: {error}")


def run_tasks(function, tasks, config, output, label):
    started, peak = time.monotonic(), 0
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {pool.submit(function, *args) for args in tasks}
        while pending:
            done, pending = wait(pending, timeout=2)
            for job in done:
                job.result()
            processes = [psutil.Process(), *psutil.Process().children(recursive=True)]
            rss = sum(p.memory_info().rss for p in processes if p.is_running())
            peak = max(peak, rss)
            if rss > config["resources"]["memory_budget_gb"] * 1024**3:
                for process in psutil.Process().children(recursive=True):
                    process.terminate()
                raise MemoryError("Independent experiment exceeded memory budget")
            if time.monotonic() - started > config["resources"]["timeout_hours"] * 3600:
                for process in psutil.Process().children(recursive=True):
                    process.terminate()
                raise TimeoutError("Independent experiment timeout")
    write_json(
        output / f"performance_{label}.json",
        dict(seconds=time.monotonic() - started, peak_gib=peak / 1024**3, workers=2),
    )


def select(output, recipe):
    frame = pd.concat([pd.read_csv(path) for path in sorted(output.glob("cv_*_fold*.csv"))])
    scores, selections, excluded = [], {}, []
    for product in recipe["products"]:
        candidates = []
        for mode in recipe["candidates"]:
            part = frame[(frame["product"] == product) & (frame["mode"] == mode)]
            failed = [
                i
                for i in range(len(recipe["folds"]))
                if not read_json(output / f"fit_{product}_{mode}_fold{i}.json").get("success", True)
            ]
            if failed:
                excluded.append(dict(product=product, mode=mode, failed_folds=failed))
                continue
            if len(part) != 8 * 7 or part.duplicated(["year", "station"]).any():
                raise ValueError("Incomplete or duplicated training CV predictions")
            score = dict(
                product=product, mode=mode, **metrics(part.predicted, part.observed, part.eligible)
            )
            scores.append(score)
            candidates.append(score)
        chosen = min(candidates, key=lambda r: (r["relative_rmse_pct"], COUNTS[r["mode"]]))
        selections[product] = dict(
            mode=chosen["mode"],
            cv_relative_rmse_pct=chosen["relative_rmse_pct"],
            criterion=recipe["selection"],
            validation_used=False,
        )
    frame.to_csv(output / "cv_predictions.csv", index=False)
    pd.DataFrame(scores).to_csv(output / "cv_metrics.csv", index=False)
    write_json(output / "selection.json", selections)
    write_json(output / "excluded_candidates.json", excluded)
    print("SELECTION", selections, flush=True)
    return selections


def prepare(recipe_path):
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    config = worker_setup(recipe["baseline_config"])
    output = Path(recipe["output_root"]) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    (output / "samples").mkdir(parents=True)
    write_json(output / "experiment.json", recipe)
    shutil.copyfile(config["config_path"], output / "execution_protocol.yaml")
    shutil.copyfile(recipe["registry"], output / "biophysical.yaml")
    shutil.copyfile(recipe["sample_recipe"], output / "sample_recipe.json")
    source = read_json(output / "sample_recipe.json")
    registry = load_registry(output / "biophysical.yaml")
    sources = [
        recipe_path,
        Path(__file__),
        Path("src/ecologyhydro/independent_calibration.py"),
        Path("src/ecologyhydro/class_response.py"),
        Path("src/ecologyhydro/regional_transfer.py"),
        config["config_path"],
        Path(recipe["registry"]),
        Path(recipe["sample_recipe"]),
        Path("scripts/verify_class_response.py"),
        Path("scripts/report_independent_calibration.py"),
    ]
    sources += [Path(p) for p in config["hydrology"].values()]
    sources.append(Path(config["products"]["fine"]["classes"]))
    for product in config["products"]:
        directory = sample_directory(config, source, product)
        sources.append(directory / f"samples_{product}.npz")
        raw = load_samples(directory, product)
        data = prepare_response(
            thin_samples(raw, recipe["calibration_budget_per_zone"]), registry, product
        )
        np.savez(output / "samples" / f"samples_{product}.npz", **data)
        if product == "fine":
            labels, components = hierarchy_components(
                raw, registry, Path(config["products"]["fine"]["classes"])
            )
            write_json(output / "hierarchy.json", dict(labels=labels, components=components))
            coarse = collapse_hierarchy(data, labels, components)
            np.savez(output / "samples/samples_fine_level1.npz", **coarse)
            del coarse
        del raw, data
        print("PREPARED", product, flush=True)
    for product in config["products"]:
        for year in config["forcing_years"]:
            sources.extend(Path(p) for p in input_paths(config, product, year).values())
    sources = list(dict.fromkeys(p.resolve() for p in sources))
    write_json(output / "sources.json", [fingerprint(p) for p in sources])
    for path in sources[:10]:
        if path.suffix == ".py":
            shutil.copyfile(path, output / f"executed_{path.name}")
    write_json(output / "status.json", dict(status="prepared", validation_used_for_selection=False))
    return output, recipe, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, default=Path("config/independent_calibration.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage", choices=("all", "calibration", "full"), default="all")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    output, recipe, config = context(args.resume) if args.resume else prepare(args.recipe)
    print("OUTPUT", output, flush=True)
    try:
        if args.stage in ("all", "calibration"):
            tasks = [
                (str(output), product, f"fold{i}", held)
                for product in recipe["products"]
                for i, held in enumerate(recipe["folds"])
                if not (output / f"cv_{product}_fold{i}.csv").exists()
            ]
            run_tasks(calibrate, tasks, config, output, "cv")
            selections = select(output, recipe)
            tasks = [
                (str(output), product, "final", [], selections[product]["mode"])
                for product in recipe["products"]
                if not (output / f"sample_predictions_{product}.csv").exists()
            ]
            run_tasks(calibrate, tasks, config, output, "final")
            write_json(
                output / "freeze.json",
                dict(status="four_independent_fits_frozen", selection=selections),
            )
        if args.stage in ("all", "full"):
            if not (output / "freeze.json").exists():
                raise ValueError("Freeze all independent parameters before full-pixel evaluation")
            tasks = [
                (str(output), p)
                for p in recipe["products"]
                if not (output / f"quadrature_{p}.json").exists()
            ]
            run_tasks(full_pixels, tasks, config, output, "full")
            from report_independent_calibration import report

            report(output)
            saved = read_json(output / "sources.json")
            if saved != [fingerprint(Path(item["path"])) for item in saved]:
                raise ValueError("Experiment sources changed during execution")
            write_json(
                output / "status.json", dict(status="complete", full_pixels=True, independent=True)
            )
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
