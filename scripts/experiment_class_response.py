"""Test fixed shared response candidates against products and actual legend hierarchy."""

import argparse
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_shared_parameters import load_samples, sample_directory
from run_landcover_comparison import worker_setup

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.class_response import (
    collapse_hierarchy,
    evaluate,
    fit_shared,
    hierarchy_components,
    prepare_response,
    thin_samples,
)
from ecologyhydro.comparison import accounts, metrics, training_mask
from ecologyhydro.long_record import STATIONS
from ecologyhydro.paired_comparison import paired_scores
from ecologyhydro.shared_biophysical import load_registry
from ecologyhydro.simulation import write_json

BASE = Path("experiments/shared_detail_calibration/20261011T092823_836045Z")
MODES = ("class", "class_local", "class_storage")
PRODUCTS = ("fine", "fine_level1", "copernicus", "worldcover")


def fit_task(output, mode, name, held):
    output = Path(output)
    config = worker_setup(BASE / "execution_protocol.yaml")
    recipe = read_json(output / "experiment.json")
    datasets = {p: load_samples(output / "samples", p) for p in PRODUCTS}
    target, adjustment, eligible, _, _ = accounts(config)
    keep = training_mask(config) & ~np.isin(config["forcing_years"], held)
    # Fold-specific seed is trained without the same held observations.
    seed = read_json(BASE / f"fit_{name}.json")["parameters"]
    started = time.monotonic()
    fit = fit_shared(datasets, target, adjustment, eligible, keep, mode, seed)
    fit.update(seconds=time.monotonic() - started, held_years=held, budget=recipe["budget"])
    write_json(output / f"fit_{mode}_{name}.json", fit)
    print(mode, name, "fit complete", round(fit["seconds"], 1), fit["parameters"], flush=True)


def predictions(output):
    config = worker_setup(BASE / "execution_protocol.yaml")
    recipe = read_json(output / "experiment.json")
    source = read_json(BASE / "experiment.json")
    registry = load_registry(BASE / "shared_biophysical.yaml")
    target, adjustment, eligible, _, _ = accounts(config)
    years = np.asarray(config["forcing_years"])
    hierarchy = read_json(output / "hierarchy.json")
    labels = {int(k): v for k, v in hierarchy["labels"].items()}
    rows, checks = [], []
    folds = [("final", config["years"]["validation"])] + [
        (f"fold{i}", held) for i, held in enumerate(source["folds"])
    ]
    for product in PRODUCTS:
        original = "fine" if product == "fine_level1" else product
        raw = load_samples(sample_directory(config, source, original), original)
        data = prepare_response(raw, registry, original)
        if product == "fine_level1":
            data = collapse_hierarchy(data, labels, hierarchy["components"])
        compressed = load_samples(output / "samples", product)
        for mode in recipe["candidates"]:
            for name, held in folds:
                x = read_json(output / f"fit_{mode}_{name}.json")["parameters"]
                full = evaluate(x, data, len(years))
                quick = evaluate(x, compressed, len(years))
                checks.append(
                    dict(
                        mode=mode,
                        fit=name,
                        product=product,
                        max_fraction_observed=float(
                            np.max(np.abs(full - quick)[eligible] / target[eligible])
                        ),
                    )
                )
                predicted = full - adjustment
                for yi in np.flatnonzero(np.isin(years, held)):
                    for si, station in enumerate(STATIONS):
                        rows.append(
                            dict(
                                mode=mode,
                                fit=name,
                                product=product,
                                period="validation" if name == "final" else "cv",
                                year=int(years[yi]),
                                station=station,
                                predicted=predicted[yi, si],
                                observed=target[yi, si],
                                eligible=bool(eligible[yi, si]),
                            )
                        )
        print("Original sample verification", product, flush=True)
        del raw, data, compressed
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "compression_checks.csv", index=False)
    scores, paired, errors = [], [], []
    for (mode, period), group in frame.groupby(["mode", "period"]):
        records, error = paired_scores(group)
        paired.extend(dict(mode=mode, period=period, **r) for r in records)
        errors.append(error.assign(mode=mode, period=period))
        for product, part in group.groupby("product"):
            scores.append(
                dict(
                    mode=mode,
                    period=period,
                    product=product,
                    **metrics(part.predicted, part.observed, part.eligible),
                )
            )
    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_scores.csv", index=False)
    pd.concat(errors).to_csv(output / "paired_errors.csv")
    write_json(
        output / "status.json",
        dict(
            status="sample_complete",
            full_pixels=False,
            maximum_compression_error=max(c["max_fraction_observed"] for c in checks),
        ),
    )
    print(pd.DataFrame(paired).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    config = worker_setup(BASE / "execution_protocol.yaml")
    source = read_json(BASE / "experiment.json")
    output = args.resume or Path("experiments/class_response") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    if not args.resume:
        (output / "samples").mkdir(parents=True)
        recipe = dict(
            candidates=list(MODES),
            budget=args.budget,
            baseline=str(BASE),
            calibration="shared class scales, equal map-family weights; training only",
            hierarchy="true legend level one; global area mixtures of subclass responses",
            validation="reused exploratory 2022-2024; all candidates reported",
            folds=source["folds"],
            selection="No selection using validation or win counts",
        )
        write_json(output / "experiment.json", recipe)
        shutil.copyfile(BASE / "execution_protocol.yaml", output / "execution_protocol.yaml")
        registry = load_registry(BASE / "shared_biophysical.yaml")
        for product in config["products"]:
            raw = load_samples(sample_directory(config, source, product), product)
            if product == "fine":
                labels, components = hierarchy_components(
                    raw, registry, Path("data/LCLU/vegetation_model_20261010_100411.csv")
                )
                write_json(output / "hierarchy.json", dict(labels=labels, components=components))
            data = prepare_response(thin_samples(raw, args.budget), registry, product)
            np.savez(output / "samples" / f"samples_{product}.npz", **data)
            if product == "fine":
                coarse = collapse_hierarchy(data, labels, components)
                np.savez(output / "samples/samples_fine_level1.npz", **coarse)
                del coarse
            print("Prepared", product, len(data["p"]), flush=True)
        write_json(
            output / "sources.json",
            [
                fingerprint(Path(p))
                for p in (
                    __file__,
                    "src/ecologyhydro/class_response.py",
                    "src/ecologyhydro/paired_comparison.py",
                    BASE / "shared_biophysical.yaml",
                    "data/LCLU/vegetation_model_20261010_100411.csv",
                )
            ],
        )
    print("OUTPUT", output, flush=True)
    recipe = read_json(output / "experiment.json")
    folds = [("final", [])] + [(f"fold{i}", held) for i, held in enumerate(source["folds"])]
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        jobs = [
            pool.submit(fit_task, str(output), mode, name, held)
            for mode in recipe["candidates"]
            for name, held in folds
            if not (output / f"fit_{mode}_{name}.json").exists()
        ]
        for job in as_completed(jobs):
            job.result()
    predictions(output)


if __name__ == "__main__":
    main()
