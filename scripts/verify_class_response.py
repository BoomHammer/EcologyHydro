"""Full-pixel official AWY verification of a candidate selected using training CV only."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
from calibrate_regional_water import block_data, open_inputs
from experiment_class_response import BASE, PRODUCTS
from experiment_shared_carryover import delivered
from experiment_shared_parameters import load_samples, sample_directory
from run_landcover_comparison import input_paths, worker_setup

from ecologyhydro.baseline import read_json
from ecologyhydro.class_response import collapse_hierarchy, evaluate, group_for, prepare_response
from ecologyhydro.comparison import accounts, metrics
from ecologyhydro.long_record import ENDPOINTS, REACH_MAP, STATIONS
from ecologyhydro.paired_comparison import paired_scores
from ecologyhydro.shared_biophysical import load_registry, parameter_table
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def official_depth(x, data):
    from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

    group = data["response_group"]
    kc = np.where(
        data["veg"] == 1, np.minimum(data["kc"] * x[7 + np.clip(group, 0, 4)], 1.3), data["kc"]
    )
    kc = np.where(group == 5, x[12], kc)
    regions = REACH_MAP[data["zone"]]
    depth = np.zeros(len(kc))
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    for region in np.unique(regions):
        mask = regions == region
        fraction = fractp_op(
            kc[mask],
            data["et"][mask],
            data["p"][mask],
            data["root_depth"][mask],
            data["soil"][mask],
            data["pawc"][mask],
            data["veg"][mask],
            nodata,
            x[region],
        )
        depth[mask] = (1 - fraction) * data["p"][mask]
    if not np.isfinite(depth).all() or (depth < -0.0005).any():
        raise ValueError("Invalid full-pixel water yield")
    return np.maximum(depth, 0)


def worker(output_string, product, mode):
    output = Path(output_string)
    config = worker_setup(BASE / "execution_protocol.yaml")
    registry = load_registry(BASE / "shared_biophysical.yaml")
    hierarchy = read_json(output / "hierarchy.json")
    labels = {int(k): v for k, v in hierarchy["labels"].items()}
    x = np.asarray(read_json(output / f"fit_{mode}_final.json")["parameters"])
    original = "fine" if product == "fine_level1" else product
    table = parameter_table(registry, original, "detail")
    lut = np.full(256, -1, dtype=int)
    for row in table:
        lut[row["lucode"]] = group_for(registry, row["profile"])
    natural = []
    for year in config["forcing_years"]:
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
        natural.append(total.cumsum()[ENDPOINTS])
    natural = np.asarray(natural)
    before = delivered(natural, x[14])
    before[:, 2:] -= x[13]
    target, adjustment, eligible, _, _ = accounts(config)
    source = read_json(BASE / "experiment.json")
    raw = load_samples(sample_directory(config, source, original), original)
    sample = prepare_response(raw, registry, original)
    if product == "fine_level1":
        sample = collapse_hierarchy(sample, labels, hierarchy["components"])
    approximation = evaluate(x, sample, len(natural))
    error = float(np.max(np.abs(before - approximation)[eligible] / target[eligible]))
    rows = []
    for yi, year in enumerate(config["forcing_years"]):
        for si, station in enumerate(STATIONS):
            rows.append(
                dict(
                    mode=mode,
                    product=product,
                    year=year,
                    station=station,
                    natural=natural[yi, si],
                    predicted=(before - adjustment)[yi, si],
                    observed=target[yi, si],
                    eligible=bool(eligible[yi, si]),
                    validation=year in config["years"]["validation"],
                )
            )
    pd.DataFrame(rows).to_csv(output / f"full_{mode}_{product}.csv", index=False)
    write_json(output / f"full_check_{mode}_{product}.json", dict(max_fraction_observed=error))
    print("Official full pixels", mode, product, "sampling error", error, flush=True)
    if error > config["evaluation"]["quadrature_max_fraction_observed"]:
        raise ValueError("Full-pixel sampling tolerance exceeded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scores = pd.read_csv(args.output / "metrics.csv")
    cv = scores[scores.period == "cv"].copy()
    cv["weight"] = np.where(cv["product"].str.startswith("fine"), 0.5, 1)
    cv["weighted_mse"] = cv.relative_rmse_pct**2 * cv.weight / 3
    selection = cv.groupby("mode").weighted_mse.sum()
    chosen = selection.idxmin()
    write_json(
        args.output / "selection.json",
        dict(
            mode=chosen,
            criterion="family_mean_CV_MSE",
            candidates=selection.to_dict(),
            validation_used=False,
            win_counts_used=False,
        ),
    )
    worker_setup(BASE / "execution_protocol.yaml")
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(worker, str(args.output), p, chosen)
            for p in PRODUCTS
            if not (args.output / f"full_check_{chosen}_{p}.json").exists()
        ]
        for future in futures:
            future.result()
    frame = pd.concat([pd.read_csv(args.output / f"full_{chosen}_{p}.csv") for p in PRODUCTS])
    frame.to_csv(args.output / "full_predictions.csv", index=False)
    selected = frame[frame.validation]
    pairs, errors = paired_scores(selected)
    pd.DataFrame(pairs).to_csv(args.output / "full_paired_scores.csv", index=False)
    errors.to_csv(args.output / "full_paired_errors.csv", index=False)
    report = [
        dict(product=p, **metrics(g.predicted, g.observed, g.eligible))
        for p, g in selected.groupby("product")
    ]
    pd.DataFrame(report).to_csv(args.output / "full_metrics.csv", index=False)
    print(
        "CV selected",
        chosen,
        "FULL VALIDATION",
        pd.DataFrame(pairs).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
