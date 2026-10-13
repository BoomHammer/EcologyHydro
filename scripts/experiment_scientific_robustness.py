"""Equal-budget sensitivity and regularized routing without overwriting the accepted run."""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from experiment_fine_focus_routing import cached_natural, natural_sample, parameters
from experiment_independent_calibration import rows_for, run_tasks, sample_response
from run_landcover_comparison import worker_setup

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.calibration_followup import freeze
from ecologyhydro.comparison import accounts, metrics, training_mask
from ecologyhydro.fine_focus_goals import assess_focus
from ecologyhydro.routing_robustness import fit, forward
from ecologyhydro.simulation import write_json


def candidates(recipe):
    return [
        dict(
            name=f"{sharing}_r{ridge:g}_t{weight:g}",
            sharing=sharing,
            ridge=ridge,
            training_weight=weight,
        )
        for sharing in recipe["sharing"]
        for ridge in recipe["ridge_strengths"]
        for weight in recipe["training_weights"]
    ]


def context(output):
    output = Path(output)
    return (
        output,
        read_json(output / "protocol.json"),
        worker_setup(output / "execution_protocol.yaml"),
    )


def fold_base(recipe, product, name, n_years):
    reference = Path(recipe["reference"])
    selected = read_json(reference / f"freeze_{product}.json")["mode"]
    seed = read_json(Path(recipe["local_run"]) / f"local_fit_{product}_{name}.json")["parameters"]
    head = read_json(reference / f"head_{product}_False_{name}.json")["parameters"]
    regions = (5, 6) if selected == "head_lower" else ()
    if selected not in ("head", "head_lower"):
        raise ValueError("Unsupported inherited candidate; audit its fold construction first")
    regional = {
        r: read_json(reference / f"regional_{product}_{r}_{name}.json")["parameters"]
        for r in regions
    }
    return freeze(seed, head, regional, np.ones(n_years), selected)


def prepare_fold_cache(output_string, product):
    output, recipe, config = context(output_string)
    cache = output / f"fold_natural_{product}.npz"
    if cache.exists():
        return
    years = config["forcing_years"]
    data = sample_response(output, product, compressed=False)
    arrays = {}
    for i in range(3):
        frozen = fold_base(recipe, product, f"fold{i}", len(years))
        arrays[f"fold{i}"] = natural_sample(data, frozen, len(years))
        arrays[f"prior{i}"] = parameters(frozen)
    np.savez(cache, **arrays)
    print("FOLD CACHE", product, flush=True)


def fit_product(output_string, product):
    output, recipe, config = context(output_string)
    reference = Path(recipe["reference"])
    years = np.array(config["forcing_years"])
    target, adjustment, eligible, _, _ = accounts(config)
    train = training_mask(config)
    selected = np.isin(years, recipe["target_years"])
    natural = cached_natural(reference, product, product, years)
    prior = parameters(read_json(reference / f"freeze_{product}.json"))
    original = pd.read_csv(reference / "full_predictions.csv")
    fold_cache = np.load(output / f"fold_natural_{product}.npz")
    folds = read_json(reference / "experiment.json")["folds"]
    scores, prediction_rows, cv_rows = [], [], []
    # CV fits depend on sharing/penalty, not target/historical mixing weights.
    cv_results = {}
    for spec in candidates(recipe):
        weights = selected / selected.sum() + spec["training_weight"] * train / train.sum()
        fitted = fit(natural, target, adjustment, eligible, weights, prior, spec)
        write_json(output / f"fit_{product}_{spec['name']}.json", {**fitted, "spec": spec})
        value = forward(natural, fitted["parameters"]) - adjustment
        rows = pd.DataFrame(
            rows_for(config, product, spec["name"], "final", value, target, eligible, selected)
        ).assign(parameter_source=product)
        prediction_rows.append(rows)
        key = (spec["sharing"], spec["ridge"])
        if key not in cv_results:
            these_rows = []
            for i, held in enumerate(folds):
                keep = train & ~np.isin(years, held)
                horizon = np.flatnonzero(keep)[-1] + 1
                fold_natural = fold_cache[f"fold{i}"]
                result = fit(
                    fold_natural[:horizon],
                    target[:horizon],
                    adjustment[:horizon],
                    eligible[:horizon],
                    keep[:horizon].astype(float),
                    fold_cache[f"prior{i}"],
                    spec,
                )
                write_json(
                    output / f"cvfit_{product}_{spec['sharing']}_{spec['ridge']:g}_{i}.json", result
                )
                pred = forward(fold_natural, result["parameters"]) - adjustment
                these_rows += rows_for(
                    config,
                    product,
                    f"{spec['sharing']}_{spec['ridge']:g}",
                    f"fold{i}",
                    pred,
                    target,
                    eligible,
                    np.isin(years, held),
                )
            cv_frame = pd.DataFrame(these_rows)
            future = cv_frame[cv_frame.fit == "fold2"]
            cv_results[key] = dict(
                cv=metrics(cv_frame.predicted, cv_frame.observed, cv_frame.eligible)[
                    "relative_rmse_pct"
                ],
                forward=metrics(future.predicted, future.observed, future.eligible)[
                    "relative_rmse_pct"
                ],
            )
            cv_rows += these_rows
        outcome = {}
        if product == "fine":
            frame = pd.concat([rows, original[original["product"] != "fine"]])
            outcome, _, _ = assess_focus(frame)
        scores.append(
            dict(
                product=product,
                **spec,
                **cv_results[key],
                **outcome,
                n_free=fitted["n_free"],
                objective=fitted["objective"],
                condition_number=fitted["condition_number"],
                local_effective_df=fitted["local_effective_df"],
                loss=fitted["parameters"][7],
                training_error=metrics(value, target, eligible & train[:, None])[
                    "relative_rmse_pct"
                ],
                target_error=metrics(value, target, eligible & selected[:, None])[
                    "relative_rmse_pct"
                ],
            )
        )
    pd.DataFrame(scores).to_csv(output / f"scores_{product}.csv", index=False)
    pd.concat(prediction_rows).to_csv(output / f"predictions_{product}.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output / f"cv_{product}.csv", index=False)
    print("FITS", product, flush=True)


def prepare(path):
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    output = Path(recipe["output_root"]) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    write_json(output / "protocol.json", recipe)
    reference = Path(recipe["reference"])
    for name in (
        "experiment.json",
        "execution_protocol.yaml",
        "biophysical.yaml",
        "sample_recipe.json",
        "hierarchy.json",
    ):
        shutil.copyfile(reference / name, output / name)
    sources = [
        path,
        Path(__file__),
        Path("src/ecologyhydro/routing_robustness.py"),
        Path(recipe["accepted"]) / "sources.json",
        reference / "sources.json",
    ]
    sources += sorted(reference.glob("natural_*.json")) + sorted(reference.glob("freeze_*.json"))
    sources += sorted(reference.glob("head_*_False_fold*.json"))
    sources += sorted(reference.glob("regional_*_fold*.json"))
    sources += sorted(Path(recipe["local_run"]).glob("local_fit_*_fold*.json"))
    write_json(output / "sources.json", [fingerprint(p) for p in sources])
    for item in sources:
        if item.suffix == ".py":
            shutil.copyfile(item, output / f"executed_{item.name}")
    print("OUTPUT", output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/scientific_robustness.yaml"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    output = args.run or prepare(args.config)
    output, recipe, config = context(output)
    run_tasks(
        prepare_fold_cache,
        [(str(output), p) for p in recipe["products"]],
        config,
        output,
        "fold_cache",
    )
    run_tasks(fit_product, [(str(output), p) for p in recipe["products"]], config, output, "fits")
    scores = pd.concat([pd.read_csv(output / f"scores_{p}.csv") for p in recipe["products"]])
    scores.to_csv(output / "candidate_metrics.csv", index=False)
    fine = scores[scores["product"] == "fine"]
    passed = fine[fine.aggregate_all_best.eq(True) & fine.station_year_40_percent_best.eq(True)]
    ranked = passed.sort_values(["cv", "n_free", "target_error"])
    ranked.to_csv(output / "accepted_candidates.csv", index=False)
    print(
        ranked[
            [
                "name",
                "aggregate_best_count",
                "station_year_wins",
                "cv",
                "forward",
                "training_error",
                "target_error",
                "loss",
                "condition_number",
            ]
        ].to_string(index=False),
        flush=True,
    )
    write_json(output / "status.json", dict(status="screened", primary_baseline_preserved=True))


if __name__ == "__main__":
    main()
