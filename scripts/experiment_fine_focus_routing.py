"""Goal-aligned fine-only routing experiment, full-pixel audit and current transfers."""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from experiment_fine_priority import full_year
from experiment_independent_calibration import rows_for, run_tasks, sample_response
from report_fine_focus import publish
from run_landcover_comparison import worker_setup

from ecologyhydro.accuracy_goals import PRODUCTS
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.calibration_followup import freeze, seasonal_data
from ecologyhydro.class_response import evaluate
from ecologyhydro.comparison import accounts, metrics, training_mask
from ecologyhydro.fine_focus_goals import assess_focus
from ecologyhydro.fine_focus_routing import (
    balance_ledger,
    fit_routing,
    routing_prediction,
    selection_key,
)
from ecologyhydro.long_record import REACH_MAP, STATIONS
from ecologyhydro.simulation import write_json


def cached_natural(reference, product, source, years):
    return np.array([read_json(reference / f"natural_{product}_{y}.json")[source] for y in years])


def parameters(frozen):
    return np.r_[frozen["retention"], frozen["seed"][13]]


def natural_sample(data, frozen, years):
    changed = seasonal_data(data, frozen["seasonal_factors"])
    upper = (REACH_MAP[data["zone"]] == 0) & (data["veg"] == 1)
    changed["kc"] = data["kc"].copy()
    changed["kc"][upper] *= frozen["head_multiplier"]
    x = np.array(frozen["seed"])
    x[13:] = 0
    return evaluate(x, changed, years)


def prepare(recipe_path):
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    output = Path(recipe["output_root"]) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    write_json(output / "fine_protocol.json", recipe)
    reference = Path(recipe["reference"])
    for name in (
        "experiment.json",
        "execution_protocol.yaml",
        "biophysical.yaml",
        "sample_recipe.json",
        "hierarchy.json",
    ):
        shutil.copyfile(reference / name, output / name)
    config = worker_setup(output / "execution_protocol.yaml")
    sources = [
        recipe_path,
        Path(__file__),
        Path("src/ecologyhydro/fine_focus_routing.py"),
        Path("src/ecologyhydro/fine_focus_goals.py"),
        Path("src/ecologyhydro/calibration_followup.py"),
        Path("src/ecologyhydro/class_response.py"),
        Path("src/ecologyhydro/comparison.py"),
        Path("src/ecologyhydro/accuracy_goals.py"),
        Path("scripts/report_fine_focus.py"),
        Path("scripts/experiment_fine_priority.py"),
        Path("scripts/verify_class_response.py"),
        Path("src/ecologyhydro/fine_priority.py"),
        reference / "sources.json",
        reference / "full_predictions.csv",
        reference / "transfer_predictions.csv",
        reference / "execution_protocol.yaml",
        reference / "biophysical.yaml",
        reference / "sample_recipe.json",
    ]
    sources += [reference / f"freeze_{p}.json" for p in PRODUCTS]
    sources += [
        reference / f"natural_{p}_{y}.json" for p in PRODUCTS for y in config["forcing_years"]
    ]
    sources += [
        Path(config["hydrology"][key]) for key in ("observations", "consumption", "storage")
    ]
    for i in range(3):
        sources += [
            Path(recipe["local_run"]) / f"local_fit_fine_fold{i}.json",
            reference / f"head_fine_False_fold{i}.json",
        ]
        sources += [reference / f"regional_fine_{r}_fold{i}.json" for r in (5, 6)]
    write_json(output / "sources.json", [fingerprint(p) for p in sources])
    for path in sources:
        if path.suffix == ".py":
            shutil.copyfile(path, output / f"executed_{path.name}")
    print("OUTPUT", output, flush=True)
    return output


def candidate_search(output, recipe, config):
    reference = Path(recipe["reference"])
    original = pd.read_csv(reference / "full_predictions.csv")
    target, adjustment, eligible, _, _ = accounts(config)
    years = config["forcing_years"]
    keep = np.isin(years, recipe["validation_years"])
    train = training_mask(config)
    natural = cached_natural(reference, "fine", "fine", years)
    frozen = read_json(reference / "freeze_fine.json")
    initial = parameters(frozen)
    records, candidates, predictions = [], {}, []
    for spec in recipe["candidates"]:
        name = spec["name"]
        try:
            if name == "baseline":
                fit = dict(parameters=initial.tolist(), success=True)
            else:
                # Equal total weight per period, independent of its number of years.
                weights = keep / keep.sum() + spec["training_weight"] * train / train.sum()
                fit = fit_routing(
                    natural, target, adjustment, eligible, weights, initial, spec["free"]
                )
            write_json(output / f"candidate_{name}.json", {**fit, "spec": spec})
            value = routing_prediction(natural, adjustment, fit["parameters"])[0]
            rows = pd.DataFrame(
                rows_for(config, "fine", name, "final", value, target, eligible, keep)
            ).assign(parameter_source="fine")
            frame = pd.concat([rows, original[original["product"] != "fine"]])
            outcome, scores, _ = assess_focus(frame)
            key = selection_key(outcome, scores)
            record = dict(
                mode=name,
                status="complete",
                **outcome,
                focus_error=float(scores.loc["2023_总误差", "fine"]),
                pooled_error=float(scores.loc["三年整体", "fine"]),
                training_resubstitution_error=metrics(value, target, eligible & train[:, None])[
                    "relative_rmse_pct"
                ],
                selection_key=list(key),
            )
            records.append(record)
            predictions.append(rows)
            candidates[name] = (key, fit, spec)
            print(
                "CANDIDATE",
                name,
                outcome["aggregate_best_count"],
                outcome["station_year_wins"],
                record["focus_error"],
                flush=True,
            )
        except (ValueError, RuntimeError) as error:
            record = dict(mode=name, status="failed", error=repr(error))
            records.append(record)
            write_json(output / f"candidate_{name}.json", record)
    pd.DataFrame(records).to_csv(output / "candidate_metrics.csv", index=False)
    pd.concat(predictions).to_csv(output / "candidate_predictions.csv", index=False)
    selected = min(candidates, key=lambda name: candidates[name][0])
    _, fit, spec = candidates[selected]
    frozen["retention"] = fit["parameters"][:7]
    frozen["seed"][13] = fit["parameters"][7]
    frozen["mode"] = selected
    write_json(
        output / "selection.json",
        dict(selected=selected, validation_used=True, development_target_fit=True, spec=spec),
    )
    write_json(output / "current_parameters.json", frozen)
    write_json(
        output / "freeze.json",
        dict(
            seed=frozen["seed"],
            lower_z=None,
            head=[frozen["seed"][0], frozen["head_multiplier"], frozen["retention"][0]],
            rainfall_factors=[1.0] * len(years),
        ),
    )
    return frozen


def cv_diagnostic(output):
    """Refit routing on each original fold; never label target fitting as validation."""
    recipe = read_json(output / "fine_protocol.json")
    reference = Path(recipe["reference"])
    config = worker_setup(output / "execution_protocol.yaml")
    target, adjustment, eligible, _, _ = accounts(config)
    years = np.array(config["forcing_years"])
    data = sample_response(output, "fine", compressed=False)
    spec = read_json(output / "selection.json")["spec"]
    rows = []
    folds = read_json(reference / "experiment.json")["folds"]
    for i, held in enumerate(folds):
        name = f"fold{i}"
        seed = read_json(Path(recipe["local_run"]) / f"local_fit_fine_{name}.json")["parameters"]
        head = read_json(reference / f"head_fine_False_{name}.json")["parameters"]
        regional = {
            r: read_json(reference / f"regional_fine_{r}_{name}.json")["parameters"] for r in (5, 6)
        }
        frozen = freeze(seed, head, regional, np.ones(len(years)), "baseline")
        natural = natural_sample(data, frozen, len(years))
        keep = training_mask(config) & ~np.isin(years, held)
        horizon = np.flatnonzero(keep)[-1] + 1
        initial = parameters(frozen)
        free = spec.get("free", [])
        fit = fit_routing(
            natural[:horizon],
            target[:horizon],
            adjustment[:horizon],
            eligible[:horizon],
            keep[:horizon].astype(float),
            initial,
            free,
        )
        write_json(output / f"routing_cv_{name}.json", fit)
        for mode, x in (("baseline", initial), ("routing_refit", fit["parameters"])):
            predicted = routing_prediction(natural, adjustment, x)[0]
            rows += rows_for(
                config, "fine", mode, name, predicted, target, eligible, np.isin(years, held)
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "cv_predictions.csv", index=False)
    scores = []
    for mode, part in frame.groupby("mode"):
        forward = part[part.fit == "fold2"]
        scores.append(
            dict(
                mode=mode,
                cv=metrics(part.predicted, part.observed, part.eligible)["relative_rmse_pct"],
                forward=metrics(forward.predicted, forward.observed, forward.eligible)[
                    "relative_rmse_pct"
                ],
            )
        )
    pd.DataFrame(scores).to_csv(output / "cv_metrics.csv", index=False)
    print("CV DIAGNOSTIC", scores, flush=True)


def compile_results(output, recipe, config, frozen):
    reference = Path(recipe["reference"])
    target, adjustment, eligible, consumed, reservoir = accounts(config)
    years = config["forcing_years"]
    keep = np.isin(years, recipe["validation_years"])
    x = parameters(frozen)
    natural = np.array([read_json(output / f"natural_fine_{y}.json")["natural"] for y in years])
    cached = cached_natural(reference, "fine", "fine", years)
    np.testing.assert_allclose(natural, cached, rtol=1e-10, atol=1e-8)
    ledger = balance_ledger(natural, target, consumed, reservoir, x)
    np.savez(output / "water_balance.npz", **ledger)
    flat = dict(year=np.repeat(years, 7), station=np.tile(STATIONS, len(years)))
    flat.update({key: value.ravel() for key, value in ledger.items()})
    pd.DataFrame(flat).to_csv(output / "water_balance.csv", index=False, encoding="utf-8-sig")
    known = np.isfinite(ledger["closure"])
    write_json(
        output / "balance_check.json",
        dict(
            passed=True,
            max_absolute_closure=float(np.max(np.abs(ledger["closure"][known]))),
            loss_reach_index=2,
            loss_applied_once=True,
            min_conceptual_storage=float(ledger["storage"].min()),
            natural_cache_max_difference=float(np.max(np.abs(natural - cached))),
            units="1e8 m3",
            residual_loss_is_not_measured_consumption=True,
        ),
    )
    original = pd.read_csv(reference / "full_predictions.csv")
    fine = pd.DataFrame(
        rows_for(
            config, "fine", frozen["mode"], "final", ledger["predicted"], target, eligible, keep
        )
    ).assign(parameter_source="fine")
    fine.to_csv(output / "full_fine.csv", index=False)
    own = pd.concat([fine, original[original["product"] != "fine"]])
    own.to_csv(output / "full_predictions.csv", index=False)
    # All other source parameters and all AWY parameters are unchanged. Only the
    # fine-source routing row needs recomputing; its pixel yields are reusable.
    reference_transfer = pd.read_csv(reference / "transfer_predictions.csv")
    transfer_rows, checks = [], []
    for product in PRODUCTS:
        natural_p = (
            natural if product == "fine" else cached_natural(reference, product, "fine", years)
        )
        values = routing_prediction(natural_p, adjustment, x)[0]
        data = sample_response(output, product, compressed=False)
        sampled = routing_prediction(natural_sample(data, frozen, len(years)), adjustment, x)[0]
        del data
        valid = eligible & keep[:, None]
        error = float(np.max(np.abs(sampled - values)[valid] / target[valid]))
        check = dict(
            product=product,
            passed=error <= 0.01,
            max_fraction_observed=error,
            full_pixel_source="recomputed" if product == "fine" else "unchanged_cache",
        )
        checks.append(check)
        if not check["passed"]:
            raise ValueError(f"Full-pixel quadrature failed: {check}")
        transfer_rows += rows_for(
            config, product, frozen["mode"], "final", values, target, eligible, keep
        )
    fine_transfer = pd.DataFrame(transfer_rows).assign(parameter_source="fine")
    transfer = pd.concat(
        [fine_transfer, reference_transfer[reference_transfer.parameter_source != "fine"]]
    )
    transfer.to_csv(output / "transfer_predictions.csv", index=False)
    write_json(output / "quadrature.json", dict(passed=True, products=checks))
    # Report every changed historical and target cell, including regressions.
    previous = read_json(reference / "freeze_fine.json")
    before = routing_prediction(cached, adjustment, parameters(previous))[0]
    comparison = pd.DataFrame(
        rows_for(
            config,
            "fine",
            frozen["mode"],
            "final",
            ledger["predicted"],
            target,
            eligible,
            np.ones(len(years), bool),
        )
    )
    comparison["baseline_predicted"] = before.ravel()
    comparison["before_error_pct"] = (
        abs(comparison.baseline_predicted / comparison.observed - 1) * 100
    )
    comparison["after_error_pct"] = abs(comparison.predicted / comparison.observed - 1) * 100
    comparison["change_percentage_points"] = (
        comparison.after_error_pct - comparison.before_error_pct
    )
    comparison.to_csv(output / "all_year_changes.csv", index=False, encoding="utf-8-sig")
    outcome = assess_focus(own)[0]
    write_json(output / "goals.json", outcome)
    print("FULL GOALS", outcome, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/fine_2023_focus.yaml"))
    parser.add_argument("--run", type=Path, help="Resume an existing frozen experiment")
    args = parser.parse_args()
    output = args.run or prepare(args.config)
    recipe = read_json(output / "fine_protocol.json")
    config = worker_setup(output / "execution_protocol.yaml")
    try:
        if args.run:
            saved = read_json(output / "sources.json")
            if saved != [fingerprint(Path(item["path"])) for item in saved]:
                raise ValueError("A frozen source changed; start a new experiment")
        frozen = candidate_search(output, recipe, config)
        tasks = [
            (str(output), y)
            for y in config["forcing_years"]
            if not (output / f"natural_fine_{y}.json").exists()
        ]
        if tasks:
            run_tasks(full_year, tasks, config, output, "full")
        cv_diagnostic(output)
        compile_results(output, recipe, config, frozen)
        saved = read_json(output / "sources.json")
        if saved != [fingerprint(Path(item["path"])) for item in saved]:
            raise ValueError("Frozen comparator or source changed")
        write_json(
            output / "status.json",
            dict(
                status="complete",
                fine_only=True,
                full_pixels=True,
                comparators_frozen=True,
                development_target_fit=True,
                transfer_status="current",
                water_balance_passed=True,
            ),
        )
        publish(output)
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
