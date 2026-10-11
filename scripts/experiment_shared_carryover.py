"""Test one shared annual carryover coefficient on frozen AWY responses."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_shared_parameters import load_samples, sample_directory
from run_landcover_comparison import worker_setup
from scipy.optimize import minimize_scalar

from ecologyhydro.annual_routing import route
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import accounts, metrics, training_mask
from ecologyhydro.long_record import STATIONS
from ecologyhydro.regional_transfer import evaluate
from ecologyhydro.shared_biophysical import (
    load_registry,
    parameter_table,
    replace_sample_parameters,
)
from ecologyhydro.simulation import write_json


def delivered(cumulative, retention):
    local = np.diff(cumulative, axis=1, prepend=0)
    flow, change, _ = route(local, [0, 1, retention, 0])
    np.testing.assert_allclose(flow + change, local, atol=1e-9)
    return flow.cumsum(axis=1)


def fit_retention(natural, target, adjustment, valid, keep, loss):
    """Only training observations and chronological forcing through last training year."""
    horizon = np.flatnonzero(keep)[-1] + 1
    mask = valid[:horizon] & keep[:horizon, None]
    observed = target[:horizon][mask]
    known = adjustment[:horizon].copy()
    known[:, 2:] += loss

    def objective(retention):
        return np.mean(
            [
                np.mean(
                    (
                        (delivered(value[:horizon], retention)[mask] - known[mask] - observed)
                        / observed
                    )
                    ** 2
                )
                for value in natural.values()
            ]
        )

    result = minimize_scalar(objective, bounds=(0, 0.5), method="bounded")
    if not result.success:
        raise RuntimeError("Shared retention did not converge")
    candidates = [0.0, float(result.x), 0.5]
    retention = min(candidates, key=objective)
    return dict(
        retention=retention,
        objective=float(objective(retention)),
        zero_retention_objective=float(objective(0)),
        observations=int(mask.sum()),
        training_indices=np.flatnonzero(keep).tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-name", default="shared_carryover")
    args = parser.parse_args()
    run = args.run.resolve()
    config = worker_setup(run / "execution_protocol.yaml")
    recipe = read_json(run / "experiment.json")
    registry = load_registry(run / "shared_biophysical.yaml")
    for source in read_json(run / "manifest.json")["sources"]:
        if fingerprint(Path(source["path"])) != source:
            raise ValueError("Run input changed")
    output = run / args.output_name
    output.mkdir(exist_ok=False)
    years = np.asarray(config["forcing_years"])
    target, adjustment, eligible, _, _ = accounts(config)
    folds = [("final", [])] + [(f"fold{i}", held) for i, held in enumerate(recipe["folds"])]
    responses = {name: {} for name, _ in folds}
    for product in config["products"]:
        raw = load_samples(sample_directory(config, recipe, product), product)
        data = replace_sample_parameters(raw, parameter_table(registry, product, "detail"))
        for name, _ in folds:
            parameters = read_json(run / f"fit_{name}.json")["parameters"]
            natural = evaluate(parameters, data, "reach7", len(years))
            natural[:, 2:] += parameters[-1]
            responses[name][product] = natural
    full = pd.read_csv(run / "predictions.csv")
    rows, scores = [], []
    for name, held in folds:
        keep = training_mask(config) & ~np.isin(years, held)
        loss = read_json(run / f"fit_{name}.json")["parameters"][-1]
        fit = fit_retention(responses[name], target, adjustment, eligible, keep, loss)
        write_json(output / f"fit_{name}.json", fit)
        for product, natural in responses[name].items():
            if name == "final":
                selected = full[(full.landcover == product) & (full["mode"] == "detail")]
                natural = selected.pivot(index="year", columns="station", values="awy_yield")
                natural = natural.loc[years, STATIONS].to_numpy()
            for case, retention in (("awy", 0), ("carryover", fit["retention"])):
                predicted = delivered(natural, retention) - adjustment
                predicted[:, 2:] -= loss
                chosen = (
                    np.isin(years, held) if held else np.isin(years, config["years"]["validation"])
                )
                for yi in np.flatnonzero(chosen):
                    for si, station in enumerate(STATIONS):
                        rows.append(
                            dict(
                                product=product,
                                case=case,
                                fit=name,
                                year=int(years[yi]),
                                station=station,
                                predicted=predicted[yi, si],
                                observed=target[yi, si],
                                eligible=bool(eligible[yi, si]),
                            )
                        )
        print("Shared carryover", name, fit, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "predictions.csv", index=False)
    frame["period"] = np.where(frame.fit == "final", "validation", "cv")
    for keys in (("product", "case", "period"), ("product", "case", "fit")):
        for (product, case, label), group in frame.groupby(list(keys)):
            scores.append(
                dict(
                    product=product,
                    case=case,
                    period=label,
                    **metrics(group.predicted, group.observed, group.eligible),
                )
            )
    pd.DataFrame(scores).drop_duplicates().to_csv(output / "metrics.csv", index=False)
    write_json(
        output / "summary.json",
        dict(
            status="complete",
            added_parameters=1,
            shared_across_products=True,
            source_run=str(run),
            full_pixel_final=True,
            awy_parameters_refitted=False,
            equations="Q_t=(1-r)*(S_(t-1)+Y_t); S_t=r*(S_(t-1)+Y_t)",
            initial_state="Equilibrium using 2011 yield; 2011-2012 unscored warmup",
            limitations="Conceptual annual extension; neither official AWY nor measured storage",
            selection="Diagnostic; no promotion using 2022-2024",
            sources=[fingerprint(Path(__file__)), fingerprint(run / "predictions.csv")],
            references=[
                registry["sources"]["INVEST"],
                "https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/baseflow/linear-reservoir-model",
            ],
        ),
    )


if __name__ == "__main__":
    main()
