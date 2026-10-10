"""Fit the frozen headwater pilot, validate through time, and route its effect."""

import multiprocessing
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from validate_requested_years import rough_accounts

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.long_record import STATIONS, load_accounts
from ecologyhydro.monthly_pilot import annual_awy, annual_bucket, fit_model, simulate
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json

YEARS = np.arange(2013, 2024)


def worker(product, output_string):
    root, output = project_root(), Path(output_string)
    configure_threads(load_config().resources)
    forcing = {k: v.to_numpy() for k, v in pd.read_csv(output / "forcing.csv").items()}
    with np.load(output / f"units_{product}.npz") as archive:
        units = dict(archive)
    with np.load(output / f"annual_{product}.npz") as archive:
        annual = dict(archive)
    observed, _, _, adjustment, eligible = load_accounts(root, years=YEARS.tolist())
    prior = pd.read_csv(root / f"project/calibration/long_record_v1/stations_{product}.csv")
    prior = prior[(prior.model == "reach7") & (prior.account == "legacy")]
    requested = pd.read_csv(root / f"project/calibration/requested_years_v1/stations_{product}.csv")
    requested = requested[requested.account == "legacy"]
    base = {}
    for phase in ("fixed_early", "rolling"):
        for y in YEARS[YEARS <= 2022] if phase == "fixed_early" else range(2019, 2023):
            rows = prior[(prior.phase == phase) & (prior.year == y)].set_index("station")
            base[phase, y] = rows.loc[STATIONS, "predicted"].to_numpy()
    base["reused_2023", 2023] = (
        requested[requested.year == 2023].set_index("station").loc[STATIONS, "predicted"].to_numpy()
    )
    base["fixed_early", 2018] = (
        requested[requested.year == 2018].set_index("station").loc[STATIONS, "predicted"].to_numpy()
    )
    rough_guide_adjustment = rough_accounts(root)[0]["mean"]
    plans = {
        "early": YEARS <= 2017,
        **{f"before_{y}": y > YEARS for y in (2020, 2021, 2022)},
        "all_development": YEARS <= 2022,
    }
    for keep in plans.values():
        keep &= eligible[:, 0]
    models = {
        "annual_local": lambda x: annual_awy(x, annual),
        "monthly": lambda x: annual_bucket(x, forcing, units, YEARS),
    }
    fits, predictions, monthly_rows, sensitivity = {}, {}, [], []
    for model, predict in models.items():
        for name, keep in plans.items():
            fit = fit_model(predict, observed[:, 0], adjustment[:, 0], keep, model)
            fit["training_years"] = YEARS[keep].tolist()
            fit["parameter_names"] = (
                ["capacity_multiplier", "kc_multiplier", "routing_months"]
                if model == "monthly"
                else ["Z", "kc_multiplier"]
            )
            fits[f"{model}__{name}"] = fit
            predictions[model, name] = predict(fit["parameters"])
            print(product, model, name, fit["parameters"], flush=True)
            if model == "monthly" and name in ("early", "all_development"):
                result = simulate(fit["parameters"], forcing, units)
                for i, y in enumerate(forcing["year"]):
                    monthly_rows.append(
                        dict(
                            landcover=product,
                            fit=name,
                            year=int(y),
                            month=int(forcing["month"][i]),
                            **{k: float(v[i]) for k, v in result.items()},
                        )
                    )
                for initial in (0.0, 1.0):
                    changed = annual_bucket(fit["parameters"], forcing, units, YEARS, initial)
                    for yi, y in enumerate(YEARS):
                        sensitivity.append(
                            dict(
                                landcover=product,
                                model=model,
                                fit=name,
                                type="initial_soil",
                                excluded_year=np.nan,
                                initial_fraction=initial,
                                year=int(y),
                                predicted=changed[yi] - adjustment[yi, 0],
                                change_pct_observed=100
                                * (changed[yi] - predictions[model, name][yi])
                                / observed[yi, 0],
                            )
                        )
        # Delete training years only; state continuity is retained when a target is excluded.
        for excluded in YEARS[plans["early"]]:
            keep = plans["early"] & (excluded != YEARS)
            fit = fit_model(predict, observed[:, 0], adjustment[:, 0], keep, model)
            fits[f"{model}__early_without_{excluded}"] = fit
            changed = predict(fit["parameters"])
            for yi, y in enumerate(YEARS):
                sensitivity.append(
                    dict(
                        landcover=product,
                        model=model,
                        fit="early",
                        type="delete_training_year",
                        excluded_year=int(excluded),
                        initial_fraction=np.nan,
                        year=int(y),
                        predicted=changed[yi] - adjustment[yi, 0],
                        change_pct_observed=100
                        * (changed[yi] - predictions[model, "early"][yi])
                        / observed[yi, 0],
                    )
                )
    records = []
    for (phase, year), original in base.items():
        yi = int(year - 2013)
        name = (
            "all_development"
            if phase == "reused_2023"
            else "early"
            if phase == "fixed_early" or year == 2019
            else f"before_{year}"
        )
        for model in ("existing_reach7", *models):
            replacement = original.copy()
            delta = 0.0
            if model != "existing_reach7":
                account = rough_guide_adjustment if year == 2018 else adjustment[yi, 0]
                delta = predictions[model, name][yi] - account - original[0]
                replacement += delta
            for si, station in enumerate(STATIONS):
                records.append(
                    dict(
                        landcover=product,
                        model=model,
                        phase=phase,
                        year=int(year),
                        station=station,
                        observed=observed[yi, si],
                        predicted=replacement[si],
                        eligible=bool(eligible[yi, si]),
                        guide_change=delta,
                        error=replacement[si] - observed[yi, si],
                        relative_error_pct=100 * (replacement[si] / observed[yi, si] - 1),
                        training_years=";".join(map(str, YEARS[plans[name]])),
                    )
                )
    pd.DataFrame(records).to_csv(output / f"stations_{product}.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(output / f"monthly_{product}.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / f"sensitivity_{product}.csv", index=False)
    write_json(output / f"fits_{product}.json", fits)


def main():
    root = project_root()
    output = root / "project/calibration/monthly_pilot_v1"
    if (output / "run_summary.json").exists():
        raise ValueError("Completed pilot exists")
    sources = read_json(output / "inputs.json")["sources"]
    for source in sources:
        if fingerprint(Path(source["path"]))["sha256"] != source["sha256"]:
            raise ValueError(f"Frozen input changed: {source['path']}")
    configure_threads(load_config().resources)
    start, peak = time.perf_counter(), 0.0
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
            peak = max(peak, memory / 1024**2)
            if peak > 24 * 1024 or time.perf_counter() - start > 12 * 3600:
                raise RuntimeError("Resource budget exceeded")
            time.sleep(0.5)
        for process in processes:
            process.join()
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError("Pilot worker failed")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    sources += [fingerprint(Path(__file__))]
    for product in ("fine", "copernicus"):
        sources += [
            fingerprint(root / f"project/calibration/long_record_v1/stations_{product}.csv"),
            fingerprint(root / f"project/calibration/requested_years_v1/stations_{product}.csv"),
        ]
    write_json(
        output / "run_summary.json",
        dict(
            elapsed_seconds=time.perf_counter() - start,
            peak_process_tree_mib=peak,
            sources=sources,
            production_changed=False,
        ),
    )


if __name__ == "__main__":
    main()
