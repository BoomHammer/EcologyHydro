"""Audit frozen errors, select finite candidates on training blocks, then freeze nine pairs."""

import argparse
import shutil
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from calibrate_long_record import full_models
from calibrate_regional_water import block_data, open_inputs
from run_landcover_comparison import input_paths, run_pool, validate_prepared, worker_setup

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import (
    accounts,
    load_protocol,
    metrics,
    phase_for,
    prediction_rows,
    resolve,
    summarize_rows,
    training_mask,
)
from ecologyhydro.comparison_candidates import candidate_sample, candidate_tables, fit_spec
from ecologyhydro.config import project_root
from ecologyhydro.long_record import ENDPOINTS, REACH_MAP, STATIONS
from ecologyhydro.regional_transfer import evaluate, fit_transfer, training_data
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows
from ecologyhydro.water_balance import REGIONS


def load_samples(baseline, product):
    with np.load(baseline / f"samples_{product}.npz") as archive:
        return {k: archive[k] for k in archive.files}


def differences(values):
    """Reach increments, preserving missing values rather than filling them."""
    return np.diff(values, axis=-1, prepend=0)


def audit(config, baseline, output):
    target, _, _, consumption, storage = accounts(config)
    frame = pd.read_csv(baseline / "predictions.csv")
    scored = frame[frame.scored].copy()
    scored["squared_error"] = scored.error**2
    scored["relative_squared_error"] = (scored.error / scored.observed) ** 2
    group = scored.groupby(["parameter_source", "landcover", "phase"])
    for field in ("squared_error", "relative_squared_error"):
        scored[f"{field}_phase_fraction"] = scored[field] / group[field].transform("sum")
    scored.to_csv(output / "error_contributions.csv", index=False)
    reaches, climate, sensitivities = [], [], []
    years = config["forcing_years"]
    for product in config["products"]:
        data = load_samples(baseline, product)
        fits = read_json(baseline / f"parameters_{product}.json")
        x = np.array(fits["final"]["parameters"])
        # Fixed-parameter perturbations are diagnostics, never candidate selection.
        for index, name in enumerate([f"Z_{i}" for i in range(7)] + ["kc_multiplier", "L"]):
            for sign in (-1, 1):
                altered = x.copy()
                altered[index] += sign * (1 if name == "L" else x[index] * 0.05)
                delta = evaluate(altered, data, "reach7", len(years)) - evaluate(
                    x, data, "reach7", len(years)
                )
                sensitivities.extend(
                    dict(
                        product=product,
                        parameter=name,
                        direction=sign,
                        year=y,
                        station=s,
                        delta_runoff=float(delta[yi, si]),
                    )
                    for yi, y in enumerate(years)
                    for si, s in enumerate(STATIONS)
                )
        for yi, year in enumerate(years):
            for ri, region in enumerate(REGIONS[:7]):
                mask = (data["groups"] // 11 == yi) & (REACH_MAP[data["zone"]] == ri)
                w = data["weights"][mask]
                climate.append(
                    dict(
                        product=product,
                        year=year,
                        region=region,
                        area_km2=float(w.sum() * 1e5),
                        p_mm=float(np.average(data["p"][mask], weights=w)),
                        pet_mm=float(np.average(data["et"][mask], weights=w)),
                        method="stratified_quadrature_not_full_pixel",
                    )
                )
        for source in config["products"]:
            loss = read_json(baseline / f"parameters_{source}.json")["final"]["parameters"][-1]
            pair = frame[(frame.landcover == product) & (frame.parameter_source == source)]
            for yi, year in enumerate(years):
                subset = pair[pair.year == year].set_index("station").loc[STATIONS]
                observed_inc = differences(target[yi])
                awy = differences(subset.awy_yield.to_numpy())
                pred = differences(subset.predicted.to_numpy())
                error = subset.predicted.to_numpy() - target[yi]
                for ri, region in enumerate(REGIONS[:7]):
                    reaches.append(
                        dict(
                            parameter_source=source,
                            landcover=product,
                            year=year,
                            region=region,
                            station=STATIONS[ri],
                            awy_increment=awy[ri],
                            observed_increment=observed_inc[ri],
                            predicted_increment=pred[ri],
                            consumption=consumption[yi, ri],
                            storage=storage[yi, ri],
                            loss=loss if ri == 2 else 0,
                            upstream_error=error[ri - 1] if ri else 0,
                            local_error=pred[ri] - observed_inc[ri],
                            cumulative_error=error[ri],
                        )
                    )
    pd.DataFrame(reaches).to_csv(output / "reach_balance.csv", index=False)
    pd.DataFrame(climate).to_csv(output / "climate_sampled.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(output / "parameter_sensitivity.csv", index=False)


def cv_worker(config_path, output_string, product):
    config, output = worker_setup(config_path), Path(output_string)
    recipe = read_json(output / "experiment.json")
    baseline = resolve(config, recipe["baseline_run"])
    target, adjustment, eligible, _, _ = accounts(config)
    train = training_mask(config)
    years = np.array(config["forcing_years"])[train]
    # Remove validation climate, observations AND accounts before any CV callback.
    data = training_data(load_samples(baseline, product), train)
    target, adjustment, eligible = (a[train] for a in (target, adjustment, eligible))
    rows, fit_records = [], {}
    for name, spec in recipe["candidates"].items():
        reuse = recipe.get("reuse_baseline_experiment")
        if name == "baseline" and reuse:
            prior = resolve(config, reuse)
            old_recipe = read_json(prior / "experiment.json")
            if (
                old_recipe["folds"] != recipe["folds"]
                or old_recipe["baseline_run"] != recipe["baseline_run"]
            ):
                raise ValueError("Baseline CV reuse requires identical folds and baseline")
            old = pd.read_csv(prior / f"cv_{product}.csv")
            rows.extend(old[old.candidate == "baseline"].to_dict("records"))
            fit_records[name] = read_json(prior / f"cv_fits_{product}.json")[name]
            print("Reused identical baseline CV", product, flush=True)
            continue
        fit_records[name] = []
        candidate = candidate_sample(data, recipe, spec, product)
        for fold, held in enumerate(recipe["folds"]):
            hold = np.isin(years, held)
            fit = fit_transfer(candidate, target, adjustment, eligible, ~hold, **fit_spec(spec))
            fit["training_years"] = years[~hold].tolist()
            fit["held_years"] = years[hold].tolist()
            fit_records[name].append(fit)
            pred = (
                evaluate(
                    fit["parameters"], candidate, spec["model"], len(years), kc_cap=spec["kc_cap"]
                )
                - adjustment
            )
            for yi in np.flatnonzero(hold):
                for si, station in enumerate(STATIONS):
                    if eligible[yi, si]:
                        rows.append(
                            dict(
                                candidate=name,
                                product=product,
                                fold=fold,
                                year=int(years[yi]),
                                station=station,
                                observed=target[yi, si],
                                predicted=pred[yi, si],
                            )
                        )
            print("CV", product, name, fold, flush=True)
    pd.DataFrame(rows).to_csv(output / f"cv_{product}.csv", index=False)
    write_json(output / f"cv_fits_{product}.json", fit_records)


def choose(frame, recipe):
    frame = frame.copy()
    expected_years = {y for fold in recipe["folds"] for y in fold}
    if set(frame.year) != expected_years:
        raise ValueError("Candidate selection must contain only declared calibration folds")
    if set(frame.candidate) != set(recipe["candidates"]):
        raise ValueError("Incomplete candidate set")
    keys = ["product", "year", "station"]
    reference = frame[frame.candidate == "baseline"].set_index(keys).sort_index()
    for _, part in frame.groupby("candidate"):
        current = part.set_index(keys).sort_index()
        if not current.index.is_unique or not current.index.equals(reference.index):
            raise ValueError("Candidates must use identical, unique station-years")
        if not np.array_equal(current.observed.to_numpy(), reference.observed.to_numpy()):
            raise ValueError("Candidate observations differ")
    frame["relative_square"] = ((frame.predicted - frame.observed) / frame.observed) ** 2
    scores = frame.groupby("candidate").relative_square.mean()
    baseline = frame[frame.candidate == "baseline"]
    decisions = {}
    for name in recipe["candidates"]:
        candidate = frame[frame.candidate == name]
        checks = {}
        for axis in ("product", "fold", "year", "station"):
            checks[axis] = int(
                (
                    candidate.groupby(axis).relative_square.mean()
                    < baseline.groupby(axis).relative_square.mean() - 1e-12
                ).sum()
            )
        passes = (
            scores[name] < scores["baseline"] - 1e-12
            and checks["product"] == frame["product"].nunique()
            and checks["fold"] == len(recipe["folds"])
            and checks["year"] >= recipe["selection"]["minimum_improved_years"]
            and checks["station"] >= recipe["selection"]["minimum_improved_stations"]
        )
        decisions[name] = dict(
            relative_rmse_pct=float(np.sqrt(scores[name]) * 100),
            improved_counts=checks,
            passes=bool(passes),
        )
    allowed = [k for k, v in decisions.items() if v["passes"]]
    selected = min(allowed, key=lambda k: scores[k]) if allowed else "baseline"
    return dict(selected=selected, candidates=decisions, validation_used_for_selection=False)


def final_fit_worker(config_path, output_string, product):
    config, output = worker_setup(config_path), Path(output_string)
    recipe = read_json(output / "experiment.json")
    selected = read_json(output / "selection.json")["selected"]
    spec = recipe["candidates"][selected]
    baseline = resolve(config, recipe["baseline_run"])
    target, adjustment, eligible, _, _ = accounts(config)
    if selected == "baseline":
        fit = read_json(baseline / f"parameters_{product}.json")["final"].copy()
        fit.setdefault("kc_cap", 1.3)
    else:
        data = load_samples(baseline, product)
        data = candidate_sample(data, recipe, spec, product)
        fit = fit_transfer(
            data, target, adjustment, eligible, training_mask(config), **fit_spec(spec)
        )
    fit["training_years"] = np.array(config["forcing_years"])[training_mask(config)].tolist()
    fit["model"] = spec["model"]
    write_json(output / f"parameters_{product}.json", {"final": fit})
    tables = read_json(baseline / f"tables_{product}.json")
    write_json(output / f"tables_{product}.json", candidate_tables(tables, recipe, spec, product))
    print("Frozen final parameters", product, selected, flush=True)


def attribute_audit(config, product, table, fit):
    """Exact class area and effective root/Kc summaries on the entire 2024 grid."""
    rows = []
    totals = {}
    paths = input_paths(config, product, 2024)
    with ExitStack() as stack:
        datasets = open_inputs(stack, paths)
        gt = datasets["zone"].GetGeoTransform()
        area = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e6
        for window in windows(datasets["zone"]):
            data, _ = block_data(datasets, window, table)
            reach = REACH_MAP[data["zone"]]
            for ri, code in np.unique(np.column_stack([reach, data["code"]]), axis=0):
                mask = (reach == ri) & (data["code"] == code)
                kc = data["kc"][mask]
                veg = data["veg"][mask] == 1
                root, soil = data["root_depth"][mask], data["soil"][mask]
                effective = np.where(veg, np.minimum(kc * fit["parameters"][-2], 1.3), kc)
                values = np.array(
                    [
                        mask.sum(),
                        (veg & (kc * fit["parameters"][-2] >= 1.3)).sum(),
                        (veg & (soil < root)).sum(),
                        effective.sum(),
                        np.minimum(root, soil).sum(),
                        data["awc"][mask].sum(),
                        (
                            veg
                            & (
                                1.25
                                + fit["parameters"][int(ri)] * data["awc"][mask] / data["p"][mask]
                                >= 5
                            )
                        ).sum(),
                    ]
                )
                key = (int(ri), int(code))
                totals[key] = totals.get(key, np.zeros(7)) + values
    lookup = {int(r["lucode"]): r for r in table}
    for (ri, code), v in totals.items():
        rows.append(
            dict(
                product=product,
                year=2024,
                region=REGIONS[ri],
                code=code,
                area_km2=v[0] * area,
                kc_clipped_fraction=v[1] / v[0],
                soil_limited_fraction=v[2] / v[0],
                effective_kc=v[3] / v[0],
                effective_root_mm=v[4] / v[0],
                awc_mm=v[5] / v[0],
                omega_capped_fraction=v[6] / v[0],
                base_kc=lookup[code]["kc"],
                root_mm=lookup[code]["root_depth"],
                veg=lookup[code]["lulc_veg"],
                kc_multiplier=fit["parameters"][-2],
            )
        )
    return rows


def final_evaluate_worker(config_path, output_string, product):
    config, output = worker_setup(config_path), Path(output_string)
    recipe = read_json(output / "experiment.json")
    baseline = resolve(config, recipe["baseline_run"])
    selected = read_json(output / "selection.json")["selected"]
    if selected == "baseline" and recipe.get("reuse_baseline_experiment"):
        prior = resolve(config, recipe["reuse_baseline_experiment"])
        if read_json(prior / "summary.json")["selection"]["selected"] != "baseline":
            raise ValueError("Can only reuse identical baseline evaluation")
        artifacts = []
        for stem, suffix in (
            ("predictions", "csv"),
            ("awy_all_stations", "csv"),
            ("attributes", "csv"),
            ("climate", "csv"),
            ("quadrature", "json"),
        ):
            source = prior / f"{stem}_{product}.{suffix}"
            shutil.copyfile(source, output / source.name)
            artifacts.append(fingerprint(source))
        write_json(output / f"reused_evaluation_{product}.json", {"sources": artifacts})
        print("Reused verified baseline full pixels", product, flush=True)
        return
    years = config["forcing_years"]
    tables = read_json(output / f"tables_{product}.json")
    fits = {p: read_json(output / f"parameters_{p}.json")["final"] for p in config["products"]}
    cases = {p: (fit["model"], fit) for p, fit in fits.items()}
    data = load_samples(baseline, product)
    data = candidate_sample(data, recipe, recipe["candidates"][selected], product)
    approximate = {
        p: evaluate(f["parameters"], data, f["model"], len(years), kc_cap=f["kc_cap"])
        for p, f in fits.items()
    }
    del data
    baseline_fit = read_json(baseline / f"parameters_{product}.json")["final"]
    original_tables = read_json(baseline / f"tables_{product}.json")
    pd.DataFrame(attribute_audit(config, product, original_tables["2024"], baseline_fit)).to_csv(
        output / f"attributes_{product}.csv", index=False
    )
    natural = {p: [] for p in fits}
    climate, raw = [], []
    from osgeo import gdal

    with gdal.Open(str(resolve(config, config["paths"]["zones"]))) as ds:
        gt = ds.GetGeoTransform()
        volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
    for year in years:
        official, weather = full_models(
            input_paths(config, product, year), tables[str(year)], cases, volume, all_stations=True
        )
        for ri, region in enumerate(REGIONS[:7]):
            area = weather["area_km2"][ri]
            climate.append(
                dict(
                    product=product,
                    year=year,
                    region=region,
                    area_km2=area,
                    p_mm=weather["p"][ri] * 1e5 / area,
                    pet_mm=weather["et"][ri] * 1e5 / area,
                )
            )
        for source, values in official.items():
            natural[source].append(values[ENDPOINTS])
            for si, station in enumerate(config["evaluation"]["all_stations"]):
                raw.append(
                    dict(
                        parameter_source=source,
                        landcover=product,
                        year=year,
                        phase=phase_for(config, year),
                        station=station,
                        awy_yield=values[si],
                        meaning="cumulative_AWY_yield_not_managed_station_runoff",
                    )
                )
        print("Full pixels", product, year, flush=True)
    target, adjustment, eligible, _, _ = accounts(config)
    rows, checks = [], {}
    for source, fit in fits.items():
        produced = np.array(natural[source])
        before = produced - np.array([0, 0, 1, 1, 1, 1, 1]) * fit["parameters"][-1]
        checks[source] = float(
            np.max(np.abs(before[eligible] - approximate[source][eligible]) / target[eligible])
        )
        rows.extend(
            prediction_rows(
                config, source, product, before - adjustment, produced, target, adjustment, eligible
            )
        )
    pd.DataFrame(rows).to_csv(output / f"predictions_{product}.csv", index=False)
    pd.DataFrame(raw).to_csv(output / f"awy_all_stations_{product}.csv", index=False)
    pd.DataFrame(climate).to_csv(output / f"climate_{product}.csv", index=False)
    write_json(output / f"quadrature_{product}.json", checks)
    if max(checks.values()) > config["evaluation"]["quadrature_max_fraction_observed"]:
        raise ValueError(f"Full pixel tolerance exceeded: {checks}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    recipe_path = project_root() / args.config
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    config = load_protocol(recipe["baseline_config"])
    validate_prepared(config)
    baseline = resolve(config, recipe["baseline_run"])
    if read_json(baseline / "summary.json")["status"] != "complete":
        raise ValueError("Baseline must be complete")
    train_years = np.array(config["forcing_years"])[training_mask(config)].tolist()
    if sorted(y for fold in recipe["folds"] for y in fold) != train_years:
        raise ValueError("Folds must partition calibration years exactly once")
    output = resolve(config, recipe["output_root"]) / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", recipe)
    write_json(
        output / "protocol.json",
        {k: v for k, v in config.items() if k not in ("root", "config_path")},
    )
    files = [
        recipe_path,
        Path(__file__),
        config["config_path"],
        project_root() / "src/ecologyhydro/regional_transfer.py",
        project_root() / "scripts/calibrate_long_record.py",
        baseline / "manifest.json",
    ]
    files += [baseline / f"samples_{p}.npz" for p in config["products"]]
    files.append(project_root() / "src/ecologyhydro/comparison_candidates.py")
    if recipe.get("reuse_baseline_experiment"):
        prior = resolve(config, recipe["reuse_baseline_experiment"])
        if read_json(prior / "summary.json")["status"] != "complete":
            raise ValueError("Reused experiment must be complete")
        files += [prior / f"cv_{p}.csv" for p in config["products"]]
        files += [prior / f"cv_fits_{p}.json" for p in config["products"]]
        files.extend([prior / "experiment.json", prior / "manifest.json", prior / "summary.json"])
    write_json(output / "manifest.json", {"sources": [fingerprint(p) for p in files]})
    print("Run directory:", output, flush=True)
    if recipe.get("reuse_baseline_experiment"):
        prior = resolve(config, recipe["reuse_baseline_experiment"])
        artifacts = []
        for name in (
            "error_contributions",
            "reach_balance",
            "climate_sampled",
            "parameter_sensitivity",
            "class_yields",
        ):
            source = prior / f"{name}.csv"
            shutil.copyfile(source, output / source.name)
            artifacts.append(fingerprint(source))
        write_json(output / "reused_audit.json", {"sources": artifacts})
    else:
        audit(config, baseline, output)
    performance = {"cv": run_pool(cv_worker, config, output)}
    cv = pd.concat([pd.read_csv(output / f"cv_{p}.csv") for p in config["products"]])
    cv.to_csv(output / "cv_predictions.csv", index=False)
    selection = choose(cv, recipe)
    write_json(output / "selection.json", selection)
    print("Frozen selection:", selection, flush=True)
    performance["final_fit"] = run_pool(final_fit_worker, config, output)
    # All selection and fitted parameter files exist before validation is evaluated.
    performance["full_pixels"] = run_pool(final_evaluate_worker, config, output)
    for stem in ("predictions", "awy_all_stations", "attributes", "climate"):
        pd.concat([pd.read_csv(output / f"{stem}_{p}.csv") for p in config["products"]]).to_csv(
            output / f"{stem}.csv", index=False
        )
    predictions = pd.read_csv(output / "predictions.csv")
    summaries = []
    (output / "pairs").mkdir()
    for (source, product), pair in predictions.groupby(["parameter_source", "landcover"]):
        pair.to_csv(output / "pairs" / f"{source}__{product}.csv", index=False)
        summaries.extend(summarize_rows(pair.to_dict("records")))
    pd.DataFrame(summaries).to_csv(output / "metrics.csv", index=False)
    comparison = []
    for label, frame in (
        ("baseline", pd.read_csv(baseline / "predictions.csv")),
        ("selected", predictions),
    ):
        for (source, product), pair in frame.groupby(["parameter_source", "landcover"]):
            for period, selected_years in (("2024", [2024]), ("2022_2024", [2022, 2023, 2024])):
                part = pair[pair.year.isin(selected_years)]
                comparison.append(
                    dict(
                        version=label,
                        parameter_source=source,
                        landcover=product,
                        period=period,
                        **metrics(part.predicted, part.observed, part.scored),
                    )
                )
    pd.DataFrame(comparison).to_csv(output / "validation_comparison.csv", index=False)
    write_json(
        output / "summary.json",
        dict(
            status="complete",
            selection=selection,
            performance=performance,
            validation="reused_exploratory_not_independent",
        ),
    )
    print("Completed:", output, flush=True)


if __name__ == "__main__":
    main()
