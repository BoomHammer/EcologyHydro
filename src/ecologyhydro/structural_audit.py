"""Diagnose structural water-budget constraints without fitting new parameters."""

import math
import time
from pathlib import Path

from ecologyhydro.aggregation import read_observations, write_csv
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import read_csv


def reach_budgets(stations, modeled, observed):
    """Difference cumulative volumes; negative observed increments are meaningful.

    Missing observations invalidate both adjacent differences. They never become
    zero, and the next reach never silently skips over a missing station.
    """
    if len(set(stations)) != len(stations):
        raise ValueError("Duplicate station in upstream order")
    previous_name, previous_model, previous_observation = "headwaters", 0.0, 0.0
    rows = []
    for name in stations:
        model, observation = modeled[name], observed[name]
        if not math.isfinite(model) or model < previous_model - 1e-6:
            raise ValueError("Cumulative unadjusted AWY must be finite and nondecreasing")
        if observation is not None and (not math.isfinite(observation) or observation < 0):
            raise ValueError("Invalid observed station volume")
        increment = (
            observation - previous_observation
            if observation is not None and previous_observation is not None
            else None
        )
        local_yield = model - previous_model
        rows.append(
            {
                "upstream": previous_name,
                "downstream": name,
                "local_awy_1e8_m3": local_yield,
                "observed_net_increment_1e8_m3": increment,
                "unexplained_difference_1e8_m3": local_yield - increment
                if increment is not None
                else None,
                "observed_decrease": increment < 0 if increment is not None else None,
                "meaning": "net_reach_balance_not_observed_local_natural_runoff",
            }
        )
        previous_name, previous_model, previous_observation = name, model, observation
    return rows


def indexed(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate diagnostic record: {key}")
        result[key] = row
    return result


def main():
    from ecologyhydro.baseline import refresh

    started = time.perf_counter()
    root, config = project_root(), load_config()
    years = config.study.calibration_years
    if not years or set(years) & set(config.study.validation_years):
        raise ValueError("Require a nonempty training period disjoint from validation")
    baseline = refresh()
    paths = {
        "models": baseline / "stations.csv",
        "bounds": baseline / "headwater_bounds.csv",
        "baseline": baseline / "manifest.json",
        "observations": config.paths.hydrology / "实测年径流量2018-2023.csv",
    }
    models = indexed(
        [
            row
            for row in read_csv(paths["models"])
            if row["variant"] == "rotation" and int(row["year"]) in years
        ],
        ("year", "station"),
    )
    bounds = indexed(
        [
            row
            for row in read_csv(paths["bounds"])
            if row["variant"] == "rotation"
            and row["landcover"] == "fine"
            and int(row["year"]) in years
        ],
        ("year", "station"),
    )
    area = float(bounds[str(min(years)), "唐乃亥"]["area_km2"])
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Invalid Tangnaihai diagnostic area")
    reaches, headwaters = [], []
    storage_equivalent = 0.0
    for year in sorted(years):
        observations = read_observations(paths["observations"], year)
        modeled = {
            name: float(models[str(year), name]["awy_yield_1e8_m3"])
            for name in config.study.station_ids
        }
        observed = {name: observations[name]["value"] for name in config.study.station_ids}
        for row in reach_budgets(config.study.station_ids, modeled, observed):
            row.update(year=year, landcover="fine", variant="rotation")
            row["run_id"] = models[str(year), row["downstream"]]["run_id"]
            reaches.append(row)
        q = observed["唐乃亥"]
        if q is None:
            raise ValueError("Storage-only diagnostic requires all headwater observations")
        budget = bounds[str(year), "唐乃亥"]
        p = float(budget["precipitation_mm"])
        et0 = float(budget["reference_et0_mm"])
        lower_bound = float(budget["lower_bound_1e8_m3"])
        if not all(math.isfinite(v) and v >= 0 for v in (p, et0, lower_bound)) or et0 == 0:
            raise ValueError("Invalid climate or necessary yield bound")
        volume = modeled["唐乃亥"]
        residual = volume - q
        storage_equivalent += residual
        headwaters.append(
            {
                "year": year,
                "area_km2": area,
                "precipitation_mm": p,
                "reference_et0_mm": et0,
                "awy_yield_1e8_m3": volume,
                "observed_1e8_m3": q,
                "awy_to_observed_ratio": volume / q,
                "awy_aet_mm": p - volume * 1e5 / area,
                "P_minus_Q_mm_not_observed_AET": p - q * 1e5 / area,
                "annual_residual_1e8_m3": residual,
                "cumulative_storage_only_equivalent_1e8_m3": storage_equivalent,
                "cumulative_storage_only_equivalent_mm": storage_equivalent * 1e5 / area,
                "necessary_yield_bound_1e8_m3": lower_bound,
                "bound_minus_raw_observation_1e8_m3": lower_bound - q,
                "meaning": "conditional_diagnostic_not_estimated_storage_or_naturalized_target",
            }
        )
    # Telescoping is a check on the decomposition, not a claim about natural runoff.
    for year in years:
        rows = [row for row in reaches if row["year"] == year]
        total = sum(row["local_awy_1e8_m3"] for row in rows)
        expected = float(models[str(year), config.study.station_ids[-1]]["awy_yield_1e8_m3"])
        if not math.isclose(total, expected, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("Reach volumes fail cumulative conservation")
    output = root / "project/diagnostics/current_structure"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "reach_budgets.csv", reaches)
    write_csv(output / "headwater_budget.csv", headwaters)
    decreases = [row for row in reaches if row["observed_decrease"] is True]
    write_json(
        output / "manifest.json",
        {
            "sources": [fingerprint(path) for path in paths.values()]
            + [fingerprint(Path(__file__)), fingerprint(root / "config.yaml")],
            "calibration_years": years,
            "validation_observations_used": False,
            "new_parameters_fitted": 0,
            "new_model_runs": 0,
            "observed_decrease_reaches": len(decreases),
            "reach_records": len(reaches),
            "storage_only_equivalent_assumption": (
                "Keep modeled P and AET fixed; set all other omitted net terms to zero ONLY "
                "in this hypothetical diagnostic. Residual is not measured storage."
            ),
            "limitation": (
                "Explicit selected baseline matched to current M2; climate bounds recomputed "
                "on the same catchment. Conditional diagnosis, not a landcover accuracy test."
            ),
            "duration_seconds": time.perf_counter() - started,
            "outputs": [
                fingerprint(output / name) for name in ("reach_budgets.csv", "headwater_budget.csv")
            ],
        },
    )
    print(f"Structural audit: {len(reaches)} reaches; {len(decreases)} observed decreases")
    print(f"Storage-only equivalent: {storage_equivalent:.2f} x 1e8 m3")
    print(output)


if __name__ == "__main__":
    main()
