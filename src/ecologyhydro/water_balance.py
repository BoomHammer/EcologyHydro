"""Explicit partial runoff restoration hypotheses, never automatic naturalization."""

import csv
import math
from pathlib import Path

from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.simulation import write_json

REGIONS = [
    "龙羊峡以上",
    "龙羊峡至兰州",
    "兰州至头道拐",
    "头道拐至龙门",
    "龙门至三门峡",
    "三门峡至花园口",
    "花园口以下",
    "黄河内流区",
]
ENDPOINTS = {"贵得": 1, "兰州": 2, "头道拐": 3, "龙门": 4, "三门峡": 5, "花园口": 6}
FIELDS = ["地表水供水", "地表水耗水", "地下水供水", "地下水耗水", "大中型水库年蓄水变化量"]


def partial_target(q, surface, groundwater, storage, alpha, beta):
    """storage=end-start; alpha/beta are declared assumptions, not estimated fractions."""
    values = (q, surface, groundwater, storage, alpha, beta)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite water balance input")
    if q <= 0 or surface < 0 or groundwater < 0 or not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError("Invalid consumption, observation or fraction")
    return q + alpha * surface + beta * groundwater + storage


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    from ecologyhydro.aggregation import read_observations, write_csv
    from ecologyhydro.baseline import current_models

    root, config = project_root(), load_config()
    output = root / "project/diagnostics/current_water_balance"
    output.mkdir(parents=True, exist_ok=True)
    sources = [config.paths.hydrology / f"{name}2018-2023.csv" for name in FIELDS]
    tables = {}
    for name, path in zip(FIELDS, sources, strict=True):
        rows = read_csv(path)
        table = {r["水资源二级区"]: r for r in rows}
        if len(table) != len(rows) or set(table) != set(REGIONS):
            raise ValueError("Water account region mismatch/duplicates")
        tables[name] = table
    observation_path = config.paths.hydrology / "实测年径流量2018-2023.csv"
    sources += [observation_path, Path(__file__)]
    accounts, regional, scenarios = [], [], []
    for year in config.study.calibration_years:
        for region in REGIONS:
            values = {k: float(tables[k][region][str(year)]) for k in FIELDS}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError("Missing/non-finite water account")
            for kind in ("地表水", "地下水"):
                if not 0 <= values[kind + "耗水"] <= values[kind + "供水"]:
                    raise ValueError("Consumption exceeds supply")
            regional.append({"year": year, "region": region, **values})
        observations = read_observations(observation_path, year)
        for station in config.study.station_ids:
            q = observations[station]["value"]
            count = ENDPOINTS.get(station)
            regions = REGIONS[:count] if count else []
            values = {
                k: sum(float(tables[k][r][str(year)]) for r in regions) if count else None
                for k in FIELDS
            }
            accounts.append(
                {
                    "year": year,
                    "station": station,
                    "observed_1e8_m3": q,
                    "regions": ";".join(regions),
                    **values,
                    "mapping": "endpoint_proxy" if count else "not_assignable",
                }
            )
            for alpha in (0.0, 0.5, 1.0):
                for beta in (0.0, 1.0):
                    target = (
                        partial_target(
                            q,
                            values["地表水耗水"],
                            values["地下水耗水"],
                            values["大中型水库年蓄水变化量"],
                            alpha,
                            beta,
                        )
                        if count and q is not None
                        else None
                    )
                    scenarios.append(
                        {
                            "year": year,
                            "station": station,
                            "alpha_surface_nonoverlap_assumed": alpha,
                            "beta_groundwater_same_year_effect_assumed": beta,
                            "partial_target_1e8_m3": target,
                            "eligible_for_calibration": False,
                            "unknown_terms": (
                                "transfers;groundwater_storage;other_storage;ET_overlap"
                            ),
                            "status": "conditional_partial_proxy"
                            if target is not None
                            else "not_assignable",
                        }
                    )
    models, baseline_sources, _ = current_models()
    sources += baseline_sources
    account_lookup = {(r["year"], r["station"]): r for r in accounts}
    comparisons = []
    for model in models:
        year, station = int(model["year"]), model["station"]
        account = account_lookup[year, station]
        q, y = account["observed_1e8_m3"], float(model["awy_yield_1e8_m3"])
        for scenario in scenarios:
            if (scenario["year"], scenario["station"]) != (year, station):
                continue
            target = scenario["partial_target_1e8_m3"]
            correction = target - q if target is not None else None
            raw_gap = y - q if q is not None else None
            comparisons.append(
                {
                    **scenario,
                    "landcover": model["landcover"],
                    "parameter_version": model["parameter_version"],
                    "run_id": model["run_id"],
                    "awy_yield_1e8_m3": y,
                    "observed_1e8_m3": q,
                    "raw_gap_1e8_m3": raw_gap,
                    "known_scenario_correction_1e8_m3": correction,
                    "remaining_gap_1e8_m3": y - target if target is not None else None,
                    "model_minus_same_correction_1e8_m3": y - correction
                    if correction is not None
                    else None,
                    "gap_explained_percent": correction / raw_gap * 100
                    if correction is not None and raw_gap
                    else None,
                }
            )
    write_csv(output / "regional_accounts.csv", regional)
    write_csv(output / "station_accounts.csv", accounts)
    write_csv(output / "partial_targets.csv", scenarios)
    write_csv(output / "comparisons.csv", comparisons)
    write_json(
        output / "manifest.json",
        {
            "sources": [fingerprint(p) for p in sources],
            "unit": "1e8_m3",
            "sign_storage": "end_minus_start",
            "formula": "Q + alpha*C_surface + beta*C_groundwater + delta_S_reservoir",
            "alpha": "assumed part not already represented in AWY AET; not fitted",
            "beta": "assumed same-year net river effect; not a measured depletion fraction",
            "unknown_terms_zero": False,
            "complete_naturalization": False,
            "groundwater_beta1_is_not_a_rigorous_upper_bound": True,
            "observations_restored_and_model_corrected_are_alternative_routes": True,
            "counts": {
                "accounts": len(accounts),
                "scenarios": len(scenarios),
                "comparisons": len(comparisons),
            },
            "outputs": [
                fingerprint(output / f"{name}.csv")
                for name in (
                    "regional_accounts",
                    "station_accounts",
                    "partial_targets",
                    "comparisons",
                )
            ],
        },
    )


if __name__ == "__main__":
    main()
