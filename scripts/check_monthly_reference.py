"""Audit whether the restricted annual control gives a misleading stability score.

The pilot remains frozen. This additional annual control restores the existing
AWY Z/Kc bounds, without selecting or refitting any monthly candidate.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from summarize_monthly_pilot import metrics

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.long_record import load_accounts
from ecologyhydro.monthly_pilot import annual_awy
from ecologyhydro.simulation import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    root = project_root()
    output = root / "project/calibration/monthly_pilot_v1"
    if (output / "annual_reference_audit.json").exists() and not args.rebuild:
        raise ValueError("Completed reference audit exists")
    write_json(
        output / "annual_reference_audit_protocol.json",
        dict(
            reason=(
                "Initial annual control reached both bounds; "
                "its zero delete-year spread is artificial"
            ),
            timing="Additional audit after pilot scoring; cannot count as prospective validation",
            bounds=dict(Z=[1, 30], kc=[0.7, 2]),
            source="Existing regional_transfer.fit_transfer bounds, not chosen on validation",
            monthly_candidate_changed=False,
            production_changed=False,
        ),
    )
    years = np.arange(2013, 2024)
    obs, _, _, adjustment, eligible = load_accounts(root, years=years.tolist())
    rows, fits, deletion = [], {}, []
    for product in ("fine", "copernicus"):
        with np.load(output / f"annual_{product}.npz") as archive:
            data = dict(archive)
        baseline = pd.read_csv(output / f"stations_{product}.csv")
        baseline = baseline[baseline.model == "existing_reach7"]
        plans = {
            "early": years <= 2017,
            **{f"before_{y}": years < y for y in (2020, 2021, 2022)},
            "all_development": years <= 2022,
            **{f"without_{y}": (years <= 2017) & (years != y) for y in range(2013, 2018)},
        }
        predictions = {}
        for name, keep in plans.items():
            keep &= eligible[:, 0]

            def residual(x, selected=keep, forcing_data=data):
                return (
                    annual_awy(x, forcing_data)[selected]
                    - adjustment[selected, 0]
                    - obs[selected, 0]
                ) / obs[selected, 0]

            attempts = [
                least_squares(
                    residual,
                    x,
                    bounds=([1, 0.7], [30, 2]),
                    max_nfev=160,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                )
                for x in ([5, 1], [20, 1.3], [29, 1.8])
            ]
            result = min([r for r in attempts if r.success], key=lambda r: r.cost)
            fits[f"{product}__{name}"] = dict(
                parameters=result.x.tolist(), training_years=years[keep].tolist()
            )
            predictions[name] = annual_awy(result.x, data) - adjustment[:, 0]
            if name.startswith("without"):
                for year in range(2019, 2023):
                    deletion.append(
                        dict(
                            landcover=product,
                            fit=name,
                            year=year,
                            predicted=predictions[name][year - 2013],
                            observed=obs[year - 2013, 0],
                        )
                    )
        for (phase, year), group in baseline.groupby(["phase", "year"]):
            if year == 2018:
                continue
            name = (
                "all_development"
                if phase == "reused_2023"
                else "early"
                if phase == "fixed_early" or year == 2019
                else f"before_{year}"
            )
            change = (
                predictions[name][year - 2013] - group[group.station == "贵得"].predicted.iloc[0]
            )
            updated = group.copy()
            updated["predicted"] += change
            updated["guide_change"] = change
            updated["model"] = "annual_original_bounds"
            updated["error"] = updated.predicted - updated.observed
            updated["relative_error_pct"] = 100 * updated.error / updated.observed
            rows.extend(updated.to_dict("records"))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "annual_reference_stations.csv", index=False)
    stats = []
    valid = frame[frame.year.between(2019, 2022) & frame.eligible]
    for (product, phase), group in valid.groupby(["landcover", "phase"]):
        for scope, part in (("all7", group), ("贵得", group[group.station == "贵得"])):
            stats.append(dict(landcover=product, phase=phase, scope=scope, **metrics(part)))
    pd.DataFrame(stats).to_csv(output / "annual_reference_metrics.csv", index=False)
    stability = (
        pd.DataFrame(deletion)
        .groupby(["landcover", "year"])
        .apply(
            lambda g: 100 * (g.predicted.max() - g.predicted.min()) / g.observed.iloc[0],
            include_groups=False,
        )
    )
    stability.rename("spread_pct_observed").to_csv(output / "annual_reference_stability.csv")
    write_json(
        output / "annual_reference_audit.json",
        dict(
            fits=fits,
            metrics=stats,
            mean_spread_pct=stability.groupby(level=0).mean().to_dict(),
            restricted_control_zero_spread_is_not_identifiability=True,
            monthly_frozen=True,
            original_acceptance=read_json(output / "summary.json")["accepted"],
            source=fingerprint(Path(__file__)),
        ),
    )
    print(pd.DataFrame(stats).to_string(index=False))
    print(stability.groupby(level=0).mean().to_dict())


if __name__ == "__main__":
    main()
