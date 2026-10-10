"""Locate inherited runoff error and bound consumption effects without refitting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.long_record import STATIONS, YEARS, load_accounts, scores
from ecologyhydro.simulation import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/diagnostics/error_origin_v1")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / "project/diagnostics"):
        raise ValueError("Output must be in diagnostics")
    output.mkdir(parents=True, exist_ok=False)
    source = root / "project/calibration/long_record_v1"
    records, comparisons, accounts_rows = [], [], []
    files = []
    for product in ("fine", "copernicus"):
        path = source / f"stations_{product}.csv"
        files.append(fingerprint(path))
        frame = pd.read_csv(path)
        parameters = read_json(source / f"parameters_{product}.json")
        for account in ("legacy", "sector_2022"):
            target, consumption, storage, _, _ = load_accounts(root, account)
            cumulative_c = np.cumsum(consumption, axis=1)
            cumulative_s = np.cumsum(storage, axis=1)
            for phase in ("fixed_early", "rolling"):
                selected = frame[
                    (frame.model == "reach7")
                    & (frame.account == account)
                    & (frame.phase == phase)
                    & frame.validation
                ]
                variants = {name: [] for name in ("baseline", "known_guide", "known_lanzhou")}
                observations = []
                for year, group in selected.groupby("year"):
                    group = group.set_index("station").loc[STATIONS]
                    pred, obs = group.predicted.to_numpy(), group.observed.to_numpy()
                    error = pred - obs
                    variants["baseline"].append(pred)
                    # Replace only the inherited upstream error; downstream parameters unchanged.
                    for name, index in (("known_guide", 0), ("known_lanzhou", 1)):
                        conditional = pred.copy()
                        conditional[index + 1 :] -= error[index]
                        variants[name].append(conditional)
                    observations.append(obs)
                    yi = YEARS.index(int(year))
                    loss = parameters[group.fit.iloc[0]]["parameters"][-1]
                    for i, station in enumerate(STATIONS):
                        records.append(
                            dict(
                                landcover=product,
                                account=account,
                                phase=phase,
                                year=year,
                                station=station,
                                cumulative_error=error[i],
                                incremental_error=error[i] - (error[i - 1] if i else 0),
                                downstream_error_after_known_lanzhou=error[i] - error[1]
                                if i > 1
                                else None,
                                diagnostic_uses_same_year_observed_upstream=True,
                            )
                        )
                        accounts_rows.append(
                            dict(
                                landcover=product,
                                account=account,
                                phase=phase,
                                year=year,
                                station=station,
                                observed=obs[i],
                                predicted=pred[i],
                                cumulative_consumption=cumulative_c[yi, i],
                                cumulative_reservoir_change=cumulative_s[yi, i],
                                awy_yield=pred[i]
                                + cumulative_c[yi, i]
                                + cumulative_s[yi, i]
                                + (loss if i >= 2 else 0),
                                no_consumption_fixed_parameters=pred[i] + cumulative_c[yi, i],
                            )
                        )
                obs = np.array(observations)[:, 2:]
                for name, rows in variants.items():
                    pred = np.array(rows)[:, 2:]
                    comparisons.append(
                        dict(
                            landcover=product,
                            account=account,
                            phase=phase,
                            variant=name,
                            scope="five_stations_below_lanzhou",
                            **scores(pred, obs, np.ones_like(obs, bool)),
                        )
                    )
    pd.DataFrame(records).to_csv(output / "error_propagation.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "conditional_downstream_scores.csv", index=False)
    pd.DataFrame(accounts_rows).to_csv(output / "water_account_bounds.csv", index=False)
    write_json(
        output / "summary.json",
        {
            "sources": files + [fingerprint(Path(__file__))],
            "method": "Algebraic upstream error replacement; no parameter optimization",
            "warning": (
                "Observed upstream inflow is diagnostic information, "
                "not independent full-basin prediction"
            ),
            "consumption_bound": (
                "Removing all consumption at fixed parameters is an extreme sensitivity, "
                "not a corrected model"
            ),
            "production_changed": False,
        },
    )
    print(
        pd.DataFrame(comparisons)
        .query("account == 'legacy' and phase == 'fixed_early'")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
