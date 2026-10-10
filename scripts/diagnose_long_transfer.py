"""Summarize temporal parameter drift and climate differences without refitting."""

import argparse

import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.config import project_root
from ecologyhydro.long_record import YEARS, load_accounts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/long_record_v1")
    args = parser.parse_args()
    root, output = project_root(), project_root() / args.output
    parameters = pd.read_csv(output / "parameter_trajectories.csv")
    past = parameters[(parameters.account == "legacy") & (parameters.window != "all_development")]
    ranges = past.groupby(["landcover", "model", "parameter"]).agg(
        minimum=("value", "min"),
        maximum=("value", "max"),
        windows=("value", "count"),
        bound_windows=("active_bound", lambda x: (x != 0).sum()),
    )
    ranges["range"] = ranges.maximum - ranges.minimum
    ranges.to_csv(output / "past_parameter_ranges.csv")
    identifiability = []
    for product in ("fine", "copernicus"):
        for key, fit in read_json(output / f"parameters_{product}.json").items():
            model, account, window = key.split("__")
            singular = np.asarray(fit["data_jacobian_singular_values"])
            identifiability.append(
                {
                    "landcover": product,
                    "model": model,
                    "account": account,
                    "window": window,
                    "data_only_scaled_jacobian_condition": singular.max() / singular.min(),
                    "smallest_singular_value": singular.min(),
                    "converged_starts": fit["converged_starts"],
                }
            )
    pd.DataFrame(identifiability).to_csv(output / "identifiability.csv", index=False)
    rain = pd.read_csv(output / "rainfall_comparison.csv")
    rain.groupby(["region_id", "region"]).agg(
        available_years=("proxy_difference_pct", "count"),
        mean_difference_pct=("proxy_difference_pct", "mean"),
        minimum_difference_pct=("proxy_difference_pct", "min"),
        maximum_difference_pct=("proxy_difference_pct", "max"),
    ).to_csv(output / "rainfall_proxy_summary.csv")
    target = load_accounts(root)[0]
    climate_rows = []
    for region_id, group in rain.groupby("region_id"):
        early = group[group.year.between(2013, 2017)]
        later = group[group.year.between(2019, 2022)]
        row = {"region_id": region_id, "region": group.region.iloc[0]}
        for label in ("precipitation_mm", "pet_mm", "statistical_precipitation_mm"):
            row[f"early_{label}"] = early[label].mean()
            row[f"later_{label}"] = later[label].mean()
            row[f"change_{label}_pct"] = (later[label].mean() / early[label].mean() - 1) * 100
        column = int(region_id) - 1
        early_q = target[np.isin(YEARS, range(2013, 2018)), column].mean()
        later_q = target[np.isin(YEARS, range(2019, 2023)), column].mean()
        row.update(
            early_cumulative_observed=early_q,
            later_cumulative_observed=later_q,
            change_cumulative_observed_pct=(later_q / early_q - 1) * 100,
        )
        climate_rows.append(row)
    pd.DataFrame(climate_rows).to_csv(output / "early_late_climate_runoff.csv", index=False)
    print("Past-only parameter ranges, identifiability and climate comparisons written")


if __name__ == "__main__":
    main()
