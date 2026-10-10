"""Verify past-only fit membership, missing-data decisions and actual reported scores."""

import argparse
from pathlib import Path

import numpy as np

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.long_record import STATIONS, YEARS, load_accounts, number, scores
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/long_record_v1")
    args = parser.parse_args()
    root, output = project_root(), project_root() / args.output
    protocol, summary = read_json(output / "protocol.json"), read_json(output / "summary.json")
    for source in read_json(output / "manifest.json")["sources"]:
        if fingerprint(Path(source["path"]))["sha256"] != source["sha256"]:
            raise ValueError(f"Changed source: {source['path']}")
    climate_checks = 0
    climate = read_json(root / "project/repairs/long_climate_v1/summary.json")
    for record in climate["records"]:
        for key in ("raw", "aligned"):
            source = record[key]
            if fingerprint(Path(source["path"]))["sha256"] != source["sha256"]:
                raise ValueError(f"Changed climate input: {source['path']}")
            climate_checks += 1
    count, discrepancy = 0, 0.0
    for product, cases in summary["products"].items():
        fits = read_json(output / f"parameters_{product}.json")
        rows = read_csv(output / f"stations_{product}.csv")
        if any(int(r["year"]) == 2023 for r in rows) or protocol["evaluated_2023"]:
            raise ValueError("Unexpected reused-test-year evaluation")
        account_data = {v: load_accounts(root, v) for v in ("legacy", "sector_2022")}
        for row in rows:
            year, station = int(row["year"]), row["station"]
            target, _, _, adjustment, eligible = account_data[row["account"]]
            yi, si = YEARS.index(year), STATIONS.index(station)
            if (row["eligible"] == "True") != eligible[yi, si]:
                raise ValueError("Changed missing-data eligibility")
            np.testing.assert_allclose(number(row["observed"]), target[yi, si])
            np.testing.assert_allclose(number(row["known_adjustment"]), adjustment[yi, si])
            if not np.isfinite(adjustment[yi, si]) and np.isfinite(number(row["predicted"])):
                raise ValueError("Missing cumulative account was silently filled")
            if row["validation"] == "True":
                trained = [YEARS[i] for i in fits[row["fit"]]["training_indices"]]
                if max(trained) >= year or 2018 in trained:
                    raise ValueError("Temporal leakage or accountless year in training")
                if row["training_years"] != ";".join(map(str, trained)):
                    raise ValueError("Incorrect displayed training years")
                if row["official_full_pixel"] != "True":
                    raise ValueError("Temporal validation lacks official pixel verification")
                discrepancy = max(
                    discrepancy,
                    abs(float(row["predicted"]) - float(row["sampled_prediction"]))
                    / target[yi, si],
                )
                count += 1
        for case_name, case in cases.items():
            model, version, phase = case_name.split("__")
            chosen = [
                r
                for r in rows
                if r["model"] == model
                and r["account"] == version
                and r["phase"] == phase
                and r["validation"] == "True"
            ]
            expected_keys = {
                (str(y), s) for y in protocol["forward_validation_years"] for s in STATIONS
            }
            if len(chosen) != 28 or {(r["year"], r["station"]) for r in chosen} != expected_keys:
                raise ValueError("Missing or repeated validation station-years")
            groups = [(chosen, case)]
            groups += [
                ([r for r in chosen if r["year"] == year], metrics)
                for year, metrics in case["years"].items()
            ]
            groups += [
                ([r for r in chosen if r["station"] != "利津"], case["common_six_stations"]),
                ([r for r in chosen if r["station"] == "利津"], case["lower_reach_conditional"]),
            ]
            for group, reported in groups:
                expected = scores(
                    np.array([number(r["predicted"]) for r in group]),
                    np.array([number(r["observed"]) for r in group]),
                    np.array([r["eligible"] == "True" for r in group]),
                )
                for key, value in expected.items():
                    np.testing.assert_allclose(reported[key], value)
            np.testing.assert_allclose(
                case["worst_year_relative_rmse_pct"],
                max(m["relative_rmse_pct"] for m in case["years"].values()),
            )
        # An alternate 2022 account cannot affect any fit whose training stops in 2021.
        for model in protocol["models"]:
            for window in ("early", "before_2020", "before_2021", "before_2022"):
                np.testing.assert_array_equal(
                    fits[f"{model}__legacy__{window}"]["parameters"],
                    fits[f"{model}__sector_2022__{window}"]["parameters"],
                )
    if count != 672 or discrepancy > 0.01:
        raise ValueError("Incomplete validation or excessive quadrature discrepancy")
    write_json(
        output / "verification.json",
        {
            "passed": True,
            "temporal_validation_rows_checked": count,
            "new_climate_input_fingerprints_checked": climate_checks,
            "max_official_quadrature_fraction_observed": discrepancy,
            "no_future_training_or_missing_fill": True,
            "scores_recomputed": True,
            "alternate_future_account_does_not_change_past_fit": True,
        },
    )
    print("Long-record verification passed", count, "rows", flush=True)


if __name__ == "__main__":
    main()
