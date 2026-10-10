"""Fixed-parameter rainfall-proxy sensitivity, not calibration or bias correction."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from calibrate_regional_water import sample_year, sampling_positions
from long_inputs import paths_for
from osgeo import gdal

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.regional_transfer import evaluate
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/diagnostics/headwater_balance_v1")
    args = parser.parse_args()
    root, output = project_root(), project_root() / args.output
    if (output / "precipitation_sensitivity.json").exists():
        raise ValueError("Completed rainfall diagnostic exists")
    configure_threads(load_config().resources)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    source = root / "project/calibration/long_record_v1"
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    rainfall = pd.read_csv(source / "rainfall_comparison.csv")
    annual = pd.read_csv(output / "annual_cumulative_balance.csv")
    write_json(
        output / "precipitation_protocol.json",
        {
            "years": [2019, 2020, 2021, 2022],
            "parameters": "Unchanged 2013-2017 reach7 fits",
            "scenario": "Match statistical rainfall depth only in the first two gauge proxies",
            "scale": "No use of observed runoff to select rainfall factors",
            "approximation": "Prior deterministic quadrature for fixed-parameter yield difference",
            "warning": (
                "Different statistical and gauge boundaries; not verified precipitation correction"
            ),
            "posthoc_diagnostic": True,
            "production_changed": False,
        },
    )
    records, sources = [], [fingerprint(source / "rainfall_comparison.csv")]
    for product in ("fine", "copernicus"):
        params_path, table_path = (
            source / f"parameters_{product}.json",
            source / f"tables_{product}.json",
        )
        parameters = read_json(params_path)["reach7__legacy__early"]["parameters"]
        tables = read_json(table_path)
        stations = pd.read_csv(source / f"stations_{product}.csv")
        positions, weights, _ = sampling_positions(paths_for(root, index, product, 2013))
        sources += [fingerprint(params_path), fingerprint(table_path)]
        for year in range(2019, 2023):
            paths = paths_for(root, index, product, year)
            data = sample_year(paths, tables[str(year)], positions, weights)
            baseline = evaluate(parameters, data, "reach7", 1)[0, :2]
            rows = stations[
                (stations.model == "reach7")
                & (stations.account == "legacy")
                & (stations.phase == "fixed_early")
                & (stations.year == year)
            ]
            rows = rows.set_index("station").loc[["贵得", "兰州"]]
            np.testing.assert_allclose(
                baseline - rows.known_adjustment.to_numpy(), rows.sampled_prediction, atol=1e-8
            )
            changed = {key: value.copy() for key, value in data.items()}
            factors = []
            for region in (1, 2):
                r = rainfall[(rainfall.year == year) & (rainfall.region_id == region)].iloc[0]
                factor = r.statistical_precipitation_mm / r.precipitation_mm
                factors.append(factor)
                mask = data["zone"] <= 1 if region == 1 else data["zone"] == 2
                changed["p"][mask] *= factor
            revised = evaluate(parameters, changed, "reach7", 1)[0, :2]
            for si, station in enumerate(("贵得", "兰州")):
                row = rows.loc[station]
                process = annual[(annual.year == year) & (annual.station == station)].iloc[0]
                amount = revised[si] - baseline[si]
                storage_effect = -process.delta_soil - process.delta_snow
                records.append(
                    dict(
                        landcover=product,
                        year=year,
                        station=station,
                        local_rainfall_factor=factors[si],
                        observed=row.observed,
                        baseline_predicted=row.predicted,
                        rainfall_yield_change=amount,
                        rainfall_only_predicted=row.predicted + amount,
                        soil_snow_effect=storage_effect,
                        rainfall_and_external_storage=row.predicted + amount + storage_effect,
                    )
                )
            print(product, year, "fixed rainfall sensitivity complete", flush=True)
    pd.DataFrame(records).to_csv(output / "precipitation_sensitivity.csv", index=False)
    write_json(
        output / "precipitation_sensitivity.json",
        {
            "sources": sources
            + [fingerprint(Path(__file__)), fingerprint(output / "precipitation_protocol.json")],
            "not_independent_validation": True,
            "no_new_parameters": True,
            "not_selected_for_production": True,
        },
    )
    print(pd.DataFrame(records).query("landcover=='fine'").to_string(index=False))


if __name__ == "__main__":
    main()
