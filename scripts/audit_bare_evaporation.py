"""Fixed-parameter bare-surface evaporation sensitivity, not new model selection."""

from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
from calibrate_regional_water import block_data, metrics, open_inputs, source_paths
from natcap.invest.annual_water_yield.annual_water_yield import fractp_op
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.regional_calibration import ENDPOINTS
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def main():
    root = project_root()
    output = root / "project/diagnostics/bare_evaporation_v1"
    output.mkdir(exist_ok=False, parents=True)
    coefficients = [0.1, 0.15, 0.2, 0.3, 0.5]
    write_json(
        output / "protocol.json",
        {
            "purpose": "fixed-parameter sensitivity after spatial audit; not model selection",
            "landcover": "fine only; Copernicus 60 is vegetated sparse, not equivalent bare ground",
            "classes": [7, 8, 9, 10, 12],
            "coefficients": coefficients,
            "bounds_source": "illustrative range, not locally measured or literature calibrated",
            "other_parameters": "regional_water_v2 legacy unchanged; no refitting",
            "formula": "nonvegetated Y=max(P-Kc*ET0,0); Z, AWC and root depth have no effect",
            "years": [2019, 2020, 2021, 2022, 2023],
            "independent_validation": False,
            "sources": [
                fingerprint(Path(__file__)),
                fingerprint(root / "project/calibration/regional_water_v1/base_tables.json"),
            ],
        },
    )
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    tables = read_json(root / "project/calibration/regional_water_v1/base_tables.json")["fine"]
    baseline = pd.read_csv(root / "project/calibration/regional_water_v2/stations_fine.csv")
    baseline = baseline[baseline.variant == "legacy"]
    class_rows, station_rows = [], []
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    for year in range(2019, 2024):
        totals = defaultdict(lambda: np.zeros(4))
        increments = np.zeros((len(coefficients), 11))
        with ExitStack() as stack:
            datasets = open_inputs(stack, source_paths(root, index, "fine", year))
            for window in windows(datasets["zone"]):
                data, _ = block_data(datasets, window, tables[str(year)])
                selected = np.isin(data["code"], [7, 8, 9, 10, 12])
                bare = {k: v[selected] for k, v in data.items()}
                if not selected.any():
                    continue
                if (bare["veg"] != 0).any() or not np.allclose(bare["kc"], 0.15):
                    raise ValueError("Baseline bare class assumption changed")
                base_yield = np.maximum(bare["p"] - 0.15 * bare["et"], 0)
                area = abs(np.prod(np.array(datasets["zone"].GetGeoTransform())[[1, 5]])) / 1e6
                for ci, coefficient in enumerate(coefficients):
                    depth = np.maximum(bare["p"] - coefficient * bare["et"], 0)
                    fraction = fractp_op(
                        np.full(len(depth), coefficient),
                        bare["et"],
                        bare["p"],
                        bare["root_depth"],
                        bare["soil"],
                        bare["pawc"],
                        bare["veg"],
                        nodata,
                        30,
                    )
                    np.testing.assert_allclose(depth, (1 - fraction) * bare["p"], atol=1e-4)
                    increments[ci] += np.bincount(
                        bare["zone"], weights=(depth - base_yield) * area / 1e5, minlength=11
                    )
                for zone in np.unique(bare["zone"]):
                    for code in np.unique(bare["code"]):
                        mask = (bare["zone"] == zone) & (bare["code"] == code)
                        totals[int(zone) + 1, int(code)] += [
                            mask.sum() * area,
                            base_yield[mask].sum() * area / 1e5,
                            bare["p"][mask].sum() * area / 1e5,
                            bare["et"][mask].sum() * area / 1e5,
                        ]
        for (zone, code), amounts in sorted(totals.items()):
            class_rows.append(
                {
                    "year": year,
                    "zone": zone,
                    "code": code,
                    "area_km2": amounts[0],
                    "yield_1e8_m3": amounts[1],
                    "p_1e8_m3": amounts[2],
                    "et0_volume_1e8_m3": amounts[3],
                }
            )
        annual = baseline[baseline.year == year]
        for ci, coefficient in enumerate(coefficients):
            for si, (_, row) in enumerate(annual.iterrows()):
                change = increments[ci].cumsum()[ENDPOINTS[si]]
                station_rows.append(
                    {
                        "year": year,
                        "station": row.station,
                        "bare_kc": coefficient,
                        "predicted": row.predicted + change,
                        "observed": row.observed,
                        "yield_change_1e8_m3": change,
                    }
                )
        print(year, "bare sensitivity audited", flush=True)
    write_csv(output / "bare_by_zone_class.csv", class_rows)
    write_csv(output / "sensitivity_stations.csv", station_rows)
    scores = []
    rows = pd.DataFrame(station_rows)
    for (coefficient, phase), subset in rows.assign(
        phase=np.where(rows.year == 2023, "reused_2023", "development_fixed_parameters")
    ).groupby(["bare_kc", "phase"]):
        scores.append(
            {
                "bare_kc": coefficient,
                "phase": phase,
                **metrics(subset.predicted.to_numpy(), subset.observed.to_numpy()),
            }
        )
    write_csv(output / "sensitivity_metrics.csv", scores)
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == "__main__":
    main()
