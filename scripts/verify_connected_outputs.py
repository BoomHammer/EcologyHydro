"""Check actual output area, rainfall means, and station accounting independently."""

from pathlib import Path

import numpy as np
import pandas as pd
from calibrate_regional_water import source_paths
from osgeo import gdal

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.reference_checks import REGIONS
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def main():
    root = project_root()
    gdal.UseExceptions()
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    folder = root / "project/diagnostics/gonghe_connectivity_v1"
    removed = 0
    with (
        gdal.Open(str(Path(index["routing"]["partitions"]) / "zones.tif")) as old,
        gdal.Open(str(folder / "zones.tif")) as new,
        gdal.Open(str(folder / "basin_mask.tif")) as mask,
    ):
        for window in windows(old):
            before, after, inside = (
                old.ReadAsArray(*window),
                new.ReadAsArray(*window),
                mask.ReadAsArray(*window) > 0,
            )
            np.testing.assert_array_equal(after[~inside], before[~inside])
            np.testing.assert_array_equal(after[inside], 0)
            removed += int(((before > 0) & inside).sum())
    rain_folder = root / "project/diagnostics/reference_precipitation_v1"
    factors = pd.read_csv(rain_folder / "factors.csv")
    maximum = 0.0
    for year in range(2019, 2024):
        with (
            gdal.Open(str(folder / "zones.tif")) as zones,
            gdal.Open(source_paths(root, index, "fine", year)["p"]) as original,
            gdal.Open(str(rain_folder / f"precipitation_{year}.tif")) as adjusted,
        ):
            sums, counts = np.zeros(11), np.zeros(11)
            for window in windows(zones):
                z, p = zones.ReadAsArray(*window), adjusted.ReadAsArray(*window)
                old_p = original.ReadAsArray(*window)
                unchanged = (z == 0) | (z > 9)
                np.testing.assert_array_equal(p[unchanged], old_p[unchanged])
                valid = z > 0
                sums += np.bincount(z[valid].astype(int) - 1, weights=p[valid], minlength=11)
                counts += np.bincount(z[valid].astype(int) - 1, minlength=11)
            for name, start, stop in REGIONS[:6]:
                mean = sums[start:stop].sum() / counts[start:stop].sum()
                expected = factors[
                    (factors.year == year) & (factors.region == name)
                ].reference_mean_mm.item()
                maximum = max(maximum, abs(mean - expected))
                np.testing.assert_allclose(mean, expected, rtol=1e-6)
    for experiment in ("connected_water_v1", "reference_rainfall_v1"):
        for product in ("fine", "copernicus"):
            rows = pd.read_csv(root / f"project/calibration/{experiment}/stations_{product}.csv")
            np.testing.assert_allclose(
                rows.predicted,
                rows.awy_yield - rows.known_adjustment_cumulative - rows.net_loss_cumulative,
            )
            assert np.isfinite(rows.select_dtypes("number")).all().all()
    old_manifest = read_json(root / "project/calibration/regional_water_v1/manifest.json")
    inputs = [item for item in old_manifest["sources"] if Path(item["path"]).suffix != ".py"]
    for item in inputs:
        assert fingerprint(Path(item["path"]))["sha256"] == item["sha256"]
    result = {
        "domain_removed_pixels": removed,
        "rainfall_max_mean_error_mm": maximum,
        "outside_domain_and_downstream_precipitation_unchanged": True,
        "station_accounting_checked": True,
        "original_inputs_unchanged": len(inputs),
    }
    write_json(root / "project/calibration/connected_water_v1/verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
