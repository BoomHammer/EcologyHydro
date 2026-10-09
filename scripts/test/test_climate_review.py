"""Numerical and calendar checks for the deliberately approximate ET0 comparator."""

import numpy as np
import pytest
from netCDF4 import Dataset, date2num
from osgeo import gdal

from ecologyhydro.climate_review import UNITS, approximate_pm, native_diagnostics, read_months


def test_pm_hand_calculation_and_wind_height():
    # T=20 C, P=100 kPa, ea=1.5 kPa, Rn=10 MJ/m2/day, u2=2 m/s.
    # Delta~0.1447, gamma=0.0665 => ET0~3.64 mm/day.
    temperature = 293.15
    q = 0.622 * 1.5 / (100 - 0.378 * 1.5)
    longwave = 5.670374419e-8 * temperature**4
    shortwave = 10 / (0.0864 * 0.77)
    et, _ = approximate_pm(temperature, 100000, q, 2, shortwave, longwave, 2)
    assert float(et) == pytest.approx(3.64, abs=0.02)
    converted, _ = approximate_pm(
        temperature, 100000, q, 2 / (4.87 / np.log(672.58)), shortwave, longwave, 10
    )
    assert float(converted) == pytest.approx(float(et))


def test_negative_energy_and_supersaturation_are_explicit():
    et, flags = approximate_pm(293.15, 100000, 0.04, 2, 0, 0, 10)
    assert float(et) == 0
    assert flags["negative_et"]
    assert flags["supersaturated"]
    with pytest.raises(ValueError):
        approximate_pm(293.15, 100000, 0.01, 2, 200, 300, 5)


def test_leap_year_and_north_up_output(tmp_path):
    from datetime import datetime

    inputs = {
        "temp": 293.15,
        "pres": 100000,
        "shum": 0.009,
        "wind": 2,
        "srad": 200,
        "lrad": 350,
        "prec": 1 / 86400,
    }
    paths = {}
    for name, value in inputs.items():
        paths[name] = tmp_path / f"{name}.nc"
        with Dataset(paths[name], "w") as ds:
            for dim, size in (("time", 12), ("lat", 2), ("lon", 2)):
                ds.createDimension(dim, size)
            ds.createVariable("lat", "f8", ("lat",))[:] = [35, 36]
            ds.createVariable("lon", "f8", ("lon",))[:] = [100, 101]
            t = ds.createVariable("time", "f8", ("time",))
            t.units = "days since 2020-01-01"
            t[:] = date2num([datetime(2020, m, 15) for m in range(1, 13)], t.units)
            var = ds.createVariable(name, "f8", ("time", "lat", "lon"))
            var.units = UNITS[name]
            var[:] = value
            if name == "prec":
                var[:, 1, :] = value * 2
    output = tmp_path / "out"
    output.mkdir()
    native_diagnostics(paths, 2020, output)
    with gdal.Open(str(output / "prec_recomputed.tif")) as ds:
        assert ds.GetGeoTransform()[5] == -1
        np.testing.assert_allclose(ds.ReadAsArray(), [[732, 732], [366, 366]])
    with Dataset(paths["prec"], "a") as ds:
        ds["prec"].units = "mm"
    with pytest.raises(ValueError, match="metadata"):
        read_months(paths["prec"], "prec", 2020)
