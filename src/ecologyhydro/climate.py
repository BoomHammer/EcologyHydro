"""Native-grid climate subsets and calendar-aware annual depth aggregation."""

import calendar
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date
from osgeo import gdal

from ecologyhydro.spatial import NODATA, write_raster


def axis_window(values, lower, upper):
    values = np.asarray(values, dtype=float)
    delta = np.diff(values)
    if not (np.all(delta > 0) or np.all(delta < 0)) or not np.allclose(
        delta, delta[0], rtol=1e-3, atol=1e-6
    ):
        raise ValueError("Expected regular monotonic climate coordinates")
    # Retain intersecting native cells, plus one interpolation support cell.
    selected = np.flatnonzero(
        (values + abs(delta[0]) / 2 >= lower) & (values - abs(delta[0]) / 2 <= upper)
    )
    if not selected.size:
        raise ValueError("Climate and reference do not overlap")
    return slice(max(0, selected[0] - 1), min(len(values), selected[-1] + 2))


def crop_netcdf(source, target, extent):
    """Copy all dates/variables, packing and masks intact, in monthly slices."""
    with Dataset(source) as original, Dataset(target, "w", format="NETCDF4") as output:
        selections = {
            "lon": axis_window(original["lon"][:], extent[0], extent[2]),
            "lat": axis_window(original["lat"][:], extent[1], extent[3]),
        }
        output.setncatts({name: original.getncattr(name) for name in original.ncattrs()})
        output.setncattr("ecologyhydro_subset", "Copernicus bounds + one native interpolation cell")
        for name, dimension in original.dimensions.items():
            size = (
                len(range(*selections[name].indices(len(dimension))))
                if name in selections
                else len(dimension)
            )
            output.createDimension(name, size)
        for name, variable in original.variables.items():
            attrs = {key: variable.getncattr(key) for key in variable.ncattrs()}
            fill = attrs.pop("_FillValue", None)
            copied = output.createVariable(
                name, variable.dtype, variable.dimensions, fill_value=fill, zlib=True, complevel=2
            )
            copied.setncatts(attrs)
            variable.set_auto_maskandscale(False)
            copied.set_auto_maskandscale(False)
            slices = [selections.get(dim, slice(None)) for dim in variable.dimensions]
            if "time" in variable.dimensions:
                axis = variable.dimensions.index("time")
                for t in range(len(original.dimensions["time"])):
                    src = list(slices)
                    dst = [slice(None)] * variable.ndim
                    src[axis] = dst[axis] = t
                    copied[tuple(dst)] = variable[tuple(src)]
            else:
                copied[:] = variable[tuple(slices)]
        original_cells = len(original.dimensions["lat"]) * len(original.dimensions["lon"])
        subset_cells = len(output.dimensions["lat"]) * len(output.dimensions["lon"])
    return {
        "source_cells_per_month": original_cells,
        "subset_cells_per_month": subset_cells,
        "fraction_retained": subset_cells / original_cells,
        "all_dates_preserved": True,
    }


def annual_precipitation(source: Path, variable: str, year: int, target: Path):
    with Dataset(source) as dataset:
        data = dataset[variable]
        if data.dimensions != ("time", "lat", "lon") or data.units != "kg m-2 s-1":
            raise ValueError(
                f"Unsupported precipitation layout/units: {data.dimensions}/{data.units}"
            )
        t = dataset["time"]
        calendar_name = getattr(t, "calendar", "standard")
        if calendar_name not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError(f"CMFD calendar not supported: {calendar_name}")
        dates = num2date(t[:], t.units, calendar=calendar_name)
        selected = [(i, date.month) for i, date in enumerate(dates) if date.year == year]
        if sorted(month for _, month in selected) != list(range(1, 13)):
            raise ValueError(f"Missing or duplicate months: {source}, {year}")
        total = np.zeros(data.shape[1:], dtype=np.float64)
        valid = np.ones(total.shape, dtype=bool)
        for index, month in selected:
            values = np.ma.asarray(data[index], dtype=np.float64)
            raw = values.filled(np.nan)
            usable = ~np.ma.getmaskarray(values) & np.isfinite(raw) & (raw >= 0)
            valid &= usable
            total += np.where(usable, raw, 0) * calendar.monthrange(year, month)[1] * 86400
        lon, lat = np.asarray(dataset["lon"][:]), np.asarray(dataset["lat"][:])
        dx, dy = float(np.mean(np.diff(lon))), float(np.mean(np.diff(lat)))
        if dx < 0:
            lon, total, valid = lon[::-1], total[:, ::-1], valid[:, ::-1]
        if dy > 0:
            lat, total, valid = lat[::-1], total[::-1], valid[::-1]
        transform = (
            float(lon[0]) - abs(dx) / 2,
            abs(dx),
            0,
            float(lat[0]) + abs(dy) / 2,
            0,
            -abs(dy),
        )
        write_raster(target, np.where(valid, total, NODATA), transform, "EPSG:4326")
    return {
        "units": "mm/year",
        "year": year,
        "calendar": calendar_name,
        "method": "sum(monthly mean kg m-2 s-1 * calendar month seconds)",
        "invalid_cells": int((~valid).sum()),
    }


def annual_pet(sources, target):
    if len(sources) != 12:
        raise ValueError("PET requires exactly twelve monthly rasters")
    total = valid = transform = projection = None
    for source in sources:
        with gdal.Open(str(source)) as dataset:
            if total is None:
                transform, projection = dataset.GetGeoTransform(), dataset.GetProjection()
                total = np.zeros((dataset.RasterYSize, dataset.RasterXSize), dtype=np.float64)
                valid = np.ones(total.shape, dtype=bool)
            if (
                dataset.GetGeoTransform() != transform
                or dataset.GetProjection() != projection
                or (dataset.RasterYSize, dataset.RasterXSize) != total.shape
            ):
                raise ValueError(f"Monthly PET grids differ: {source}")
            values = dataset.ReadAsArray()
            usable = dataset.GetRasterBand(1).GetMaskBand().ReadAsArray() != 0
            usable &= np.isfinite(values) & (values >= 0)
            valid &= usable
            total += np.where(usable, values, 0) * 0.1
    write_raster(target, np.where(valid, total, NODATA), transform, projection)
    return {
        "units": "mm/year",
        "method": "sum(raw monthly PET * 0.1); no day multiplier",
        "zero_cells": int((valid & (total == 0)).sum()),
        "invalid_cells": int((~valid).sum()),
    }
