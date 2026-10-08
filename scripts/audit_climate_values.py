"""Check monthly climate values using a provisional basin pixel-center mask."""

import csv
import json

import numpy as np
from netCDF4 import Dataset, num2date
from osgeo import gdal, ogr, osr

from ecologyhydro.config import load_config, project_root


def basin_mask(root, lon, lat):
    if np.ma.getmaskarray(lon).any() or np.ma.getmaskarray(lat).any():
        raise ValueError("Missing longitude/latitude coordinates")
    lon, lat = np.asarray(lon), np.asarray(lat)
    boundary = ogr.Open(str(root / "data/Hydrology/黄河流域.shp"))
    layer = boundary.GetLayer(0)
    source = layer.GetSpatialRef().Clone()
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    for crs in (source, target):
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    options = osr.CoordinateTransformationOptions()
    options.SetBallparkAllowed(True)
    transform = osr.CreateCoordinateTransformation(source, target, options)
    vector = ogr.GetDriverByName("Memory").CreateDataSource("")
    polygons = vector.CreateLayer("basin", srs=target, geom_type=ogr.wkbMultiPolygon)
    for feature in layer:
        geometry = feature.GetGeometryRef().Clone()
        geometry.Transform(transform)
        output = ogr.Feature(polygons.GetLayerDefn())
        output.SetGeometry(geometry)
        polygons.CreateFeature(output)
    dx = float(np.median(np.diff(lon)))
    dy = float(np.median(np.diff(lat)))
    mask = gdal.GetDriverByName("MEM").Create("", len(lon), len(lat), 1, gdal.GDT_Byte)
    mask.SetGeoTransform((float(lon[0] - dx / 2), dx, 0, float(lat[-1] + dy / 2), 0, -dy))
    mask.SetProjection(target.ExportToWkt())
    gdal.RasterizeLayer(mask, [1], polygons, burn_values=[1])
    return mask.ReadAsArray()[::-1] == 1


def main():
    gdal.UseExceptions()
    ogr.UseExceptions()
    config = load_config()
    root = project_root()
    records = []
    for path in sorted(config.paths.climate.rglob("*.nc")):
        name = path.name.split("_")[0]
        with Dataset(path) as dataset:
            inside = basin_mask(root, dataset["lon"][:], dataset["lat"][:])
            if not inside.any():
                raise ValueError("No basin pixels in climate grid")
            time = dataset["time"]
            dates = num2date(time[:], time.units, calendar=time.calendar)
            variable = dataset[name]
            for index, date in enumerate(dates):
                if not 2019 <= date.year <= 2023:
                    continue
                data = np.ma.asarray(variable[index, :, :])[inside]
                missing = np.ma.getmaskarray(data) | ~np.isfinite(data.data)
                values = data.data[~missing]
                invalid = values < 0
                if name in ("pres", "temp"):
                    invalid = values <= 0
                elif name == "rhum":
                    invalid |= values > 100
                elif name == "shum":
                    invalid |= values > 1
                records.append(
                    {
                        "variable": name,
                        "year": date.year,
                        "month": date.month,
                        "units": variable.units,
                        "basin_pixels": int(inside.sum()),
                        "missing": int(missing.sum()),
                        "invalid_basic_range": int(invalid.sum()),
                        "min": float(values.min()) if values.size else None,
                        "max": float(values.max()) if values.size else None,
                    }
                )
    observations = []
    with (config.paths.hydrology / "实测年径流量2018-2023.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            for year in range(2019, 2024):
                raw = row[str(year)].strip()
                if not raw:
                    status = "missing"
                else:
                    try:
                        value = float(raw)
                        status = "valid" if np.isfinite(value) and value > 0 else "anomaly"
                    except ValueError:
                        status = "anomaly"
                observations.append({"station": row["测站"], "year": year, "status": status})
    report = {
        "scope": "Provisional basin pixel-center mask; datum approximate, inner basin not removed",
        "checks": "Missing/nonfinite and basic sign/range only, not a full plausibility assessment",
        "monthly": records,
        "observations": observations,
    }
    output = config.paths.cache / "inventory/climate_values.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {output}")
    for name in sorted({record["variable"] for record in records}):
        rows = [record for record in records if record["variable"] == name]
        print(
            name,
            "months",
            len(rows),
            "pixels",
            rows[0]["basin_pixels"],
            "missing",
            sum(row["missing"] for row in rows),
            "invalid",
            sum(row["invalid_basic_range"] for row in rows),
            "range",
            min(row["min"] for row in rows if row["min"] is not None),
            max(row["max"] for row in rows if row["max"] is not None),
        )
    print(
        "Observations:",
        {
            status: sum(row["status"] == status for row in observations)
            for status in ("valid", "missing", "anomaly")
        },
    )


if __name__ == "__main__":
    main()
