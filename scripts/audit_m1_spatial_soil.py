"""Read-only M1 diagnostics; boundary transformation is provisional."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

from ecologyhydro.config import project_root


def inspect_spatial(root: Path) -> dict:
    raster = gdal.Open(str(root / "data/LCLU/vegetation_model_20261007_154221plusIce.tif"))
    boundary = ogr.Open(str(root / "data/Hydrology/黄河流域.shp"))
    layer = boundary.GetLayer(0)
    source = layer.GetSpatialRef().Clone()
    target = raster.GetSpatialRef().Clone()
    for crs in (source, target):
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    options = osr.CoordinateTransformationOptions()
    options.SetBallparkAllowed(True)
    transform = osr.CreateCoordinateTransformation(source, target, options)
    vector = ogr.GetDriverByName("Memory").CreateDataSource("")
    polygons = vector.CreateLayer("boundary", srs=target, geom_type=ogr.wkbMultiPolygon)
    for feature in layer:
        geometry = feature.GetGeometryRef().Clone()
        geometry.Transform(transform)
        output = ogr.Feature(polygons.GetLayerDefn())
        output.SetGeometry(geometry)
        polygons.CreateFeature(output)
    mask = gdal.GetDriverByName("MEM").Create(
        "", raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_Byte
    )
    mask.SetGeoTransform(raster.GetGeoTransform())
    mask.SetProjection(raster.GetProjection())
    gdal.RasterizeLayer(mask, [1], polygons, burn_values=[1])
    counts = Counter()
    zero_examples = []
    gt = raster.GetGeoTransform()
    for row in range(0, raster.RasterYSize, 256):
        rows = min(256, raster.RasterYSize - row)
        data = raster.ReadAsArray(0, row, raster.RasterXSize, rows)
        inside = mask.ReadAsArray(0, row, raster.RasterXSize, rows) == 1
        values, numbers = np.unique(data[inside], return_counts=True)
        counts.update(dict(zip(values.tolist(), numbers.tolist(), strict=True)))
        if len(zero_examples) < 5:
            ys, xs = np.nonzero(inside & (data == 0))
            for y, x in zip(ys[:1], xs[:1], strict=True):
                zero_examples.append([gt[0] + (x + 0.5) * gt[1], gt[3] + (row + y + 0.5) * gt[5]])
    return {
        "scope": "Provisional boundary overlap; pixel-center counts, not area fractions",
        "crs_limitation": "Boundary datum unspecified; ballpark transformation allowed",
        "counts": dict(sorted(counts.items())),
        "zero_pixel_fraction": counts[0] / counts.total(),
        "zero_example_lon_lat": zero_examples,
    }


def inspect_soil(root: Path, output_dir: Path) -> dict:
    sys.path.insert(0, str(root / ".tools/mdb-reader"))
    from access_parser import AccessParser

    path = root / "data/Soil/HWSD2.mdb"
    with path.open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    database = AccessParser(str(path))
    exported = {}
    for name in (
        "HWSD2_SMU_METADATA",
        "HWSD2_LAYERS_METADATA",
        "D_ROOT_DEPTH",
        "D_AWC",
        "HWSD2_SMU",
    ):
        table = database.parse_table(name)
        lengths = {len(values) for values in table.values()}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent column lengths: {name}")
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(table)
            writer.writerows(zip(*table.values(), strict=True))
        exported[name] = lengths.pop()
    smu = table
    raster = gdal.Open(str(root / "data/Soil/hwsd.bil"))
    gt = raster.GetGeoTransform()
    # Inspect only a geographic rectangle around the study area, not all global soil cells.
    x = int((95 - gt[0]) / gt[1])
    y = int((43 - gt[3]) / gt[5])
    width, height = round(25 / gt[1]), round(-12 / gt[5])
    codes, numbers = np.unique(raster.ReadAsArray(x, y, width, height), return_counts=True)
    ids = set(smu["HWSD2_SMU_ID"])
    old_ids = set(smu["HWSD1_SMU_ID"])
    mapping = dict(zip(codes.tolist(), numbers.tolist(), strict=True))
    missing = {code: number for code, number in mapping.items() if code not in ids}
    changed = [
        [old, new]
        for old, new in zip(smu["HWSD1_SMU_ID"], smu["HWSD2_SMU_ID"], strict=True)
        if old in mapping and old != new
    ]
    with path.open("rb") as stream:
        after = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != after:
        raise RuntimeError("Source database checksum changed")
    return {
        "reader": "access-parser 0.0.6; construct 2.10.70; tabulate 0.10.0",
        "tables": list(database.catalog),
        "exports_rows": exported,
        "mdb_sha256": before,
        "mdb_unchanged": True,
        "soil_window": "95-120E, 31-43N based on world file; datum not yet verified",
        "soil_unique_codes": len(mapping),
        "unmatched_hwsd2_codes_counts": missing,
        "unmatched_hwsd1_codes": sorted(set(mapping) - old_ids),
        "old_to_new_differing_pairs_in_window": changed,
        "limitation": "Numeric key matches alone cannot establish raster version",
    }


def main():
    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
    root = project_root()
    directory = root / "project/cache/inventory/m1_spatial_soil"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"spatial": inspect_spatial(root), "soil": inspect_soil(root, directory)}
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
