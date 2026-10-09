"""Check real repaired capacity conservation, closed-lake capture and station areas."""

import json
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

from ecologyhydro.aggregation import write_csv
from ecologyhydro.config import project_root
from ecologyhydro.endorheic import check_exclusion
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    gdal.UseExceptions()
    root = project_root()
    old = read_json(root / "project/cache/m2/runs/before_soil_routing_repair_20261009.json")
    new = read_json(root / "project/cache/m2/latest.json")
    output = root / "project/repairs/soil_routing"
    output.mkdir(parents=True, exist_ok=True)
    old_zones = Path(old["routing"]["partitions"]) / "zones.tif"
    new_zones = Path(new["routing"]["partitions"]) / "zones.tif"
    lake = Path(new["routing"]["closed_lakes"]) / "mask.tif"
    matrix = np.zeros((12, 12), dtype=np.int64)
    with gdal.Open(str(old_zones)) as before, gdal.Open(str(new_zones)) as after:
        for window in windows(before):
            a, b = before.ReadAsArray(*window), after.ReadAsArray(*window)
            matrix += np.bincount((a.astype(int) * 12 + b).ravel(), minlength=144).reshape(12, 12)
        pixel_area = abs(before.GetGeoTransform()[1] * before.GetGeoTransform()[5]) / 1e6
        vector_path = root / "project/diagnostics/deep_review/lake_capture/capture.gpkg"
        with ogr.Open(str(vector_path)) as vector:
            mask = gdal.GetDriverByName("MEM").Create(
                "", before.RasterXSize, before.RasterYSize, 1, gdal.GDT_Byte
            )
            mask.SetGeoTransform(before.GetGeoTransform())
            mask.SetProjection(before.GetProjection())
            gdal.RasterizeLayer(mask, [1], vector.GetLayer(), burn_values=[1])
            captured = remaining = 0
            for window in windows(before):
                inside = mask.ReadAsArray(*window) == 1
                captured += int(inside.sum())
                remaining += int(np.count_nonzero(inside & (after.ReadAsArray(*window) > 0)))
    areas = []
    outlets = read_json(Path(new["routing"]["outlets"]) / "outlets.json")
    for i, outlet in enumerate(outlets, 1):
        a = float(matrix[1 : i + 1, :].sum() * pixel_area)
        b = float(matrix[:, 1 : i + 1].sum() * pixel_area)
        areas.append(
            {
                "station": outlet["station"],
                "before_km2": a,
                "after_km2": b,
                "reference_km2": outlet["reference_km2"],
                "change_km2": b - a,
                "relative_to_reference_percent": (b / outlet["reference_km2"] - 1) * 100,
                "new_outlet_shift_m": outlet["outlet_shift_m"],
            }
        )
    write_csv(output / "area_comparison.csv", areas)
    soil_dir = Path(new["native"]["pawc"]).parent
    with (
        gdal.Open(new["native"]["pawc"]) as p,
        gdal.Open(new["native"]["root_depth"]) as d,
        gdal.Open(str(soil_dir / "capacity_mm.tif")) as c,
    ):
        error = 0.0
        for window in windows(p):
            valid = p.GetRasterBand(1).GetMaskBand().ReadAsArray(*window) > 0
            expected = p.ReadAsArray(*window).astype(float) * d.ReadAsArray(*window)
            if valid.any():
                error = max(
                    error, float(np.max(np.abs(expected[valid] - c.ReadAsArray(*window)[valid])))
                )
    if error > 0.0001:
        raise ValueError("Root-zone capacity conservation failed")
    report = {
        "closed_lake_check": check_exclusion(new_zones, lake),
        "old_d8_lake_capture_area_km2": captured * pixel_area,
        "old_d8_lake_capture_still_in_mainstem_km2": remaining * pixel_area,
        "old_mainstem_area_removed_km2": float(matrix[1:, 0].sum() * pixel_area),
        "new_mainstem_area_added_km2": float(matrix[0, 1:].sum() * pixel_area),
        "soil_capacity_max_error_mm": error,
        "soil_method": read_json(soil_dir / "manifest.json")["details"],
        "old_to_new_zone_pixels": matrix.tolist(),
        "areas": areas,
        "note": "Reference tables never used to alter geometry; all changes derived from routing.",
    }
    write_json(output / "checks.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "old_to_new_zone_pixels"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
