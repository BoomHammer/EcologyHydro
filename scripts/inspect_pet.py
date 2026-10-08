"""Audit downloaded PET rasters, retaining raw values and recording scale 0.1."""

import json
import re
from pathlib import Path

import numpy as np
from audit_climate_values import basin_mask
from osgeo import gdal


def main():
    gdal.UseExceptions()
    root = Path(__file__).resolve().parents[1]
    records = []
    reference_grid = None
    coverage = None
    inside = None
    for path in sorted((root / "data/Climate/TerraClimate_PET").glob("*.tif")):
        before = path.stat()
        with gdal.Open(str(path)) as dataset:
            grid = (
                dataset.RasterXSize,
                dataset.RasterYSize,
                dataset.GetGeoTransform(),
                dataset.GetProjection(),
            )
            if reference_grid is None:
                reference_grid = grid
                width, height, gt, _ = grid
                if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
                    raise ValueError("Expected a north-up PET grid")
                padding = 100
                lon = gt[0] + (np.arange(-padding, width + padding) + 0.5) * gt[1]
                lat = (gt[3] + (np.arange(-padding, height + padding) + 0.5) * gt[5])[::-1]
                extended = basin_mask(root, lon, lat)[::-1]
                inside = extended[padding : padding + height, padding : padding + width]
                if not inside.any():
                    raise ValueError("No provisional basin pixels in PET grid")
                coverage = {
                    "method": "Pixel centers, approximate datum; inner basin not removed",
                    "inside_grid": int(inside.sum()),
                    "outside_grid": int(extended.sum() - inside.sum()),
                    "padded_mask_touches_edge": bool(
                        extended[0].any()
                        or extended[-1].any()
                        or extended[:, 0].any()
                        or extended[:, -1].any()
                    ),
                }
            elif grid != reference_grid:
                raise ValueError(f"PET grid mismatch: {path.name}")
            band = dataset.GetRasterBand(1)
            values = band.ReadAsArray()
            valid = (band.GetMaskBand().ReadAsArray() != 0) & np.isfinite(values)
            selected = values[valid]
            match = re.fullmatch(r"PET(\d{2})(\d{2})", path.stem)
            records.append(
                {
                    "file": path.name,
                    "month_from_filename_only": (f"20{match[1]}-{match[2]}" if match else None),
                    "shape": list(values.shape),
                    "band_count": dataset.RasterCount,
                    "transform": dataset.GetGeoTransform(),
                    "projection": dataset.GetProjection(),
                    "metadata": dataset.GetMetadata(),
                    "band_metadata": band.GetMetadata(),
                    "description": band.GetDescription(),
                    "dtype": gdal.GetDataTypeName(band.DataType),
                    "unit": band.GetUnitType(),
                    "scale": band.GetScale(),
                    "offset": band.GetOffset(),
                    "nodata": band.GetNoDataValue(),
                    "invalid_pixels": int(values.size - selected.size),
                    "zero_pixels": int(np.count_nonzero(selected == 0)),
                    "negative_pixels": int(np.count_nonzero(selected < 0)),
                    "basin_invalid_pixels": int(np.count_nonzero(inside & ~valid)),
                    "basin_zero_pixels": int(np.count_nonzero(inside & valid & (values == 0))),
                    "raw_min": float(selected.min()) if selected.size else None,
                    "raw_max": float(selected.max()) if selected.size else None,
                }
            )
        after = path.stat()
        records[-1]["unchanged_during_read"] = (
            before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        )
    output = root / "project/cache/inventory/pet_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    expected = {f"{year}-{month:02d}" for year in range(2019, 2024) for month in range(1, 13)}
    months = [record["month_from_filename_only"] for record in records]
    summary = {
        "files": len(records),
        "missing_months": sorted(expected - set(months)),
        "unexpected_months": [month for month in months if month not in expected],
        "duplicate_months": len(months) != len(set(months)),
        "coverage": coverage,
        "scale_to_monthly_mm": 0.1,
        "invalid_pixels": sum(record["invalid_pixels"] for record in records),
        "negative_pixels": sum(record["negative_pixels"] for record in records),
        "all_files_stable": all(record["unchanged_during_read"] for record in records),
    }
    output.with_name("pet_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    for record in records:
        print(
            record["file"],
            record["shape"],
            record["raw_min"],
            record["raw_max"],
            "invalid:",
            record["invalid_pixels"],
            "zero:",
            record["zero_pixels"],
        )
    if records:
        print(
            "Same grid:",
            all(
                all(record[key] == records[0][key] for key in ("shape", "transform", "projection"))
                for record in records
            ),
        )


if __name__ == "__main__":
    main()
