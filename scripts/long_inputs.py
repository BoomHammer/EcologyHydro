"""Reusable forcing and contemporaneous crop Kc inputs for the extended record."""

from copy import deepcopy

import numpy as np
import yaml
from osgeo import gdal, osr
from prepare_long_climate import monthly_data
from scipy.ndimage import map_coordinates

from ecologyhydro.baseline import read_json
from ecologyhydro.rotation import weighted_kc
from ecologyhydro.spatial import crs


def paths_for(root, index, product, year):
    if year <= 2018:
        folder = root / "project/repairs/long_climate_v1"
        p, et = folder / f"ppt_{year}_aligned.tif", folder / f"pet_{year}_aligned.tif"
    else:
        folder = root / "project/repairs/climate_reference_full"
        p, et = (
            folder / f"provider_ppt_{year}_aligned.tif",
            folder / f"provider_pet_{year}_aligned.tif",
        )
    return {
        "zone": str(root / "project/repairs/closed_routing_v3/zones.tif"),
        "code": index["aligned"][f"lulc_{product}"],
        "pawc": index["aligned"]["pawc"],
        "soil": index["aligned"]["root_depth"],
        "p": str(p),
        "et": str(et),
    }


def tables_for(root, index, product, years, positions, weights):
    original = read_json(root / "project/calibration/regional_water_v1/base_tables.json")[product][
        "2019"
    ]
    tables = {year: deepcopy(original) for year in years}
    if product != "fine":
        return tables, []
    with gdal.Open(index["aligned"]["lulc_fine"]) as fine:
        crop = fine.ReadAsArray().ravel()[positions] == 2
        yy, xx = np.divmod(positions[crop], fine.RasterXSize)
        gt = fine.GetGeoTransform()
        xs = gt[0] + (xx + 0.5) * gt[1] + (yy + 0.5) * gt[2]
        ys = gt[3] + (xx + 0.5) * gt[4] + (yy + 0.5) * gt[5]
        transform = osr.CoordinateTransformation(crs(fine.GetProjection()), crs("EPSG:4326"))
        lonlat = np.array(transform.TransformPoints(np.column_stack([xs, ys])))
    coefficients = yaml.safe_load((root / "config/crop_systems.yaml").read_text(encoding="utf-8"))[
        "systems"
    ]["winter_wheat_summer_maize"]["monthly_kc"]
    records = []
    for year in years:
        path = (
            root / "project/repairs/long_climate_v1" / f"terraclimate_pet_{year}.nc"
            if year <= 2018
            else root
            / "project/repairs/climate_reference_full/provider"
            / f"terraclimate_current_pet_{year}.nc"
        )
        values, source_gt, _ = monthly_data(path, "pet", year)
        col = (lonlat[:, 0] - source_gt[0]) / source_gt[1] - 0.5
        row = (lonlat[:, 1] - source_gt[3]) / source_gt[5] - 0.5
        monthly_sums = []
        for month in values:
            sampled = map_coordinates(month, [row, col], order=1, mode="constant", cval=np.nan)
            if not np.isfinite(sampled).all() or (sampled < 0).any():
                raise ValueError("Incomplete monthly crop PET coverage")
            monthly_sums.append(float(sampled @ weights[crop]))
        kc = weighted_kc(monthly_sums, coefficients)
        for entry in tables[year]:
            if int(entry["lucode"]) == 2:
                entry["kc"] = kc
        records.append(
            {
                "year": year,
                "kc": kc,
                "pet_source": str(path),
                "method": "Same-year V1.1 PET; weighted crop strata; fixed crop calendar",
            }
        )
    return tables, records
