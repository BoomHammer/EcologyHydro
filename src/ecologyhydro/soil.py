"""Depth-integrated HWSD2 layer capacities, independent of ambiguous SMU AWC units."""

import csv
import json
from collections import defaultdict

import numpy as np
from osgeo import gdal

from ecologyhydro.spatial import NODATA, write_raster

# HWSD2 technical report Tables 2.2/2.3, USDA texture codes from the MDB dictionary.
TEXTURE_MM_M = {
    1: 175,
    2: 175,
    3: 175,
    4: 158,
    5: 158,
    6: 158,
    7: 158,
    8: 175,
    9: 158,
    10: 158,
    11: 125,
    12: 75,
    13: 75,
}
SOURCE = "https://pure.iiasa.ac.at/id/eprint/18595/1/cc3823en.pdf"


def layer_pawc(row):
    if row["FAO90"] in {"UR", "RK", "WR", "DS", "GG", "IS", "ST", "FP"}:
        # HWSD miscellaneous map units have no soil profile; preserve their explicit zero.
        if float(row["SOURCE_AWC"]) != 0:
            raise ValueError("Non-soil mapping unit has unexpected available-water storage")
        return 0.0
    amount = 208 if row["FAO90"].startswith("HS") else TEXTURE_MM_M[int(float(row["TEXTURE_USDA"]))]
    coarse, cec, ec = (float(row[k]) for k in ("COARSE", "CEC_CLAY", "ELEC_COND"))
    if (
        not all(np.isfinite(v) for v in (coarse, cec, ec))
        or not 0 <= coarse <= 100
        or min(cec, ec) < 0
    ):
        raise ValueError("Invalid layer coarse fragments, clay CEC or salinity")
    mineral = 0.8 if cec < 24 else 1.0
    # The report starts at EC=4. No reduction below 4; interpolate only listed levels.
    salinity = (
        0
        if ec < 4
        else np.interp(
            ec, [4, 6, 12, 16, 18, 20, 22, 25, 30], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
    )
    return amount / 1000 * (1 - coarse / 100) * mineral * (1 - salinity)


def profile_capacity(rows, depth_mm):
    """Integrate actual layer overlaps; no extrapolation across missing horizons."""
    total = covered = 0.0
    for row in sorted(rows, key=lambda r: float(r["TOPDEP"])):
        top, bottom = float(row["TOPDEP"]) * 10, float(row["BOTDEP"]) * 10
        if top >= depth_mm:
            break
        if bottom <= top or not np.isclose(top, covered):
            raise ValueError("Gap or overlap in rootable soil profile")
        overlap = min(bottom, depth_mm) - top
        total += layer_pawc(row) * overlap
        covered += overlap
    if not np.isclose(covered, depth_mm):
        raise ValueError("Incomplete layer coverage to root-restricting depth")
    return total / depth_mm, total


def soil_parameters(source, table, depth_lookup, output):
    depth_lookup = {int(key): float(value) for key, value in depth_lookup.items()}
    with table.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
        smus = {int(r["HWSD2_SMU_ID"]): r for r in records}
        if len(records) != len(smus):
            raise ValueError("Duplicate soil mapping unit")
    with table.with_name("layers.csv").open(encoding="utf-8-sig", newline="") as stream:
        profiles = defaultdict(list)
        for row in csv.DictReader(stream):
            profiles[int(row["HWSD2_SMU_ID"])].append(row)
    pawc = np.full(65536, NODATA, dtype=np.float32)
    depth, capacity = pawc.copy(), pawc.copy()
    diagnostics, invalid = [], []
    with gdal.Open(str(source)) as dataset:
        codes = dataset.ReadAsArray()
        for code in np.unique(codes):
            if int(code) not in smus:
                invalid.append(int(code))
                continue
            row = smus[int(code)]
            for horizon in profiles[int(code)]:
                horizon["FAO90"] = row["FAO90"]
                horizon["SOURCE_AWC"] = row["AWC"]
            try:
                d = depth_lookup[int(row["ROOT_DEPTH"])]
                if not np.isfinite(d) or d <= 0:
                    raise ValueError("Invalid soil depth proxy")
                fraction, total = profile_capacity(profiles[int(code)], d)
                if not 0 <= fraction <= 1:
                    raise ValueError("PAWC outside volumetric range")
            except (ValueError, KeyError) as error:
                invalid.append(int(code))
                diagnostics.append({"smu": int(code), "status": str(error)})
                continue
            pawc[code], depth[code], capacity[code] = fraction, d, total
            diagnostics.append(
                {
                    "smu": int(code),
                    "status": "layer_integrated",
                    "depth_mm": d,
                    "pawc": fraction,
                    "capacity_mm": total,
                    "source_awc_ambiguous_units": row["AWC"],
                }
            )
        for name, lookup in (("pawc", pawc), ("root_depth", depth), ("capacity_mm", capacity)):
            write_raster(
                output / f"{name}.tif",
                lookup[codes],
                dataset.GetGeoTransform(),
                dataset.GetProjection(),
            )
    (output / "profiles.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return {
        "pawc_method": "HWSD2 corrected layer capacities integrated over rootable depth",
        "source": SOURCE,
        "depth_proxy_mm": depth_lookup,
        "depth_status": "class-midpoint proxy; not measured; profile-homogeneous AWY approximation",
        "soil_components": "dominant component sequence 1, consistently with previous baseline",
        "ambiguous_smu_awc_used_in_calculation": False,
        "unmapped_codes": invalid,
        "mapped_codes": len(diagnostics)
        - len([r for r in diagnostics if r["status"] != "layer_integrated"]),
    }
