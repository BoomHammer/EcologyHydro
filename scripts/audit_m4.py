"""Reproducible calibration-period science audit; no parameter fitting."""

import csv
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import yaml
from osgeo import gdal

from ecologyhydro.aggregation import read_observations, write_csv
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.model_inputs import verified_artifact
from ecologyhydro.spatial import windows


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_rasters(index, priors, root, output):
    paths = {
        "zones": str(Path(index["routing"]["partitions"]) / "zones.tif"),
        **{
            name: path
            for name, path in index["aligned"].items()
            if name in {"pawc", "root_depth", "lulc_fine", "lulc_copernicus"}
            or any(name == f"{v}_{y}" for v in ("prec", "bcpr", "pet") for y in range(2019, 2023))
        },
    }
    for path in paths.values():
        verified_artifact(Path(path))
    stats, classes, cross_counts = {}, {}, {}
    with ExitStack() as stack:
        ds = {name: stack.enter_context(gdal.Open(path)) for name, path in paths.items()}
        zones = ds["zones"]
        for name, raster in ds.items():
            if (
                raster.GetGeoTransform() != zones.GetGeoTransform()
                or raster.GetProjection() != zones.GetProjection()
                or (raster.RasterXSize, raster.RasterYSize)
                != (zones.RasterXSize, zones.RasterYSize)
            ):
                raise ValueError(f"Grid mismatch: {name}")
        gt = zones.GetGeoTransform()
        area = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e6
        for block in windows(zones):
            ids = zones.ReadAsArray(*block)
            inside = ids > 0
            if not inside.any():
                continue
            data = {name: raster.ReadAsArray(*block) for name, raster in ds.items()}
            pair_ids = data["lulc_fine"][inside].astype(np.int64) * 256
            pair_ids += data["lulc_copernicus"][inside].astype(np.int64)
            pairs, frequencies = np.unique(pair_ids, return_counts=True)
            for pair, count in zip(pairs, frequencies, strict=True):
                cross_counts[int(pair)] = cross_counts.get(int(pair), 0) + int(count)
            for name, values in data.items():
                if name == "zones":
                    continue
                valid = np.isfinite(values) & (
                    ds[name].GetRasterBand(1).GetMaskBand().ReadAsArray(*block) > 0
                )
                if np.any(inside & ~valid):
                    raise ValueError(f"Missing input: {name}")
                if name.startswith("lulc"):
                    continue
                for zone in np.unique(ids[inside]):
                    selected = values[ids == zone].astype(np.float64)
                    rec = stats.setdefault(
                        (name, int(zone)), [0, 0.0, float("inf"), float("-inf"), 0]
                    )
                    rec[0] += selected.size
                    rec[1] += selected.sum()
                    rec[2] = min(rec[2], selected.min())
                    rec[3] = max(rec[3], selected.max())
                    rec[4] += np.count_nonzero(selected == 0)
            for product, mapping in priors["mapping"].items():
                codes = data[f"lulc_{product}"]
                for code in np.unique(codes[inside]):
                    group = mapping[str(int(code))]
                    params = priors["groups"][group]
                    selected = inside & (codes == code)
                    count = np.count_nonzero(selected)
                    rec = classes.setdefault((product, int(code)), [0, 0, 0, 0.0])
                    rec[0] += count
                    rec[1] += (
                        np.count_nonzero(selected & (data["root_depth"] < params["root_depth"]))
                        if params["lulc_veg"]
                        else 0
                    )
                    rec[2] += np.count_nonzero(selected & (data["pawc"] == 0))
                    if params["lulc_veg"]:
                        rec[3] += np.sum(
                            np.minimum(data["root_depth"][selected], params["root_depth"])
                            * data["pawc"][selected],
                            dtype=np.float64,
                        )
    stat_rows = [
        {
            "variable": name,
            "zone_id": zone,
            "pixels": rec[0],
            "area_km2": rec[0] * area,
            "mean": rec[1] / rec[0],
            "min": rec[2],
            "max": rec[3],
            "zero_pixels": rec[4],
        }
        for (name, zone), rec in sorted(stats.items())
    ]
    write_csv(output / "input_zone_statistics.csv", stat_rows)
    class_rows = []
    for (product, code), rec in sorted(classes.items()):
        legend_rows = rows(root / index["recipe"]["class_tables"][product])
        legend = {int(next(iter(r.values()))): r for r in legend_rows}
        group = priors["mapping"][product][str(code)]
        params = priors["groups"][group]
        class_rows.append(
            {
                "product": product,
                "lucode": code,
                "legend": json.dumps(legend[code], ensure_ascii=False),
                "group": group,
                **params,
                "pixels": rec[0],
                "area_km2": rec[0] * area,
                "soil_limited_fraction": rec[1] / rec[0] if params["lulc_veg"] else None,
                "zero_pawc_fraction": rec[2] / rec[0],
                "mean_effective_awc_mm": rec[3] / rec[0] if params["lulc_veg"] else None,
                "evidence": "engineering_prior_not_locally_validated",
                "source": "config/biophysical_priors.yaml",
            }
        )
    write_csv(output / "class_parameter_audit.csv", class_rows)
    write_csv(
        output / "landcover_crosswalk_area.csv",
        [
            {
                "fine_code": pair // 256,
                "copernicus_code": pair % 256,
                "pixels": count,
                "area_km2": count * area,
                "meaning": "spatial_overlap_not_accuracy_or_semantic_equivalence",
            }
            for pair, count in sorted(cross_counts.items())
        ],
    )
    return paths, stat_rows, class_rows


def audit_water(root, output):
    hydro = root / "data/Hydrology"
    names = ["地表水供水", "地表水耗水", "地下水供水", "地下水耗水", "大中型水库年蓄水变化量"]
    sources = {name: hydro / f"{name}2018-2023.csv" for name in names}
    tables = {name: {r["水资源二级区"]: r for r in rows(path)} for name, path in sources.items()}
    regions = [
        "龙羊峡以上",
        "龙羊峡至兰州",
        "兰州至头道拐",
        "头道拐至龙门",
        "龙门至三门峡",
        "三门峡至花园口",
        "花园口以下",
        "黄河内流区",
    ]
    if any(set(table) != set(regions) for table in tables.values()):
        raise ValueError("Water accounting region mismatch")
    regional = []
    endpoints = {"贵得": 1, "兰州": 2, "头道拐": 3, "龙门": 4, "三门峡": 5, "花园口": 6}
    station_rows = []
    for year in range(2019, 2023):
        for region in regions:
            values = {name: float(table[region][str(year)]) for name, table in tables.items()}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError("Non-finite water accounting input")
            for kind in ("地表水", "地下水"):
                if not 0 <= values[kind + "耗水"] <= values[kind + "供水"]:
                    raise ValueError("Consumption outside supply bounds")
            regional.append(
                {
                    "region": region,
                    "year": year,
                    "unit": "1e8_m3",
                    **values,
                    "mainstem_included": region != "黄河内流区",
                }
            )
        observations = read_observations(hydro / "实测年径流量2018-2023.csv", year)
        for station, obs in observations.items():
            number = endpoints.get(station)
            selected = regions[:number] if number else []
            if "黄河内流区" in selected:
                raise ValueError("Endorheic region cannot enter mainstem sum")
            sums = {
                name: sum(float(tables[name][r][str(year)]) for r in selected) if selected else None
                for name in names
            }
            candidate = (
                obs["value"] + sums["地表水耗水"] + sums["大中型水库年蓄水变化量"]
                if number and obs["value"] is not None
                else None
            )
            station_rows.append(
                {
                    "station": station,
                    "year": year,
                    "observed_1e8_m3": obs["value"],
                    "regions": ";".join(selected),
                    "mapping": "endpoint_proxy" if number else "no_complete_region_mapping",
                    **{f"upstream_{k}_1e8_m3": v for k, v in sums.items()},
                    "partial_Q_plus_surface_consumption_plus_storage_1e8_m3": candidate,
                    "fully_naturalized_1e8_m3": None,
                    "comparable_model_runoff_1e8_m3": None,
                    "eligible_for_calibration": False,
                    "limitations": (
                        "region_boundary_mismatch;ET_overlap_unresolved;"
                        "transfers_unknown;groundwater_exchange_unknown"
                    ),
                }
            )
    write_csv(output / "regional_water_accounts.csv", regional)
    write_csv(output / "station_comparison_status.csv", station_rows)
    return list(sources.values()) + [hydro / "实测年径流量2018-2023.csv"]


def main():
    gdal.UseExceptions()
    root = project_root()
    output = root / "project/m4/audit"
    output.mkdir(parents=True, exist_ok=True)
    index_path = root / "project/cache/m2/latest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prior_path = root / "config/biophysical_priors.yaml"
    priors = yaml.safe_load(prior_path.read_text(encoding="utf-8"))
    paths, stats, classes = audit_rasters(index, priors, root, output)
    water_paths = audit_water(root, output)
    summary = {
        "status": "audit_completed_scientific_issues_open",
        "calibration_ready": False,
        "years": [2019, 2020, 2021, 2022],
        "validation_used_for_selection": False,
        "station_year_rows": 44,
        "partial_proxy_rows": 24,
        "classes": {
            p: {
                "count": sum(r["product"] == p for r in classes),
                "unique_parameter_tuples": len(
                    {
                        (r["lulc_veg"], r["kc"], r["root_depth"])
                        for r in classes
                        if r["product"] == p
                    }
                ),
            }
            for p in priors["mapping"]
        },
        "input_fingerprints": [
            fingerprint(Path(p))
            for p in [
                index_path,
                prior_path,
                Path(__file__),
                root / "docs/DATA_DESCRIPTION.md",
                *[root / path for path in index["recipe"]["class_tables"].values()],
                *paths.values(),
                *water_paths,
            ]
        ],
        "zone_statistic_rows": len(stats),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "input_fingerprints"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
