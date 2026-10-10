"""Trace independently documented inland basins and version domain candidates."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from osgeo import gdal, ogr, osr

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import OPTIONS, crs, windows

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def trace_exit(direction, mask, start):
    """Trace valid D8 cells until first polygon exit; reject cycles and bad directions."""
    offsets = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    x, y = map(int, start)
    visited, path = set(), []
    while 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        if (x, y) in visited:
            raise ValueError("D8 cycle in overlap trace")
        visited.add((x, y))
        path.append((x, y))
        if not mask[y, x]:
            return np.array(path), "polygon_exit"
        code = int(direction[y, x])
        if code not in range(8):
            return np.array(path), "terminal"
        dx, dy = offsets[code]
        x, y = x + dx, y + dy
    return np.array(path), "grid_exit"


def main():
    gdal.UseExceptions()
    ogr.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", choices=("qinghai", "gonghe"), default="gonghe")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = (root / (args.output or f"project/diagnostics/{args.basin}_connectivity_v1")).resolve()
    if not output.is_relative_to(root.resolve() / "project/diagnostics"):
        raise ValueError("Output must be inside project/diagnostics")
    output.mkdir(parents=True, exist_ok=False)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    zones_path = Path(index["routing"]["partitions"]) / "zones.tif"
    reference = root / "project/references/hydrobasins_as06/hybas_as_lev06_v1c.shp"
    seed = ogr.CreateGeometryFromWkt("POINT (100.2 36.8)")
    with ogr.Open(str(reference)) as vector:
        layer = vector.GetLayer()
        if args.basin == "qinghai":
            matches = [f for f in layer if f.GetGeometryRef().Contains(seed)]
            if len(matches) != 1 or matches[0]["ENDO"] != 2:
                raise ValueError("Qinghai Lake seed must identify a unique inland sink")
            sink = int(matches[0]["NEXT_SINK"])
        else:
            sink = 4060051460
        layer.ResetReading()
        layer.SetAttributeFilter(f"NEXT_SINK = {sink}")
        with gdal.Open(str(zones_path)) as zones:
            srs = layer.GetSpatialRef()
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            project = osr.CoordinateTransformation(srs, crs(zones.GetProjection()))
            gpkg = ogr.GetDriverByName("GPKG").CreateDataSource(str(output / "basin.gpkg"))
            target = gpkg.CreateLayer("basin", crs(zones.GetProjection()), ogr.wkbMultiPolygon)
            target.CreateField(ogr.FieldDefn("HYBAS_ID", ogr.OFTInteger64))
            ids = []
            for feature in layer:
                if feature["ENDO"] not in (1, 2):
                    raise ValueError("Exorheic polygon in inland NEXT_SINK group")
                geometry = feature.GetGeometryRef().Clone()
                geometry.Transform(project)
                item = ogr.Feature(target.GetLayerDefn())
                item["HYBAS_ID"] = int(feature["HYBAS_ID"])
                item.SetGeometry(ogr.ForceToMultiPolygon(geometry))
                target.CreateFeature(item)
                ids.append(int(feature["HYBAS_ID"]))
            with gdal.GetDriverByName("GTiff").Create(
                str(output / "basin_mask.tif"),
                zones.RasterXSize,
                zones.RasterYSize,
                1,
                gdal.GDT_Byte,
                options=OPTIONS,
            ) as mask_ds:
                mask_ds.SetGeoTransform(zones.GetGeoTransform())
                mask_ds.SetProjection(zones.GetProjection())
                mask_ds.GetRasterBand(1).Fill(0)
                gdal.RasterizeLayer(mask_ds, [1], target, burn_values=[1])
                mask = mask_ds.ReadAsArray() > 0
            gpkg.Close()
            gt, projection = zones.GetGeoTransform(), zones.GetProjection()
            original = zones.ReadAsArray()
            area = abs(gt[1] * gt[5]) / 1e6
            rows = []
            with gdal.GetDriverByName("GTiff").CreateCopy(
                str(output / "zones.tif"), zones, options=OPTIONS
            ) as corrected:
                for window in windows(zones):
                    x, y, width, height = window
                    values = zones.ReadAsArray(*window)
                    values[mask[y : y + height, x : x + width]] = 0
                    corrected.GetRasterBand(1).WriteArray(values, x, y)
            for zone in range(1, 12):
                rows.append(
                    {
                        "zone": zone,
                        "removed_area_km2": int(((original == zone) & mask).sum()) * area,
                        "old_cumulative_area_km2": int(((original > 0) & (original <= zone)).sum())
                        * area,
                        "new_cumulative_area_km2": int(
                            ((original > 0) & (original <= zone) & ~mask).sum()
                        )
                        * area,
                    }
                )
    write_csv(output / "area_change.csv", rows)
    overlap = mask & (original == 2)
    indices = np.flatnonzero(overlap)
    if not len(indices):
        raise ValueError("No overlap with Guide increment")
    with gdal.Open(index["routing"]["direction"]) as ds:
        direction = ds.ReadAsArray()
    with gdal.Open(index["routing"]["fill"]) as ds:
        filled = ds.ReadAsArray()
    with gdal.Open(index["aligned"]["dem"]) as ds:
        dem = ds.ReadAsArray()
    forced = np.load(Path(index["routing"]["conditioning"]) / "network_edges.npy")
    forced_cells = set(map(tuple, forced[:, :2]))
    to_geo = osr.CoordinateTransformation(crs(projection), crs("EPSG:4326"))
    trace_rows, paths = [], []
    for i, position in enumerate(indices[np.linspace(0, len(indices) - 1, 12).astype(int)]):
        y, x = np.unravel_index(position, original.shape)
        cells, status = trace_exit(direction, mask, (x, y))
        paths.append(cells)
        xx, yy = cells.T
        sx, sy = gdal.ApplyGeoTransform(gt, float(x) + 0.5, float(y) + 0.5)
        ex, ey = gdal.ApplyGeoTransform(gt, float(xx[-1]) + 0.5, float(yy[-1]) + 0.5)
        lon, lat, _ = to_geo.TransformPoint(ex, ey)
        trace_rows.append(
            {
                "sample": i,
                "status": status,
                "cells": len(cells),
                "start_x": sx,
                "start_y": sy,
                "exit_lon": lon,
                "exit_lat": lat,
                "forced_cells_before_exit": sum(tuple(p) in forced_cells for p in cells[:-1]),
                "max_dem_fill_m": float(np.max(filled[yy, xx] - dem[yy, xx])),
                "start_elevation_m": float(dem[y, x]),
                "exit_elevation_m": float(dem[yy[-1], xx[-1]]),
            }
        )
    write_csv(output / "trace_samples.csv", trace_rows)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    yy, xx = np.where(mask)
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    preview = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local_mask, local_zone = mask[y0:y1, x0:x1], original[y0:y1, x0:x1]
    preview[local_mask] = 1
    preview[local_mask & (local_zone > 0)] = 2
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(
        preview,
        origin="upper",
        cmap=ListedColormap(["white", "#cadfdc", "#dea065"]),
        extent=[x0, x1, y1, y0],
        vmin=0,
        vmax=2,
    )
    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color="#8b2020", linewidth=0.7)
    ax.set(
        title=f"{args.basin} 参考闭流盆地：橙色为原黄河汇水范围，红线为 D8 路径",
        xlabel="模型像元列",
        ylabel="模型像元行",
    )
    fig.tight_layout()
    fig.savefig(output / "routes.png", dpi=180)
    plt.close(fig)
    write_json(
        output / "summary.json",
        {
            "next_sink": sink,
            "hydrobasins_ids": ids,
            "domain_candidate": str(output / "zones.tif"),
            "policy": f"Exclude documented {args.basin} surface basin only",
            "boundary_source": "HydroBASINS Asia level06 v1c, explicit NEXT_SINK",
            "closed_basin_evidence": [
                "https://www.fao.org/4/X2614E/x2614e12.htm",
                "https://www.gonghe.gov.cn/lnb/zjgh1/ghnj1__zjgh/ghnj/content_1013653549",
                "https://www.sciencedirect.com/science/article/abs/pii/S0169555X26001534",
            ],
            "reference": "https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf",
            "limitation": "Approximate divides; groundwater not assumed disconnected",
            "runoff_or_area_used_to_fit_boundary": False,
            "trace_count": len(paths),
            "sources": [
                fingerprint(Path(__file__)),
                fingerprint(zones_path),
                fingerprint(reference),
                fingerprint(Path(index["routing"]["direction"])),
            ],
        },
    )
    print(rows[:3], trace_rows[:2], flush=True)


if __name__ == "__main__":
    main()
