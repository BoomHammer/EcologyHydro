"""Audit mapped inland polygons and desert yield without fitting catchment boundaries."""

import argparse
import shutil
import time
import urllib.request
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from calibrate_regional_water import block_data, open_inputs, source_paths
from osgeo import gdal, ogr, osr

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.regional_calibration import ZONE_REGION_FOUR
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import OPTIONS, crs, windows


def fetch_basins(root):
    folder = root / "project/references/hydrobasins_as06"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "hybas_as_lev06_v1c.zip"
    url = "https://f000.backblazeb2.com/file/hydrobasins/standard/hybas_as_lev06_v1c.zip?full=1"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as stream:
            total = 0
            while chunk := response.read(1024**2):
                total += len(chunk)
                if total > 50 * 1024**2:
                    raise ValueError("Reference download exceeds 50 MiB budget")
                stream.write(chunk)
                print(f"Reference download: {total / 1024**2:.1f} MiB", flush=True)
        with zipfile.ZipFile(partial) as z:
            if z.testzip():
                raise ValueError("Corrupt reference archive")
        partial.rename(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            destination = (folder / member.filename).resolve()
            if not destination.is_relative_to(folder.resolve()):
                raise ValueError("Unsafe archive member")
            if not member.is_dir() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
    write_json(
        folder / "source.json",
        {
            "product": "HydroBASINS Asia level 06 version 1c, standard",
            "catalog": "https://www.hydrosheds.org/products/hydrobasins",
            "catalog_tree": "https://f000.backblazeb2.com/file/hydrobasins/HydroBASINSstandard_files.json",
            "official_link": "https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_as_lev06_v1c.zip",
            "download_url": url,
            "attribution": "Lehner and Grill (2013), doi:10.1002/hyp.9740",
            "archive": fingerprint(archive),
            "limitation": "Shares HydroSHEDS ancestry with HydroRIVERS; not ground truth",
        },
    )
    return next(folder.rglob("*.shp"))


def build_mask(basins, template, output):
    """Rasterize both upstream inland polygons (1) and terminal polygons (2)."""
    with gdal.Open(str(template)) as reference, ogr.Open(str(basins)) as vector:
        layer = vector.GetLayer()
        # Avoid the archive's legacy ESRI .sbn spatial index (GDAL cannot read it).
        # Sequential envelope checks preserve the original reference files.
        layer.SetAttributeFilter("ENDO > 0")
        source_srs = layer.GetSpatialRef()
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source_srs, crs(reference.GetProjection()))
        target_vector = ogr.GetDriverByName("GPKG").CreateDataSource(str(output / "inland.gpkg"))
        target_layer = target_vector.CreateLayer(
            "inland", crs(reference.GetProjection()), ogr.wkbMultiPolygon
        )
        for name in ("HYBAS_ID", "ENDO", "NEXT_DOWN"):
            target_layer.CreateField(ogr.FieldDefn(name, ogr.OFTInteger64))
        records = []
        for feature in layer:
            geometry = feature.GetGeometryRef().Clone()
            xmin, xmax, ymin, ymax = geometry.GetEnvelope()
            if xmax < 95 or xmin > 120 or ymax < 31 or ymin > 43:
                continue
            geometry.Transform(transform)
            if geometry.GetGeometryType() == ogr.wkbPolygon:
                geometry = ogr.ForceToMultiPolygon(geometry)
            item = ogr.Feature(target_layer.GetLayerDefn())
            row = {name: int(feature[name]) for name in ("HYBAS_ID", "ENDO", "NEXT_DOWN")}
            for name, value in row.items():
                item[name] = value
            item.SetGeometry(geometry)
            target_layer.CreateFeature(item)
            records.append({**row, "area_km2": geometry.GetArea() / 1e6})
        with gdal.GetDriverByName("GTiff").Create(
            str(output / "inland_mask.tif"),
            reference.RasterXSize,
            reference.RasterYSize,
            1,
            gdal.GDT_Byte,
            options=OPTIONS,
        ) as mask:
            mask.SetGeoTransform(reference.GetGeoTransform())
            mask.SetProjection(reference.GetProjection())
            mask.GetRasterBand(1).Fill(0)
            gdal.RasterizeLayer(mask, [1], target_layer, burn_values=[1])
        target_vector.Close()
    write_csv(output / "inland_polygons.csv", records)


def main():
    from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/diagnostics/dryland_connectivity_v2")
    args = parser.parse_args()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root.resolve() / "project/diagnostics"):
        raise ValueError("Output must be inside project/diagnostics")
    output.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter()
    basins = fetch_basins(root)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    template = Path(index["routing"]["partitions"]) / "zones.tif"
    build_mask(basins, template, output)
    tables = read_json(root / "project/calibration/regional_water_v1/base_tables.json")
    rows = []
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    for product in ("fine", "copernicus"):
        parameters = read_json(
            root / f"project/calibration/regional_water_v2/parameters_{product}.json"
        )["legacy"]["parameters"]
        for year in range(2019, 2024):
            totals = defaultdict(lambda: np.zeros(5))
            with ExitStack() as stack:
                datasets = open_inputs(stack, source_paths(root, index, product, year))
                inland = stack.enter_context(gdal.Open(str(output / "inland_mask.tif")))
                gt = datasets["zone"].GetGeoTransform()
                area_km2 = abs(gt[1] * gt[5]) / 1e6
                for window in windows(datasets["zone"]):
                    data, valid = block_data(datasets, window, tables[product][str(year)])
                    endo = inland.ReadAsArray(*window).ravel()[valid] > 0
                    desert = np.isin(data["code"], [9, 10, 11, 12] if product == "fine" else [60])
                    kc = np.where(
                        data["veg"] == 1, np.minimum(data["kc"] * parameters[-1], 1.3), data["kc"]
                    )
                    depth = np.zeros(len(kc))
                    for region in range(4):
                        selected = ZONE_REGION_FOUR[data["zone"]] == region
                        if not selected.any():
                            continue
                        fraction = fractp_op(
                            kc[selected],
                            data["et"][selected],
                            data["p"][selected],
                            data["root_depth"][selected],
                            data["soil"][selected],
                            data["pawc"][selected],
                            data["veg"][selected],
                            nodata,
                            parameters[region],
                        )
                        depth[selected] = np.maximum(0, (1 - fraction) * data["p"][selected])
                    for zone in np.unique(data["zone"]):
                        for label, selected in (
                            ("all", np.ones(len(kc), dtype=bool)),
                            ("mapped_inland", endo),
                            ("desert_proxy", desert),
                            ("inland_or_desert", endo | desert),
                        ):
                            mask = (data["zone"] == zone) & selected
                            totals[int(zone) + 1, label] += [
                                mask.sum() * area_km2,
                                depth[mask].sum() * area_km2 / 1e5,
                                data["p"][mask].sum() * area_km2 / 1e5,
                                data["et"][mask].sum() * area_km2 / 1e5,
                                mask.sum(),
                            ]
            for (zone, label), values in sorted(totals.items()):
                rows.append(
                    {
                        "landcover": product,
                        "year": year,
                        "zone": zone,
                        "category": label,
                        "area_km2": values[0],
                        "yield_1e8_m3": values[1],
                        "precipitation_1e8_m3": values[2],
                        "et0_volume_1e8_m3": values[3],
                        "pixels": int(values[4]),
                    }
                )
            print(product, year, "audited", flush=True)
    write_csv(output / "yield_by_zone_category.csv", rows)
    write_json(
        output / "summary.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "reference": str(basins),
            "mask": str(output / "inland_mask.tif"),
            "model_domain_modified": False,
            "mask_policy": "ENDO=1 or 2, pixel centers; polygon disagreement sensitivity only",
            "desert_policy": (
                "fine classes 9-12; Copernicus 60 is bare/sparse proxy, "
                "not an equivalent desert map"
            ),
            "parameters": "regional_water_v2 legacy frozen; no refitting in this audit",
            "sources": [fingerprint(Path(__file__)), fingerprint(template), fingerprint(basins)],
        },
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
