"""Preserve Gonghe terminal drainage and rebuild accumulation and station basins."""

import argparse
import time
from pathlib import Path

import numpy as np
from osgeo import gdal

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.closed_boundary import close_boundary
from ecologyhydro.config import project_root
from ecologyhydro.endorheic import check_exclusion
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import OPTIONS
from ecologyhydro.watersheds import delineate, partition, routing_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/repairs/closed_routing_v3")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root.resolve() / "project/repairs"):
        raise ValueError("Output must be inside project/repairs")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    index_path = root / "project/repairs/soil_routing/m2_index.json"
    index = read_json(index_path)
    routing = index["routing"]
    mask_path = root / "project/diagnostics/gonghe_connectivity_v1/basin_mask.tif"
    with gdal.Open(routing["direction"]) as source, gdal.Open(str(mask_path)) as mask:
        if (
            source.GetGeoTransform() != mask.GetGeoTransform()
            or not source.GetSpatialRef().IsSame(mask.GetSpatialRef())
            or (source.RasterXSize, source.RasterYSize) != (mask.RasterXSize, mask.RasterYSize)
        ):
            raise ValueError("Closed basin and D8 grids differ")
        inside = mask.ReadAsArray() == 1
        old = source.ReadAsArray()
        new, counts = close_boundary(old, inside, source.GetRasterBand(1).GetNoDataValue())
        channel = np.load(Path(routing["conditioning"]) / "network_edges.npy")
        if np.any(new[channel[:, 1], channel[:, 0]] != old[channel[:, 1], channel[:, 0]]):
            raise ValueError("Closed boundary conflicts with enforced exorheic channels")
        with gdal.GetDriverByName("GTiff").CreateCopy(
            str(output / "direction.tif"), source, options=OPTIONS
        ) as target:
            target.GetRasterBand(1).WriteArray(new)
        if not np.array_equal(
            close_boundary(new, inside, source.GetRasterBand(1).GetNoDataValue())[0], new
        ):
            raise ValueError("Boundary still contains an outgoing edge")
        pixel_km2 = abs(source.GetGeoTransform()[1] * source.GetGeoTransform()[5]) / 1e6
    del old, new
    print("Closed outgoing D8 edges:", sum(counts), flush=True)
    routing_step(output / "direction.tif", output / "accumulation.tif", "accumulation", output)
    print("Accumulation rebuilt; delineating unchanged station outlets", flush=True)
    delineate(output / "direction.tif", Path(routing["outlets"]) / "outlets.gpkg", output)
    station_manifest = read_json(Path(routing["mainstem"]) / "manifest.json")
    result = partition(
        output / "watersheds.gpkg",
        index["aligned"]["dem"],
        station_manifest["details"]["stations"],
        output,
    )
    if not result["accepted"]:
        raise ValueError("Rebuilt station basins are not nested")
    with (
        gdal.Open(str(output / "zones.tif")) as corrected,
        gdal.Open(str(Path(routing["partitions"]) / "zones.tif")) as original,
        gdal.Open(str(mask_path.parent / "zones.tif")) as masked,
        gdal.Open(str(output / "accumulation.tif")) as accumulated,
    ):
        zones, old_zones = corrected.ReadAsArray(), original.ReadAsArray()
        station_checks = []
        for row in read_json(Path(routing["outlets"]) / "outlets.json"):
            count = np.count_nonzero((zones > 0) & (zones <= row["ws_id"]))
            flow_area = float(accumulated.ReadAsArray(row["pixel_x"], row["pixel_y"], 1, 1)[0, 0])
            if abs(flow_area - count) > 1:
                raise ValueError("Accumulation and delineated area disagree")
            station_checks.append({"station": row["station"], "pixels": int(count)})
        changed = zones != old_zones
        if np.any(changed & (zones > 0)):
            raise ValueError("Edge termination unexpectedly assigned new downstream basins")
        comparison = {
            "removed_inside_km2": float(np.count_nonzero(changed & inside) * pixel_km2),
            "removed_outside_km2": float(np.count_nonzero(changed & ~inside) * pixel_km2),
            "different_from_mask_only_pixels": int(np.count_nonzero(zones != masked.ReadAsArray())),
        }
    result.update(
        comparison=comparison,
        station_accumulation_checks=station_checks,
        exclusion=check_exclusion(output / "zones.tif", mask_path),
        outgoing_edges_terminated=sum(counts),
        elapsed_seconds=time.perf_counter() - started,
        policy="Fixed HydroBASINS closed boundary; no runoff or reference-area fitting",
        limitation="D8 graph constraint, not DEM refill or internal water/groundwater model",
        references=[
            "https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf"
        ],
        sources=[
            fingerprint(p)
            for p in (
                Path(__file__),
                root / "src/ecologyhydro/closed_boundary.py",
                index_path,
                Path(routing["direction"]),
                mask_path,
                Path(routing["outlets"]) / "outlets.gpkg",
            )
        ],
    )
    write_json(output / "summary.json", result)
    print(comparison, flush=True)


if __name__ == "__main__":
    main()
