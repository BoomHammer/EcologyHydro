"""Hydrography-constrained mainstem connectivity with explicit inland sinks."""

import json

import numpy as np
from osgeo import gdal, ogr, osr

from ecologyhydro.spatial import OPTIONS, crs


def loop_erased_path(cells):
    """Remove subpixel meander loops without creating non-neighbor D8 jumps."""
    path, positions = [], {}
    for cell in cells:
        cell = tuple(cell)
        if cell in positions:
            index = positions[cell]
            for removed in path[index + 1 :]:
                del positions[removed]
            del path[index + 1 :]
        else:
            positions[cell] = len(path)
            path.append(cell)
    return path


def channel_cells(mainstem, transform):
    inverse = gdal.InvGeoTransform(transform)
    with ogr.Open(str(mainstem)) as vector:
        # Mainstem is stored mouth to headwater; reverse feature order.
        lines = [
            np.array(feature.GetGeometryRef().GetPoints())[:, :2] for feature in vector.GetLayer()
        ][::-1]
    ordered, gaps = [], []
    for index, points in enumerate(lines):
        if index + 1 < len(lines):
            downstream = lines[index + 1][[0, -1]]
            first = np.linalg.norm(downstream - points[0], axis=1).min()
            last = np.linalg.norm(downstream - points[-1], axis=1).min()
            if first < last:
                points = points[::-1]
        elif ordered and np.linalg.norm(points[-1] - ordered[-1]) < np.linalg.norm(
            points[0] - ordered[-1]
        ):
            points = points[::-1]
        if ordered:
            gaps.append(float(np.linalg.norm(points[0] - ordered[-1])))
        ordered.extend(points)
    if max(gaps, default=0) > abs(transform[1]):
        raise ValueError("Mainstem geometry has a gap greater than one model pixel")
    pixels = np.array([gdal.ApplyGeoTransform(inverse, *point) for point in ordered])
    cells = []
    for start, end in zip(pixels[:-1], pixels[1:], strict=True):
        n = max(1, int(np.ceil(np.max(np.abs(end - start)) * 4)))
        cells.extend(np.floor(np.linspace(start, end, n + 1)).astype(int))
    return loop_erased_path(cells), max(gaps, default=0)


def condition_dem(dem, mainstem, network, output, burn_depth=5):
    with gdal.Open(str(dem)) as original:
        cells, gap = channel_cells(mainstem, original.GetGeoTransform())
        # One float32 DEM (~179 MB at 250 m); no multiyear raster stacks.
        values = original.ReadAsArray()
        nodata = original.GetRasterBand(1).GetNoDataValue()
        for x, y in cells:
            if not (0 <= x < original.RasterXSize and 0 <= y < original.RasterYSize):
                raise ValueError("Mainstem exceeds DEM coverage")
            if values[y, x] == nodata:
                raise ValueError("Mainstem intersects DEM NoData")
        xs, ys = np.array(cells).T
        elevation = values[ys, xs].copy()
        values[ys, xs] -= burn_depth
        # A single drain cell represents the mapped mouth, not an artificial basin boundary.
        values[ys[-1], xs[-1]] = nodata
        inverse = gdal.InvGeoTransform(original.GetGeoTransform())
        sinks = []
        with ogr.Open(str(network)) as vector:
            layer = vector.GetLayer()
            source_crs = layer.GetSpatialRef()
            source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            transform = osr.CoordinateTransformation(source_crs, crs(original.GetProjection()))
            for feature in layer:
                if feature["ENDORHEIC"] != 1 or feature["NEXT_DOWN"] != 0:
                    continue
                geometry = feature.GetGeometryRef().Clone()
                geometry.Transform(transform)
                # HydroRIVERS polylines run downstream; preserve the mapped terminal sink.
                point = geometry.GetPoint(geometry.GetPointCount() - 1)
                x, y = map(int, np.floor(gdal.ApplyGeoTransform(inverse, *point[:2])))
                if (
                    0 <= x < original.RasterXSize
                    and 0 <= y < original.RasterYSize
                    and values[y, x] != nodata
                ):
                    values[y, x] = nodata
                    sinks.append({"river_id": int(feature["HYRIV_ID"]), "x": x, "y": y})
        # Fail instead of imposing an ocean-connected route through a mapped inland sink.
        if any(values[y, x] == nodata for x, y in cells[:-1]):
            raise ValueError("Mapped endorheic outlet overlaps the mainstem")
        with gdal.GetDriverByName("GTiff").CreateCopy(
            str(output / "dem.tif"), original, options=OPTIONS
        ) as target:
            target.GetRasterBand(1).WriteArray(values)
        np.save(output / "channel_cells.npy", np.array(cells, dtype=np.int32))
    (output / "sinks.json").write_text(json.dumps(sinks, indent=2), encoding="utf-8")
    return {
        "channel_cells": len(cells),
        "channel_burn_m": burn_depth,
        "max_geometry_gap_m": gap,
        "endorheic_sinks": len(sinks),
        "uphill_channel_steps_before_conditioning": int((np.diff(elevation) > 0).sum()),
        "maximum_uphill_step_m": float(max(0, np.diff(elevation).max())),
        "method": "5m mainstem burn, mapped terminal drains; D8 channel enforcement after filling",
        "limitation": "Hydrography-conditioned candidate; uncharted inland sinks may remain",
    }


def enforce_channel(direction, cells_path, output):
    cells = np.load(cells_path)
    mapping = {
        (1, 0): 0,
        (1, -1): 1,
        (0, -1): 2,
        (-1, -1): 3,
        (-1, 0): 4,
        (-1, 1): 5,
        (0, 1): 6,
        (1, 1): 7,
    }
    with gdal.Open(str(direction)) as source:
        values = source.ReadAsArray()
        changed = 0
        for (x, y), (next_x, next_y) in zip(cells[:-1], cells[1:], strict=True):
            step = (int(next_x - x), int(next_y - y))
            if step not in mapping:
                raise ValueError(f"Nonadjacent channel pixels: {step}")
            changed += int(values[y, x] != mapping[step])
            values[y, x] = mapping[step]
        values[cells[-1, 1], cells[-1, 0]] = source.GetRasterBand(1).GetNoDataValue()
        with gdal.GetDriverByName("GTiff").CreateCopy(
            str(output / "data.tif"), source, options=OPTIONS
        ) as target:
            target.GetRasterBand(1).WriteArray(values)
    return {
        "changed_flow_directions": changed,
        "channel_pixels": len(cells),
        "method": "loop-erased mainstem D8 path terminates at mapped mouth drain",
    }
