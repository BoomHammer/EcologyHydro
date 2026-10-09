"""Acyclic major-tributary D8 connections anchored on the verified mainstem."""

import numpy as np
from osgeo import gdal, ogr, osr

from ecologyhydro.spatial import crs


def attach_path(edges, nodes, path):
    """Attach new cells at the first existing downstream path; never overwrite it."""
    join = next((i for i, cell in enumerate(path) if cell in nodes), None)
    if join is None:
        return False
    for a, b in zip(path[:join], path[1 : join + 1], strict=True):
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) != 1:
            raise ValueError("Nonadjacent river connection")
        edges[a] = b
        nodes.add(a)
    return True


def major_channel_edges(network, mainstem_cells, transform, projection, min_area):
    from ecologyhydro.conditioning import loop_erased_path

    records = {}
    with ogr.Open(str(network)) as vector:
        layer = vector.GetLayer()
        layer.SetAttributeFilter(f"ENDORHEIC = 0 AND UPLAND_SKM >= {float(min_area)}")
        source_crs = layer.GetSpatialRef()
        source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        coordinate_transform = osr.CoordinateTransformation(source_crs, crs(projection))
        for feature in layer:
            geometry = feature.GetGeometryRef().Clone()
            geometry.Transform(coordinate_transform)
            records[int(feature["HYRIV_ID"])] = {
                "down": int(feature["NEXT_DOWN"]),
                "main": int(feature["MAIN_RIV"]),
                "area": float(feature["UPLAND_SKM"]),
                "points": np.asarray(geometry.GetPoints())[:, :2],
            }
    main = max(records.values(), key=lambda r: r["area"])["main"]
    records = {i: row for i, row in records.items() if row["main"] == main}
    inverse = gdal.InvGeoTransform(transform)
    paths = {}
    for identifier, row in records.items():
        points = row["points"]
        if row["down"] in records:
            downstream = records[row["down"]]["points"][[0, -1]]
            first = np.linalg.norm(downstream - points[0], axis=1).min()
            last = np.linalg.norm(downstream - points[-1], axis=1).min()
            if first < last:
                points = points[::-1]
            connection = downstream[np.linalg.norm(downstream - points[-1], axis=1).argmin()]
            if np.linalg.norm(connection - points[-1]) > abs(transform[1]):
                raise ValueError(f"Major tributary geometry gap exceeds one cell: {identifier}")
            points = np.vstack([points, connection])
        pixels = np.array([gdal.ApplyGeoTransform(inverse, *point) for point in points])
        cells = []
        for start, end in zip(pixels[:-1], pixels[1:], strict=True):
            steps = max(1, int(np.ceil(np.max(np.abs(end - start)) * 4)))
            cells.extend(np.floor(np.linspace(start, end, steps + 1)).astype(int))
        paths[identifier] = loop_erased_path(cells)
    mainstem = [tuple(c) for c in mainstem_cells]
    edges = dict(zip(mainstem[:-1], mainstem[1:], strict=True))
    nodes = set(mainstem)
    pending = sorted(records, key=lambda i: records[i]["area"], reverse=True)
    while pending:
        deferred = [i for i in pending if not attach_path(edges, nodes, paths[i])]
        if len(deferred) == len(pending):
            raise ValueError(f"Major tributaries disconnected from mainstem: {deferred[:10]}")
        pending = deferred
    result = np.array([[*a, *b] for a, b in edges.items()], dtype=np.int32)
    return result, {
        "selected_reaches": len(records),
        "minimum_upstream_area_km2": min_area,
        "connected_cells": len(nodes),
        "mainstem_priority": True,
        "policy": "connect mapped major river paths; first confluence wins; no cycles",
        "threshold_status": "fixed engineering network density, not fitted to area or runoff",
    }
