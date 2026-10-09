"""Regression checks for root-zone units and station/lake routing semantics."""

import numpy as np
import pytest
from osgeo import gdal, ogr

from ecologyhydro.channel_network import attach_path, major_channel_edges
from ecologyhydro.endorheic import check_exclusion, inland_channel_mask
from ecologyhydro.soil import layer_pawc, profile_capacity
from ecologyhydro.spatial import write_raster
from ecologyhydro.watersheds import append_feature, make_layer, make_outlets


def soil_layer(top=0, bottom=20, texture=11, coarse=20, cec=30, ec=0):
    return {
        "TOPDEP": top,
        "BOTDEP": bottom,
        "TEXTURE_USDA": texture,
        "COARSE": coarse,
        "CEC_CLAY": cec,
        "ELEC_COND": ec,
        "FAO90": "LPi",
    }


def test_layer_depth_integration_and_single_coarse_correction():
    rows = [soil_layer(), soil_layer(20, 40, coarse=40)]
    # Sandy loam: .125 * .8 * 200 + .125 * .6 * 100 = 27.5 mm.
    pawc, capacity = profile_capacity(rows, 300)
    assert capacity == pytest.approx(27.5)
    assert pawc * 300 == pytest.approx(27.5)
    assert profile_capacity(rows, 100)[1] == pytest.approx(10)
    assert layer_pawc(soil_layer(cec=20, ec=6)) == pytest.approx(0.125 * 0.8 * 0.8 * 0.8)
    with pytest.raises(ValueError, match="Incomplete"):
        profile_capacity(rows, 500)
    with pytest.raises(ValueError, match="Gap"):
        profile_capacity([soil_layer(), soil_layer(30, 40)], 400)
    nonsoil = {**soil_layer(), "FAO90": "WR", "TEXTURE_USDA": "", "SOURCE_AWC": 0}
    assert layer_pawc(nonsoil) == 0
    organic = {**soil_layer(), "FAO90": "HSf", "TEXTURE_USDA": ""}
    assert layer_pawc(organic) == pytest.approx(0.208 * 0.8)


def test_nearest_mainstem_pixel_does_not_jump_to_high_accumulation(tmp_path):
    raster = tmp_path / "accumulation.tif"
    values = np.ones((5, 10))
    values[2, 6] = 10000
    write_raster(raster, values, (0, 250, 0, 1250, 0, -250), "EPSG:32650")
    mainstem = tmp_path / "mainstem.gpkg"
    vector, layer = make_layer(mainstem, "river", "EPSG:32650", ogr.wkbLineString, [])
    append_feature(layer, ogr.CreateGeometryFromWkt("LINESTRING (125 625,2375 625)"), {})
    vector.Close()
    cells = tmp_path / "cells.npy"
    np.save(cells, np.array([(i, 2) for i in range(10)]))
    stations = [{"x": 890, "y": 625, "ws_id": 1, "station": "test"}]
    result = make_outlets(mainstem, stations, raster, 1500, tmp_path, channel_path=cells)
    assert result["stations"][0]["pixel_x"] == 3
    assert result["stations"][0]["pixel_y"] == 2
    assert result["stations"][0]["outlet_shift_m"] == pytest.approx(15)


def test_closed_lake_in_mainstem_cannot_pass(tmp_path):
    mask, zones = tmp_path / "lake.tif", tmp_path / "zones.tif"
    transform = (0, 250, 0, 500, 0, -250)
    write_raster(mask, np.array([[1, 0], [0, 0]]), transform, "EPSG:32650", nodata=0)
    write_raster(zones, np.array([[0, 1], [1, 1]]), transform, "EPSG:32650", nodata=0)
    assert check_exclusion(zones, mask)["included_in_mainstem"] == 0
    write_raster(zones, np.ones((2, 2)), transform, "EPSG:32650", nodata=0)
    with pytest.raises(ValueError, match="exclusion failed"):
        check_exclusion(zones, mask)


def test_tributary_attachment_preserves_mainstem_and_cannot_create_cycle():
    edges = {(2, 0): (2, 1), (2, 1): (2, 2)}
    nodes = {(2, 0), (2, 1), (2, 2)}
    assert attach_path(edges, nodes, [(0, 0), (1, 0), (2, 1), (1, 1)])
    assert edges[(0, 0)] == (1, 0)
    assert edges[(1, 0)] == (2, 1)
    assert edges[(2, 1)] == (2, 2)
    assert (1, 1) not in nodes
    assert not attach_path(edges, nodes, [(9, 9), (9, 8)])


def test_major_channel_graph_follows_downstream_links(tmp_path):
    source = tmp_path / "rivers.gpkg"
    fields = [
        (name, ogr.OFTInteger)
        for name in ("HYRIV_ID", "MAIN_RIV", "NEXT_DOWN", "ENDORHEIC", "UPLAND_SKM")
    ]
    vector, layer = make_layer(source, "rivers", "EPSG:32650", ogr.wkbLineString, fields)
    for identifier, downstream, area, line in (
        (1, 0, 10000, "LINESTRING (250 350,250 50)"),
        (2, 1, 3000, "LINESTRING (50 350,250 250)"),
        # Reversed source geometry must be oriented using the downstream connection.
        (3, 2, 1000, "LINESTRING (50 350,50 150)"),
    ):
        append_feature(
            layer,
            ogr.CreateGeometryFromWkt(line),
            {
                "HYRIV_ID": identifier,
                "MAIN_RIV": 1,
                "NEXT_DOWN": downstream,
                "ENDORHEIC": 0,
                "UPLAND_SKM": area,
            },
        )
    # A mapped inland river must not join the external drainage network, even
    # when its geometry touches the mainstem and its upstream area is large.
    append_feature(
        layer,
        ogr.CreateGeometryFromWkt("LINESTRING (450 350,250 250)"),
        {"HYRIV_ID": 4, "MAIN_RIV": 1, "NEXT_DOWN": 1, "ENDORHEIC": 1, "UPLAND_SKM": 5000},
    )
    vector.Close()
    edges, report = major_channel_edges(
        source, [(2, i) for i in range(4)], (0, 100, 0, 400, 0, -100), "EPSG:32650", 1000
    )
    mapping = {tuple(r[:2]): tuple(r[2:]) for r in edges}
    current = (0, 2)
    seen = set()
    while current in mapping:
        assert current not in seen
        seen.add(current)
        current = mapping[current]
    assert current == (2, 3)
    assert mapping[(2, 1)] == (2, 2)
    assert report["selected_reaches"] == 3
    assert (4, 0) not in mapping


def test_inland_mask_uses_all_inland_reaches_and_preserves_external_rivers(tmp_path):
    source = tmp_path / "rivers.gpkg"
    vector, layer = make_layer(
        source,
        "rivers",
        "EPSG:32650",
        ogr.wkbLineString,
        [("ENDORHEIC", ogr.OFTInteger), ("NEXT_DOWN", ogr.OFTInteger)],
    )
    for x, inland, downstream in ((50, 1, 2), (150, 1, 0), (250, 0, 0)):
        append_feature(
            layer,
            ogr.CreateGeometryFromWkt(f"LINESTRING ({x} 350,{x} 50)"),
            {"ENDORHEIC": inland, "NEXT_DOWN": downstream},
        )
    vector.Close()
    template = tmp_path / "dem.tif"
    write_raster(template, np.ones((4, 4)), (0, 100, 0, 400, 0, -100), "EPSG:32650")
    result = inland_channel_mask(source, template, tmp_path)
    with gdal.Open(str(tmp_path / "mask.tif")) as mask:
        values = mask.ReadAsArray()
    assert result["mapped_endorheic_reaches"] == 2
    assert np.all(values[:, :2] == 1)
    assert np.all(values[:, 2:] == 0)
