"""Scientific input contracts: classifications, P aggregation, K units and references."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from osgeo import gdal

from ecologyhydro.sediment import (
    c_parameters,
    class_p_statistics,
    observation_rows,
    run_sediment,
    sediment_coverage,
)
from ecologyhydro.spatial import ALBERS, write_raster

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def c_sources(tmp_path):
    legend = tmp_path / "legend.csv"
    legend.write_text(
        "数字,一级类,二级类,三级类,四级类\n"
        "1,A,Crop,Rainfed,Corn\n2,F,Forest,Mixed,\n3,W,Water,,\n4,A,Crop,Double,\n",
        encoding="utf-8",
    )
    crop = tmp_path / "crop.csv"
    crop.write_text("n,Ccrop\n2,0.38\n3,0.2\n", encoding="utf-8")
    noncrop = tmp_path / "noncrop.csv"
    noncrop.write_text("n,CNAn (maximum–minimum)\n5,0.0001–0.003\n11,No data\n", encoding="utf-8")
    settings = {
        "policy": {"noncrop_range_position": 0.5},
        "fine_c": {
            1: {"label": "Corn", "crop": 2},
            2: {"label": "Mixed", "noncrop": 5},
            3: {"label": "Water", "value": 0},
            4: {"label": "Double", "crop_mix": {2: 0.5, 3: 0.5}},
        },
    }
    return settings, legend, crop, noncrop


def test_c_values_and_unknown_values(c_sources):
    settings, *paths = c_sources
    values = c_parameters(settings, *paths)
    assert [r["usle_c"] for r in values] == pytest.approx([0.38, 0.00155, 0, 0.29])
    wrong = copy.deepcopy(settings)
    wrong["fine_c"][1]["label"] = "Different map"
    with pytest.raises(ValueError, match="Stale SDR"):
        c_parameters(wrong, *paths)
    wrong = copy.deepcopy(settings)
    wrong["fine_c"][3] = {"label": "Water", "noncrop": 11}
    with pytest.raises(ValueError, match="No data"):
        c_parameters(wrong, *paths)


def test_p_area_weighting_preserves_zero_and_records_missing(tmp_path):
    gdal.UseExceptions()
    paths = [tmp_path / name for name in ("lulc.tif", "p.tif", "zones.tif")]
    arrays = [
        [[1, 1, 2], [1, 2, 99]],
        [[0, 0.5, 1], [-9999, 0.2, -9999]],
        [[1, 1, 1], [1, 1, 0]],
    ]
    for path, data in zip(paths, arrays, strict=True):
        write_raster(path, np.array(data, dtype=np.float32), (0, 250, 0, 500, 0, -250), ALBERS)
    result = class_p_statistics(*paths, [1, 2, 3])
    assert result[1]["usle_p"] == pytest.approx(0.5)
    assert result[1]["p_missing_pixels"] == 1
    assert result[2]["usle_p"] == pytest.approx(0.6)
    assert result[3]["class_pixels"] == 0


def test_k_cannot_be_converted_twice(tmp_path):
    config = tmp_path / "sediment.yaml"
    config.write_text(yaml.safe_dump({"policy": {"k_multiplier": 0.1317}}), encoding="utf-8")
    with pytest.raises(ValueError, match="already metric"):
        run_sediment(config)


def test_reference_zero_is_valid_and_unknown_unit_is_not_guessed(tmp_path):
    source = tmp_path / "obs.csv"
    source.write_text("测站,2011,2012\nA,0,\nB,0.1,-1\n", encoding="utf-8")
    rows = observation_rows(source, "unknown", {"A": 1000})
    assert rows[0]["observation_status"] == "valid"
    assert rows[0]["observed_t_yr"] is None
    assert rows[1]["observation_status"] == "missing_or_invalid"
    assert rows[2]["comparison_status"] == "no_existing_station_catchment"
    assert rows[3]["observation_status"] == "missing_or_invalid"
    assert observation_rows(source, "100million_t/yr", {})[2]["observed_t_yr"] == 1e7


def test_sdr_320_hectare_units_and_stream_exclusions(tmp_path):
    gdal.UseExceptions()
    grid = {"crs": ALBERS, "bounds": [0, 0, 500, 500], "width": 2, "height": 2, "resolution": 250}
    (tmp_path / "grid.json").write_text(json.dumps({"grid": grid}), encoding="utf-8")
    (tmp_path / "protocol.yaml").write_text(
        yaml.safe_dump({"paths": {"grid_index": "grid.json", "zones": "zones.tif"}}),
        encoding="utf-8",
    )
    (tmp_path / "members.csv").write_text("station_id,zone_id\n1,1\n", encoding="utf-8")
    for name, data in {
        "zones": [[1, 1], [1, 1]],
        "sed_export": [[1, 1], [-9999, -9999]],
        "stream": [[0, 0], [1, 0]],
    }.items():
        write_raster(
            tmp_path / f"{name}.tif",
            np.array(data, dtype=np.float32),
            (0, 250, 0, 500, 0, -250),
            ALBERS,
        )
    row = sediment_coverage(
        tmp_path,
        tmp_path,
        {
            "comparison_config": "protocol.yaml",
            "zone_station": "members.csv",
        },
    )[1]
    assert row["independent_raster_sum_t_yr"] == 12.5
    assert row["stream_excluded_pixels"] == 1
    assert row["unresolved_nonstream_pixels"] == 1
    assert row["hillslope_coverage_fraction"] == pytest.approx(2 / 3)
