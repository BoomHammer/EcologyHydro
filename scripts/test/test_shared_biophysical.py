"""Scientific invariants of the shared registry and sample translation."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ecologyhydro.shared_biophysical import (
    FIELDS,
    load_registry,
    parameter_table,
    replace_sample_parameters,
    resolve_profile,
)


def registry():
    return load_registry(Path("config/shared_biophysical.yaml"))


def test_identical_profiles_across_maps_and_bare_branch():
    spec = registry()
    seen = {}
    for product in spec["mapping"]:
        for row in parameter_table(spec, product):
            values = tuple(row[field] for field in FIELDS)
            assert seen.setdefault(row["profile"], values) == values
            if row["profile"] == "bare":
                assert row["lulc_veg"] == 0
                assert row["root_depth"] == 1


def test_root_units_and_no_unjustified_steppe_extrapolation():
    spec = registry()
    tree = resolve_profile(spec, "tree")
    assert tree["root_depth"] == pytest.approx(983.522725, rel=1e-6)
    assert resolve_profile(spec, "alpine_meadow")["root_depth"] == 300
    assert (
        resolve_profile(spec, "alpine_steppe")["root_depth"]
        == resolve_profile(spec, "grass")["root_depth"]
    )
    assert resolve_profile(spec, "alpine_meadow", "parent") == resolve_profile(spec, "grass")


def test_inheritance_validation():
    spec = registry()
    spec["profiles"]["tree"]["parent"] = "deciduous_tree"
    with pytest.raises(ValueError, match="Cyclic"):
        resolve_profile(spec, "tree")
    spec = registry()
    spec["profiles"]["open_tree"]["kc"] = 0.4
    with pytest.raises(ValueError, match="evidence"):
        resolve_profile(spec, "open_tree")


def test_updated_residential_legend_shift():
    rows = {row["lucode"]: row for row in parameter_table(registry(), "fine")}
    assert rows[24]["profile"] == "built"
    assert rows[24]["lulc_veg"] == 0
    assert rows[25]["profile"] == "tree"
    assert rows[26]["profile"] == rows[27]["profile"] == "water"
    assert rows[28]["profile"] == "wetland"
    assert rows[29]["profile"] == "snow"


def test_translation_preserves_forcing_and_rebuilds_awc():
    data = {
        "code": np.array([9, 18]),
        "soil": np.array([500, 500]),
        "pawc": np.array([0.2, 0.2]),
        "p": np.array([400, 600]),
        "weights": np.array([1.0, 2.0]),
        "awc": np.array([99.0, 99.0]),
    }
    original = deepcopy(data)
    out = replace_sample_parameters(data, parameter_table(registry(), "fine"))
    np.testing.assert_allclose(out["awc"], [0.2, 60])
    np.testing.assert_array_equal(out["veg"], [0, 1])
    for key in original:
        np.testing.assert_array_equal(data[key], original[key])
    assert out["weights"] is data["weights"]
    data["code"][0] = 95
    with pytest.raises(ValueError, match="Unmapped"):
        replace_sample_parameters(data, parameter_table(registry(), "fine"))
