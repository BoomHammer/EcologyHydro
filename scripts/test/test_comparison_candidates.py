"""Prior changes must match sampled/full-pixel inputs and leave the baseline intact."""

import numpy as np

from ecologyhydro.comparison_candidates import candidate_sample, candidate_tables


def test_dryland_candidate_changes_only_kc_and_preserves_original():
    data = {
        "code": np.array([9, 11, 30]),
        "kc": np.array([0.15, 0.3, 0.6]),
        "veg": np.array([0, 1, 1]),
    }
    tables = {
        "2024": [
            dict(lucode=str(c), kc=k, lulc_veg=v)
            for c, k, v in zip(data["code"], data["kc"], data["veg"], strict=True)
        ]
    }
    recipe, spec = {"dryland_codes": {"fine": [9]}}, {"dryland_kc": 0.5}
    sample = candidate_sample(data, recipe, spec, "fine")
    full = candidate_tables(tables, recipe, spec, "fine")
    np.testing.assert_array_equal(sample["kc"], [r["kc"] for r in full["2024"]])
    np.testing.assert_array_equal(sample["veg"], data["veg"])
    assert sample["kc"][0] == 0.5
    assert data["kc"][0] == tables["2024"][0]["kc"] == 0.15
