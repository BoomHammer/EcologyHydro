"""Calibration must match the chosen response without leaking held years."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from ecologyhydro.comparison import load_protocol
from ecologyhydro.shared_biophysical import load_registry, parameter_table
from ecologyhydro.simulation import write_json


@pytest.mark.parametrize("mode,expected_root", [("parent", 609.01), ("detail", 300)])
def test_shared_fit_response_and_excluded_years(monkeypatch, tmp_path, mode, expected_root):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("experiment_shared_parameters")
    config = load_protocol("config/landcover_comparison.yaml")
    years = np.asarray(config["forcing_years"])
    raw = {
        "groups": np.arange(len(years)) * 11,
        "zone": np.zeros(len(years), dtype=int),
        "code": np.full(len(years), 18),
        "soil": np.full(len(years), 2000),
        "pawc": np.full(len(years), 0.2),
        "p": years,
    }
    registry_path = Path("config/shared_biophysical.yaml")
    registry = load_registry(registry_path)
    # All three test products represent the same meadow: equivalent classes must agree.
    registry["mapping"] = {p: {"18": "alpine_meadow"} for p in config["products"]}
    monkeypatch.setattr(module, "load_registry", lambda _: registry)
    monkeypatch.setattr(module, "worker_setup", lambda _: config)
    monkeypatch.setattr(module, "load_samples", lambda *_: raw)
    target = np.ones((len(years), 7))
    monkeypatch.setattr(module, "accounts", lambda _: (target, target, target > 0, None, None))
    write_json(tmp_path / "experiment.json", dict(fit_mode=mode, baseline_run="unused"))

    def capture(data, *_):
        assert set(data["p"]) == {2016, 2017, 2019, 2020, 2021}
        expected = parameter_table(registry, "fine", mode)[0]["root_depth"]
        np.testing.assert_allclose(data["root_depth"], expected)
        np.testing.assert_allclose(data["awc"], expected * 0.2)
        if mode == "detail":
            assert expected == expected_root
        else:
            assert expected > 600
        assert len(np.unique(data["groups"])) == 15
        return {"parameters": [5] * 7 + [1, 0]}

    monkeypatch.setattr(module, "fit_transfer", capture)
    module.fit_task("unused", str(tmp_path), "test", [2013, 2014, 2015])


def test_single_root_ablation_preserves_other_profiles(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("diagnose_fine_response")
    registry = load_registry(Path("config/shared_biophysical.yaml"))
    parent = parameter_table(registry, "fine", "parent")
    changed = module.case_table(registry, "fine", "root_alpine_meadow")
    for before, after in zip(parent, changed, strict=True):
        if before["profile"] == "alpine_meadow":
            assert after["root_depth"] == 300
        else:
            assert before == after


def test_carryover_fit_cannot_see_validation_observations_or_forcing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("experiment_shared_carryover")
    natural = np.array([100, 150, 80, 200, 90, 120, 220, 100, 110, 230, 70, 140, 150, 90])
    natural = natural[:, None] * np.arange(1, 8)[None, :]
    target = module.delivered(natural, 0.2)
    keep = np.zeros(14, dtype=bool)
    keep[[2, 3, 4, 5, 6, 8, 9, 10]] = True
    adjustment = np.zeros_like(target)
    eligible = np.ones_like(target, dtype=bool)
    fit = module.fit_retention(
        {"a": natural, "b": natural * 1}, target, adjustment, eligible, keep, 0
    )
    assert fit["retention"] == pytest.approx(0.2, abs=1e-5)
    target[11:] = np.nan
    natural[11:] = 1e15
    poisoned = module.fit_retention(
        {"a": natural, "b": natural * 1}, target, adjustment, eligible, keep, 0
    )
    assert fit == poisoned
    np.testing.assert_allclose(module.delivered(natural, 0), natural)
