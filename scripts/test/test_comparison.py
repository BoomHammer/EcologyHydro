"""Current temporal protocol and nine-pair evaluation cannot leak or fabricate scores."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from ecologyhydro.comparison import (
    accounts,
    load_protocol,
    metrics,
    phase_for,
    prediction_rows,
    resolve,
    summarize_rows,
    training_mask,
)
from ecologyhydro.config import project_root


@pytest.fixture
def protocol():
    return load_protocol("config/landcover_comparison.yaml")


def test_exact_user_periods_and_training_exclusions(protocol):
    years = np.array(protocol["forcing_years"])
    assert years.tolist() == list(range(2011, 2025))
    assert years[training_mask(protocol)].tolist() == [
        2013,
        2014,
        2015,
        2016,
        2017,
        2019,
        2020,
        2021,
    ]
    assert years[training_mask(protocol, True)].tolist() == list(range(2013, 2018))
    assert [phase_for(protocol, y) for y in (2011, 2018, 2024)] == [
        "warmup",
        "bridge",
        "validation",
    ]


def test_paths_do_not_depend_on_working_directory(protocol, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve(protocol, protocol["hydrology"]["observations"]) == (
        project_root() / "data/Hydrology/实测年径流量2011-2024.csv"
    )


@pytest.mark.parametrize("change", ["overlap", "gap", "order"])
def test_invalid_periods_fail_before_execution(protocol, tmp_path, change):
    config = deepcopy(protocol)
    for key in ("root", "config_path", "forcing_years"):
        del config[key]
    if change == "overlap":
        config["years"]["validation"].insert(0, 2021)
    elif change == "gap":
        config["years"]["calibration_stage1"].remove(2015)
    else:
        config["years"]["warmup"].reverse()
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered, disjoint"):
        load_protocol(path)


def test_nine_pairs_keep_all_years_but_do_not_score_warmup_or_bridge(protocol):
    shape = (14, 7)
    target = np.full(shape, 100.0)
    predicted = target + 3
    valid = np.ones(shape, dtype=bool)
    valid[3, 2:] = False
    adjustment = np.zeros(shape)
    predicted[3, 2:] = np.nan
    rows = []
    for source in protocol["products"]:
        for product in protocol["products"]:
            pair = prediction_rows(
                protocol, source, product, predicted, target, target, adjustment, valid
            )
            rows.extend(pair)
            assert len(pair) == 98
            assert not any(r["scored"] for r in pair if r["year"] in (2011, 2012, 2018))
            assert all(r["station_year_rmse"] == 3 for r in pair if r["scored"])
            summary = summarize_rows(pair)
            validation = next(
                r for r in summary if r["scope"] == "period" and r["label"] == "validation"
            )
            assert validation["n"] == 21 and validation["rmse"] == 3
            assert all(r["rmse"] is None for r in summary if r["label"] in ("2011", "2012", "2018"))
    assert len(rows) == 882
    assert len({(r["parameter_source"], r["landcover"]) for r in rows}) == 9


def test_metrics_are_rmse_not_mean_absolute_error():
    result = metrics([101, 107], [100, 100], [True, True])
    assert result["rmse"] == 5
    assert result["relative_rmse_pct"] == pytest.approx(5)
    with pytest.raises(ValueError):
        metrics([np.nan], [100], [True])
    assert metrics([np.nan], [100], [False])["rmse"] is None


def test_missing_accounts_propagate_but_not_into_other_years(protocol, monkeypatch):
    from ecologyhydro import comparison
    from ecologyhydro.long_record import STATIONS
    from ecologyhydro.water_balance import REGIONS

    years = protocol["forcing_years"]
    observed = {s: {str(y): "100" for y in years} for s in STATIONS}
    consumed = {s: {str(y): "1" for y in years} for s in REGIONS[:7]}
    stored = deepcopy(consumed)
    stored[REGIONS[2]]["2014"] = ""
    for s in consumed:
        consumed[s]["2018"] = ""
    records = iter([observed, consumed, stored])
    monkeypatch.setattr(comparison, "table", lambda *_: next(records))
    _, adjustment, valid, _, _ = accounts(protocol)
    assert valid[3].tolist() == [True, True, False, False, False, False, False]
    assert not valid[7].any()
    assert valid[13].all()
    assert np.isnan(adjustment[3, 2:]).all()


def test_completed_report_exports_exactly_nine_pairs(protocol, tmp_path):
    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.simulation import write_json
    from scripts.report_landcover_comparison import report

    protocol["root"] = tmp_path
    protocol["paths"]["output_root"] = "runs"
    output = tmp_path / "runs" / "fixture"
    output.mkdir(parents=True)
    write_json(output / "summary.json", {"status": "complete", "pairs": 9})
    write_json(
        output / "protocol.json",
        {
            "years": protocol["years"],
            "products": protocol["products"],
        },
    )
    target, predicted = np.full((14, 7), 100.0), np.full((14, 7), 103.0)
    all_rows, all_metrics = [], []
    for source in protocol["products"]:
        write_json(
            output / f"parameters_{source}.json",
            {
                "final": {
                    "training_years": np.array(protocol["forcing_years"])[
                        training_mask(protocol)
                    ].tolist()
                },
            },
        )
        for product in protocol["products"]:
            rows = prediction_rows(
                protocol,
                source,
                product,
                predicted,
                predicted,
                target,
                np.zeros((14, 7)),
                np.ones((14, 7), dtype=bool),
            )
            all_rows.extend(rows)
            all_metrics.extend(summarize_rows(rows))
    write_csv(output / "predictions.csv", all_rows)
    write_csv(output / "metrics.csv", all_metrics)
    report(protocol, "fixture")
    assert len(list((output / "pairs").glob("*.csv"))) == 9
    assert (output / "report.md").is_file()
    with pytest.raises(ValueError, match="inside"):
        report(protocol, "../outside")
    all_rows[0]["scored"] = True
    write_csv(output / "predictions.csv", all_rows)
    with pytest.raises(ValueError, match="Warmup/bridge"):
        report(protocol, "fixture")
