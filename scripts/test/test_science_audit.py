"""Check partial water accounts cannot masquerade as complete naturalization."""

import importlib.util
from pathlib import Path

import pytest

from ecologyhydro.aggregation import write_csv

SPEC = importlib.util.spec_from_file_location("audit_m4", Path(__file__).parents[1] / "audit_m4.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_partial_accounts_sign_scope_and_holdout(tmp_path):
    hydro = tmp_path / "data/Hydrology"
    hydro.mkdir(parents=True)
    output = tmp_path / "audit"
    output.mkdir()
    regions = [
        "龙羊峡以上",
        "龙羊峡至兰州",
        "兰州至头道拐",
        "头道拐至龙门",
        "龙门至三门峡",
        "三门峡至花园口",
        "花园口以下",
        "黄河内流区",
    ]
    for name, value in [
        ("地表水供水", 10),
        ("地表水耗水", 3),
        ("地下水供水", 5),
        ("地下水耗水", 2),
        ("大中型水库年蓄水变化量", -1),
    ]:
        write_csv(
            hydro / f"{name}2018-2023.csv",
            [
                {
                    "水资源二级区": region,
                    **{str(y): value for y in range(2019, 2023)},
                    "2023": "must_not_be_read",
                }
                for region in reversed(regions)
            ],
        )
    write_csv(
        hydro / "实测年径流量2018-2023.csv",
        [
            {
                "测站": station,
                **{str(y): 100 for y in range(2019, 2023)},
                "2023": "must_not_be_read",
            }
            for station in ["贵得", "花园口", "唐乃亥", "利津"]
        ],
    )
    AUDIT.audit_water(tmp_path, output)
    records = {
        r["station"]: r
        for r in AUDIT.rows(output / "station_comparison_status.csv")
        if r["year"] == "2019"
    }
    field = "partial_Q_plus_surface_consumption_plus_storage_1e8_m3"
    assert float(records["贵得"][field]) == pytest.approx(102)
    assert float(records["花园口"][field]) == pytest.approx(112)
    assert records["唐乃亥"][field] == records["利津"][field] == ""
    assert all(r["fully_naturalized_1e8_m3"] == "" for r in records.values())
    assert all(r["eligible_for_calibration"] == "False" for r in records.values())
    assert "黄河内流区" not in records["花园口"]["regions"]
