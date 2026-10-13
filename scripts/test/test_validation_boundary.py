"""Future holdouts must reject reused years, missing stations and changed models."""

from itertools import product

import pandas as pd
import pytest

from ecologyhydro.cache import fingerprint
from ecologyhydro.validation_boundary import check_holdout


@pytest.fixture
def sealed(tmp_path):
    model = tmp_path / "model.json"
    model.write_text('{"retention":0.2}', encoding="utf-8")
    contract = dict(
        development_years=list(range(2011, 2025)),
        eligible_years=[2025, 2026, 2027],
        products=["fine", "coarse"],
        stations=["A", "B"],
        model_files=[fingerprint(model)],
        registered_at_utc="2026-10-12T00:00:00+00:00",
    )
    records = pd.DataFrame(
        product(contract["products"], contract["eligible_years"], contract["stations"]),
        columns=["product", "year", "station"],
    )
    records = records.assign(observed=100.0, predicted=110.0, eligible=True)
    provenance = dict(
        first_observation_access_utc="2028-01-01T00:00:00+00:00",
        observation_sources=["test source"],
        forcing_sources=["test forcing"],
        account_sources=["test account"],
        access_log="test log",
    )
    return contract, records, provenance, model


def test_valid_panel_is_eligible_not_automatic_independence_certificate(sealed):
    contract, records, provenance, _ = sealed
    result = check_holdout(contract, records, provenance)
    assert result["eligible_for_audited_evaluation"]
    assert not result["independence_certified"]


@pytest.mark.parametrize(
    "failure", ["reused", "missing", "duplicate", "changed", "early", "unequal"]
)
def test_invalid_holdout_is_rejected(sealed, failure):
    contract, records, provenance, model = sealed
    if failure == "reused":
        records.loc[0, "year"] = 2024
    elif failure == "missing":
        records = records.iloc[1:]
    elif failure == "duplicate":
        records = pd.concat([records, records.iloc[[0]]])
    elif failure == "changed":
        model.write_text('{"retention":0.3}', encoding="utf-8")
    elif failure == "early":
        provenance["first_observation_access_utc"] = "2026-10-11T00:00:00+00:00"
    elif failure == "unequal":
        records.loc[0, "observed"] = 90.0
    with pytest.raises(ValueError):
        check_holdout(contract, records, provenance)
