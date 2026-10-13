"""Necessary eligibility checks before a sealed model can be tested on new years."""

from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

from ecologyhydro.cache import fingerprint


def check_holdout(contract, records, provenance):
    """Reject development reuse, incomplete panels and changed models.

    Provenance is an auditable declaration, not proof supplied by this function.
    Predictions still require a separate audit against the frozen forward calculation.
    Passing never automatically certifies scientific independence.
    """
    expected_years = set(contract["eligible_years"])
    years = set(records["year"])
    if years & set(contract["development_years"]):
        raise ValueError("Development years cannot become independent validation")
    if years != expected_years:
        raise ValueError("Require every registered year; no result-dependent early stopping")
    keys = ["product", "year", "station"]
    expected = set(product(contract["products"], expected_years, contract["stations"]))
    actual = set(records[keys].itertuples(index=False, name=None))
    if actual != expected or records.duplicated(keys).any():
        raise ValueError("Incomplete or duplicated product-year-station panel")
    if not records.eligible.eq(True).all():
        raise ValueError("Registered stations cannot be silently excluded")
    if not np.isfinite(records[["predicted", "observed"]].to_numpy()).all():
        raise ValueError("Nonfinite observations or predictions")
    if (records.observed <= 0).any():
        raise ValueError("Relative errors require positive observed flows")
    if (records.groupby(["year", "station"])["observed"].nunique() != 1).any():
        raise ValueError("Products must be compared against identical observations")
    frozen = contract["model_files"]
    if not frozen or frozen != [fingerprint(Path(item["path"])) for item in frozen]:
        raise ValueError("Frozen model or protocol changed")
    registered = datetime.fromisoformat(contract["registered_at_utc"])
    first_access = datetime.fromisoformat(provenance["first_observation_access_utc"])
    if registered.tzinfo is None or first_access.tzinfo is None or first_access <= registered:
        raise ValueError("Observation access must be documented after registration")
    for key in ("observation_sources", "forcing_sources", "account_sources", "access_log"):
        if not provenance.get(key):
            raise ValueError(f"Missing independent-data provenance: {key}")
    return dict(eligible_for_audited_evaluation=True, independence_certified=False)
