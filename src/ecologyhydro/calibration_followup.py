"""Comparable regional candidates with a fixed, forcing-only seasonality hypothesis."""

import numpy as np

from ecologyhydro.class_response import evaluate
from ecologyhydro.fine_priority import regional_delivery
from ecologyhydro.long_record import REACH_MAP


def seasonality_factors(monthly, keep):
    """Effective wet months, normalized exclusively by this fold's training climate."""
    monthly = np.asarray(monthly, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    if monthly.shape != (len(keep), 12) or not keep.any():
        raise ValueError("Expected twelve months per year and nonempty training mask")
    if not np.isfinite(monthly).all() or (monthly < 0).any():
        raise ValueError("Invalid monthly precipitation")
    total = monthly.sum(axis=1)
    if (total <= 0).any():
        raise ValueError("Annual precipitation must be positive")
    effective = total**2 / (monthly**2).sum(axis=1)
    return effective / effective[keep].mean()


def seasonal_data(data, factors):
    """Z*factor and AWC*factor are algebraically identical inside omega only."""
    changed = data.copy()
    changed["awc"] = data["awc"].copy()
    upper = REACH_MAP[data["zone"]] == 0
    changed["awc"][upper] *= np.asarray(factors)[data["groups"][upper] // 11]
    return changed


def freeze(seed, head, regional, factors, mode):
    x = np.asarray(seed).copy()
    retention = np.full(7, x[14])
    head = [x[0], 1.0, x[14]] if head is None else head
    x[0], retention[0] = head[0], head[2]
    for region, values in regional.items():
        x[int(region)], retention[int(region)] = values
    return dict(
        seed=x.tolist(),
        head_multiplier=float(head[1]),
        retention=retention.tolist(),
        seasonal_factors=np.asarray(factors).tolist(),
        mode=mode,
    )


def predict(data, frozen, years):
    changed = seasonal_data(data, frozen["seasonal_factors"])
    upper = (REACH_MAP[data["zone"]] == 0) & (data["veg"] == 1)
    changed["kc"] = data["kc"].copy()
    changed["kc"][upper] *= frozen["head_multiplier"]
    x = np.asarray(frozen["seed"]).copy()
    loss = x[13]
    x[13:] = 0
    natural = evaluate(x, changed, years)
    delivered, _ = regional_delivery(natural, frozen["retention"])
    delivered[:, 2:] -= loss
    return delivered
