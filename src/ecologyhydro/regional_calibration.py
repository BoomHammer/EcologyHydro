"""Regional AWY hypotheses and explicit, conditional surface-water accounting."""

import numpy as np
from scipy.optimize import least_squares

# Zero-based incremental catchments. Boundaries are fixed before fitting.
# Above Lanzhou; Lanzhou--Longmen; below Longmen. These are hydrological
# proxies, not a claimed map of homogeneous climate zones.
ZONE_REGION = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
ZONE_REGION_FOUR = np.array([0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3])
ENDPOINTS = np.array([1, 2, 5, 6, 7, 8])


def yield_depth(parameters, data):
    """Independent Fu equation; one, three or four Zs followed by Kc scale."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape not in ((2,), (4,), (5,)) or not np.isfinite(parameters).all():
        raise ValueError("Expected one, three or four finite Z values and a Kc multiplier")
    regions = ZONE_REGION_FOUR if len(parameters) == 5 else ZONE_REGION
    z = parameters[0] if len(parameters) == 2 else parameters[regions[data["zone"]]]
    kc = np.where(data["veg"] == 1, np.minimum(data["kc"] * parameters[-1], 1.3), data["kc"])
    p = data["p"]
    ratio = data["et"] * kc / p
    omega = np.minimum(5, 1.25 + z * data["awc"] / p)
    aet_ratio = (
        1 + ratio - np.exp(np.logaddexp(0, omega * np.log(np.maximum(ratio, 1e-300))) / omega)
    )
    fraction = np.where(data["veg"] == 1, np.minimum(ratio, aet_ratio), np.minimum(1, ratio))
    return np.maximum(0, p * (1 - fraction))


def predict(parameters, data, years=4):
    amounts = yield_depth(parameters, data) * data["weights"]
    incremental = np.bincount(data["groups"], weights=amounts, minlength=years * 11)
    return incremental.reshape(years, 11).cumsum(axis=1)


def route(yield_at_endpoints, consumption, storage):
    """Q = cumulative Y - cumulative consumptive use - cumulative delta storage.

    Consumption and storage are NON-overlapping regional amounts, in 1e8 m3.
    Supply/withdrawal is not subtracted again, and a negative storage change
    increases flow. Unknown groundwater/exchanges remain unresolved; this is
    not a full naturalization. Negative predicted flow is retained as a failure
    diagnostic rather than clipped to improve error metrics.
    """
    y, c, s = (np.asarray(v, dtype=float) for v in (yield_at_endpoints, consumption, storage))
    if y.shape != c.shape or y.shape != s.shape or y.ndim != 2:
        raise ValueError("Expected matching year-by-region arrays")
    if not all(np.isfinite(v).all() for v in (y, c, s)) or (c < 0).any():
        raise ValueError("Missing/invalid water account; never fill unknown values with zero")
    return y - np.cumsum(c + s, axis=1)


def fit(data, target, adjustment, regional=False, keep=None, *, regions=3):
    """Fit development years only, with the same objective and multistarts."""
    target, adjustment = np.asarray(target), np.asarray(adjustment)
    if target.shape != (4, 6) or adjustment.shape != target.shape:
        raise ValueError("Require four development years and six common endpoints")
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Invalid observed runoff")
    keep = np.ones(4, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    if keep.shape != (4,) or not keep.any():
        raise ValueError("Invalid development-year mask")
    if regions not in (3, 4):
        raise ValueError("Supported regional partitions have three or four regions")
    n_z = regions if regional else 1

    def residual(parameters):
        modeled = predict(parameters, data)[:, ENDPOINTS] - adjustment
        return ((modeled - target) / target)[keep].ravel()

    fits = [
        least_squares(
            residual,
            [z] * n_z + [kc],
            bounds=([1] * n_z + [0.7], [30] * n_z + [2]),
            max_nfev=100,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for z, kc in ((5, 1), (20, 1.3), (29, 1.8))
    ]
    converged = [f for f in fits if f.success and np.isfinite(f.fun).all()]
    if not converged:
        raise ValueError("No converged regional calibration")
    best = min(converged, key=lambda f: np.mean(f.fun**2))
    return {
        "parameters": best.x.tolist(),
        "relative_rmse_pct": float(np.sqrt(np.mean(best.fun**2)) * 100),
        "jacobian_singular_values": np.linalg.svd(best.jac, compute_uv=False).tolist(),
        "active_bounds": best.active_mask.tolist(),
        "converged_starts": len(converged),
    }
