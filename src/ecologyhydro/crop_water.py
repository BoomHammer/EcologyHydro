"""Conservative crop-process diagnostics, distinct from official InVEST AWY.

Rainfall fills a finite soil bucket; overflow drains and water-limited ET empties
the bucket. Supplemental irrigation is an explicitly consumed, same-month flux,
not gross diversion or inferred observed pixel irrigation. See the process report.
"""

import numpy as np


def rain_bucket(precipitation, demand, capacity, initial_fraction=0.5):
    """Advance monthly states (time, pixel), returning depths in mm."""
    p, d = np.asarray(precipitation, float), np.asarray(demand, float)
    c = np.asarray(capacity, float)
    if p.ndim != 2 or p.shape != d.shape or c.shape != p.shape[1:]:
        raise ValueError("Expected monthly pixel matrices and one capacity per pixel")
    if not all(np.isfinite(a).all() and (a >= 0).all() for a in (p, d, c)):
        raise ValueError("Forcing and capacity must be finite and nonnegative")
    if not np.isfinite(initial_fraction) or not 0 <= initial_fraction <= 1:
        raise ValueError("Initial fraction must lie in [0, 1]")
    state = c * initial_fraction
    q, et, delta = (np.empty_like(p) for _ in range(3))
    for t in range(len(p)):
        old = state.copy()
        available = old + p[t]
        stored = np.minimum(available, c)
        q[t] = np.maximum(available - c, 0)
        # Exponential stress is a conceptual monthly approximation, not FAO-56
        # daily Ks. The zero-capacity limit has no retained water.
        et[t] = stored * -np.expm1(-d[t] / np.maximum(c, 1e-12))
        et[t] = np.where(c > 0, et[t], np.minimum(p[t], d[t]))
        q[t] = np.where(c > 0, q[t], p[t] - et[t])
        state = stored - np.where(c > 0, et[t], 0)
        delta[t] = state - old
    closure = p - et - q - delta
    if np.max(np.abs(closure), initial=0) > 1e-8:
        raise ValueError("Crop rainfall balance failed")
    return dict(q=q, aet=et, delta_soil=delta, final_soil=state, closure=closure)


def consumptive_supplement(deficit, weights, irrigated, budget):
    """Allocate an annual ET budget by deficit, never to rainfed land.

    weights convert mm to the budget volume unit. The caller must exclude
    exports, other sectors and groundwater from a surface-irrigation budget.
    Allocation uses the whole year's deficits: accounting diagnostic, not an
    operational scheduling algorithm. Unknown budgets remain unknown.
    """
    deficit, weights = np.asarray(deficit, float), np.asarray(weights, float)
    irrigated = np.asarray(irrigated, bool)
    if deficit.ndim != 2 or weights.shape != deficit.shape[1:] or irrigated.shape != weights.shape:
        raise ValueError("Incompatible deficit, weights and irrigation mask")
    if not all(np.isfinite(a).all() and (a >= 0).all() for a in (deficit, weights)):
        raise ValueError("Invalid deficit or area weights")
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Irrigation budget must be known and nonnegative")
    permitted = np.where(irrigated[None, :], deficit, 0)
    maximum = float(permitted.sum(axis=0) @ weights)
    factor = min(1.0, budget / maximum) if maximum else 0.0
    supplement = permitted * factor
    used = float(supplement.sum(axis=0) @ weights)
    return supplement, used, budget - used


def managed_balance(rain_yield, total_consumption, exports, crop_et, reservoir_delta):
    """Replace the modeled part of consumption exactly once; preserve unknowns.

    All inputs are local volume increments. 'rain_yield' already includes soil
    storage changes. Consumption includes exports in the bulletin accounting.
    P-ET-total includes crop_et; residual consumption excludes that same flux.
    """
    y, c, x, e, s = np.broadcast_arrays(
        *[
            np.asarray(a, float)
            for a in (rain_yield, total_consumption, exports, crop_et, reservoir_delta)
        ]
    )
    known = np.isfinite(c) & np.isfinite(x) & np.isfinite(e)
    if np.any(known & ((x < 0) | (e < 0) | (c < x + e - 1e-9))):
        raise ValueError("Exports plus modeled crop ET exceed the consumption account")
    residual = c - x - e
    net_land = y - e
    runoff = net_land - residual - x - s
    return dict(runoff=runoff, net_land=net_land, residual_consumption=residual, exports=x)
