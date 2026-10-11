"""Joint development calibration of existing annual stores and one LTG net loss.

No pixel flux, observed account or year-specific parameter is changed. See
docs/FINE_FOCUS_ROUTING.md for scope, limitations and method references.
"""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.fine_priority import regional_delivery


def routing_prediction(natural, adjustment, parameters):
    """Apply the LTG loss once locally, then propagate its effect downstream."""
    natural = np.asarray(natural, dtype=float)
    adjustment = np.asarray(adjustment, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    if (
        natural.ndim != 2
        or natural.shape[1] != 7
        or not len(natural)
        or adjustment.shape != natural.shape
        or parameters.shape != (8,)
        or not np.isfinite(natural).all()
        or not np.isfinite(parameters).all()
        or not 0 <= parameters[7] <= 100
    ):
        raise ValueError("Expected finite seven-reach yields and bounded routing parameters")
    delivered, change = regional_delivery(natural, parameters[:7])
    loss = np.zeros_like(natural)
    loss[:, 2] = parameters[7]
    predicted = delivered - adjustment - loss.cumsum(axis=1)
    return predicted, delivered, change, loss


def fit_routing(natural, target, adjustment, eligible, year_weights, initial, free):
    """Fit a single spatial parameter vector to all selected years and stations.

    Zero-weight observations are never read by the residual. Chronological forcing
    remains necessary for state continuity; callers truncate at their fit horizon.
    """
    target = np.asarray(target, dtype=float)
    weights = np.asarray(year_weights, dtype=float)
    initial = np.asarray(initial, dtype=float)
    free = np.asarray(free, dtype=int)
    if weights.shape != (len(target),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Year weights must be finite and nonnegative")
    valid = np.asarray(eligible, dtype=bool) & (weights[:, None] > 0)
    observed = target[valid]
    if not len(observed) or not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("No finite positive fit observations")
    if not np.isfinite(np.asarray(adjustment)[valid]).all():
        raise ValueError("Missing fit water accounts")
    if not len(free):
        return dict(
            parameters=initial.tolist(),
            success=True,
            free_indices=[],
            observations=int(valid.sum()),
            fit_year_indices=np.flatnonzero(weights > 0).tolist(),
            starts=[],
        )
    scale = np.broadcast_to(weights[:, None], target.shape)[valid] ** 0.5 / observed
    scale /= np.sqrt(np.sum(np.broadcast_to(weights[:, None], target.shape)[valid]))

    def unpack(values):
        result = initial.copy()
        result[free] = values
        return result

    def residual(values):
        predicted = routing_prediction(natural, adjustment, unpack(values))[0]
        return (predicted[valid] - observed) * scale

    bounds = (np.zeros(8), np.r_[np.full(7, 0.5), 100.0])
    starts = [initial, np.r_[np.full(7, 0.1), initial[7]], np.r_[np.full(7, 0.35), initial[7]]]
    fits = [
        least_squares(
            residual,
            start[free],
            bounds=(bounds[0][free], bounds[1][free]),
            max_nfev=500,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        for start in starts
    ]
    successful = [f for f in fits if f.success and np.isfinite(f.fun).all()]
    if not successful:
        raise RuntimeError("No routing start converged")
    best = min(successful, key=lambda f: f.cost)
    return dict(
        parameters=unpack(best.x).tolist(),
        success=True,
        free_indices=free.tolist(),
        observations=int(valid.sum()),
        fit_year_indices=np.flatnonzero(weights > 0).tolist(),
        weighted_relative_rmse_pct=float(np.sqrt(2 * best.cost) * 100),
        starts=[dict(success=bool(f.success), nfev=int(f.nfev), cost=float(f.cost)) for f in fits],
    )


def selection_key(outcome, scores):
    """Hard goals first; then 2023 error, with no CV veto or transfer veto."""
    annual_deficit = outcome["aggregate_total"] - outcome["aggregate_best_count"]
    station_deficit = max(0, outcome["station_year_required"] - outcome["station_year_wins"])
    return (
        int(annual_deficit > 0) + int(station_deficit > 0),
        annual_deficit,
        station_deficit,
        float(scores.loc["2023_总误差", "fine"]),
        float(scores.loc["三年整体", "fine"]),
    )


def balance_ledger(natural, target, consumed, reservoir_change, parameters):
    """Return explicit local accounts, including the single residual loss term."""
    parameters = np.asarray(parameters, dtype=float)
    adjustment = (consumed + reservoir_change).cumsum(axis=1)
    predicted, delivered, change, loss = routing_prediction(natural, adjustment, parameters)
    local_natural = np.diff(natural, axis=1, prepend=0)
    local_flow = np.diff(predicted, axis=1, prepend=0)
    closure = local_natural - (local_flow + consumed + reservoir_change + change + loss)
    known = np.isfinite(closure)
    if not known.any() or np.max(np.abs(closure[known])) > 1e-8:
        raise ValueError("Regional water budget does not close")
    state = local_natural[0] * parameters[:7] / (1 - parameters[:7])
    stores = state + np.cumsum(change, axis=0)
    if np.min(stores) < -1e-8:
        raise ValueError("Negative conceptual storage")
    return dict(
        natural=natural,
        local_natural=local_natural,
        delivered=delivered,
        local_delivered=np.diff(delivered, axis=1, prepend=0),
        storage_change=change,
        storage=stores,
        loss=loss,
        consumed=consumed,
        reservoir_change=reservoir_change,
        predicted=predicted,
        local_predicted=local_flow,
        observed=target,
        closure=closure,
    )
