"""Fine-map headwater calibration and conservative regional annual delivery."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.class_response import evaluate
from ecologyhydro.independent_local_balance import reach_data
from ecologyhydro.regional_transfer import training_data


def rainfall_factors(climate, years, keep, use_reference):
    """Reference precipitation is a forcing proxy, never a runoff-based correction."""
    if not use_reference:
        return np.ones(len(years))
    head = climate[climate.region == 0].set_index("year").loc[years]
    ratios = (head.reference_p / head.p).to_numpy()
    known = np.isfinite(ratios) & (ratios > 0)
    if not np.all(known[keep]):
        raise ValueError("Missing training precipitation reference")
    return np.where(known, ratios, np.mean(ratios[keep]))


def head_prediction(parameters, data, seed, years, factors):
    local = reach_data(data, 0)
    local["p"] *= factors[local["groups"] // 11]
    local["kc"] = np.where(local["veg"] == 1, local["kc"] * parameters[1], local["kc"])
    x = np.asarray(seed).copy()
    x[0], x[14], x[13] = parameters[0], parameters[2], 0.0
    return evaluate(x, local, years)[:, 0]


def fit_head(data, target, adjustment, eligible, keep, seed, factors, settings):
    horizon = np.flatnonzero(keep)[-1] + 1
    forcing = np.arange(len(keep)) < horizon
    local = training_data(reach_data(data, 0), forcing)
    observed, known = target[:horizon, 0], adjustment[:horizon, 0]
    valid = keep[:horizon] & eligible[:horizon, 0]
    if valid.sum() < 4 or not np.isfinite((observed + known)[valid]).all():
        raise ValueError("Insufficient headwater training data")

    def residual(parameters):
        predicted = head_prediction(parameters, local, seed, horizon, factors[:horizon])
        return (predicted - known - observed)[valid] / observed[valid]

    fits = [
        least_squares(
            residual,
            start,
            bounds=settings["headwater_bounds"],
            x_scale="jac",
            max_nfev=settings["max_nfev"],
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        for start in settings["headwater_starts"]
    ]
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
    if not successful:
        raise RuntimeError("All headwater starts failed")
    best = min(successful, key=lambda fit: fit.cost)
    return dict(
        parameters=best.x.tolist(),
        training_indices=np.flatnonzero(keep).tolist(),
        forcing_indices=np.flatnonzero(forcing).tolist(),
        objective=float(2 * best.cost),
        active_bounds=best.active_mask.tolist(),
        success=True,
        starts=[dict(success=bool(f.success), nfev=int(f.nfev), cost=float(f.cost)) for f in fits],
    )


def predict_candidate(data, seed, lower_z, head, factors, years):
    x = np.asarray(seed).copy()
    if lower_z is not None:
        x[6] = lower_z
    predicted = evaluate(x, data, years)
    if head is not None:
        original_head = predicted[:, 0].copy()
        new_head = head_prediction(head, data, x, years, factors)
        predicted += (new_head - original_head)[:, None]
    return predicted


def regional_delivery(cumulative, retention):
    """Preserve water in seven local stores; no second routing of upstream flow."""
    natural = np.diff(np.asarray(cumulative), axis=1, prepend=0)
    retention = np.asarray(retention)
    if np.any((retention < 0) | (retention > 0.5)) or np.any(natural < -1e-8):
        raise ValueError("Invalid regional storage inputs")
    state = natural[0] * retention / (1 - retention)
    delivered, change = [], []
    for value in natural:
        available = state + value
        next_state = retention * available
        delivered.append((1 - retention) * available)
        change.append(next_state - state)
        state = next_state
    delivered, change = np.asarray(delivered), np.asarray(change)
    np.testing.assert_allclose(delivered + change, natural, atol=1e-9)
    return delivered.cumsum(axis=1), change
