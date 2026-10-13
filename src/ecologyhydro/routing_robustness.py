"""Bounded, regularized annual routing with analytic derivatives and shared regions."""

import numpy as np
from scipy.optimize import least_squares

GROUPS = {
    "shared": np.zeros(7, dtype=int),
    "head_rest": np.array([0, 1, 1, 1, 1, 1, 1]),
    "upper_lower": np.array([0, 0, 0, 1, 1, 1, 1]),
    "three_zone": np.array([0, 0, 1, 1, 1, 2, 2]),
    "regional": np.arange(7),
}
SCALES = np.r_[np.full(7, 0.1), 20.0]


def forward(natural, parameters, jacobian=False):
    """Cumulative delivery minus one local LTG sink; known accounts are external."""
    natural = np.asarray(natural, dtype=float)
    x = np.asarray(parameters, dtype=float)
    if (
        natural.ndim != 2
        or natural.shape[1] != 7
        or len(natural) == 0
        or not np.isfinite(natural).all()
        or x.shape != (8,)
        or not np.isfinite(x).all()
        or (x[:7] < 0).any()
        or (x[:7] > 0.5).any()
        or not 0 <= x[7] <= 100
    ):
        raise ValueError("Invalid routing yields or parameters")
    local = np.diff(natural, axis=1, prepend=0)
    if (local < -1e-8).any():
        raise ValueError("Negative local annual yield")
    r = x[:7]
    state = r / (1 - r) * local[0]
    state_jac = local[0] / (1 - r) ** 2
    prediction = np.empty_like(natural)
    derivative = np.zeros((*natural.shape, 8)) if jacobian else None
    for yi, value in enumerate(local):
        available = state + value
        prediction[yi] = np.cumsum((1 - r) * available)
        prediction[yi, 2:] -= x[7]
        if jacobian:
            local_jac = -available + (1 - r) * state_jac
            derivative[yi, :, :7] = np.tril(np.ones((7, 7))) * local_jac
            derivative[yi, 2:, 7] = -1
            state_jac = available + r * state_jac
        state = r * available
    return (prediction, derivative) if jacobian else prediction


def fit(natural, target, adjustment, eligible, weights, prior, spec, fixed_loss=None):
    """One parameter vector for the entire series; held observations are never read.

    Ridge centres are inherited historical fold parameters, not target-fitted values.
    Sharing is applied to the actual retention values, reducing routing degrees of freedom.
    """
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    prior = np.asarray(prior, dtype=float)
    if weights.shape != (len(target),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid year weights")
    mask = np.asarray(eligible, dtype=bool) & (weights[:, None] > 0)
    observed = target[mask]
    known = np.asarray(adjustment)[mask]
    if (
        not len(observed)
        or not np.isfinite(observed + known).all()
        or (observed <= 0).any()
        or spec["ridge"] < 0
    ):
        raise ValueError("Invalid calibration observations or penalty")
    groups = GROUPS[spec["sharing"]]
    ng = int(groups.max()) + 1
    expand = np.zeros((8, ng + int(fixed_loss is None)))
    expand[np.arange(7), groups] = 1
    if fixed_loss is None:
        expand[7, -1] = 1
    offset = np.zeros(8)
    if fixed_loss is not None:
        offset[7] = fixed_loss
    start = np.array([np.mean(prior[:7][groups == i]) for i in range(ng)])
    upper = np.full(ng, 0.5)
    if fixed_loss is None:
        start, upper = np.r_[start, prior[7]], np.r_[upper, 100.0]
    cell_weights = np.broadcast_to(weights[:, None], target.shape)[mask]
    scale = np.sqrt(cell_weights / cell_weights.sum()) / observed
    penalty = np.sqrt(spec["ridge"] / 8) / SCALES
    cache_v, cache = None, None

    def compute(values):
        nonlocal cache_v, cache
        if cache_v is None or not np.array_equal(cache_v, values):
            x = expand @ values + offset
            predicted, jac = forward(natural, x, True)
            residual = (predicted[mask] - known - observed) * scale
            derivative = (jac[mask] @ expand) * scale[:, None]
            cache = (
                np.r_[residual, penalty * (x - prior)],
                np.vstack([derivative, penalty[:, None] * expand]),
                derivative,
            )
            cache_v = values.copy()
        return cache

    starts = [start.copy() for _ in range(3)]
    starts[1][:ng], starts[2][:ng] = 0.1, 0.35
    fits = [
        least_squares(
            lambda v: compute(v)[0],
            value,
            jac=lambda v: compute(v)[1],
            bounds=(np.zeros_like(upper), upper),
            max_nfev=250,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        for value in starts
    ]
    converged = [item for item in fits if item.success and np.isfinite(item.fun).all()]
    if not converged:
        raise RuntimeError("No regularized routing start converged")
    best = min(converged, key=lambda item: item.cost)
    residual, jac, data_jac = compute(best.x)
    singular = np.linalg.svd(data_jac * upper, compute_uv=False)
    effective_df = np.trace(data_jac @ np.linalg.pinv(jac.T @ jac) @ data_jac.T)
    return dict(
        parameters=(expand @ best.x + offset).tolist(),
        success=True,
        n_free=len(best.x),
        fit_indices=np.flatnonzero(weights > 0).tolist(),
        observations=int(mask.sum()),
        objective=float(2 * best.cost),
        data_objective=float(np.sum(residual[: len(observed)] ** 2)),
        singular_values_scaled=singular.tolist(),
        condition_number=float(singular[0] / max(singular[-1], 1e-30)),
        local_effective_df=float(effective_df),
        starts=[
            dict(
                success=bool(item.success),
                nfev=int(item.nfev),
                objective=float(2 * item.cost),
                parameters=(expand @ item.x + offset).tolist(),
            )
            for item in fits
        ],
    )
